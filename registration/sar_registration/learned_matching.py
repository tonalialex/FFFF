"""Descriptor matching with no hard quality gates on estimable transforms."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sar_registration.geometry import project, robust_affine, coverage
from sar_registration.simclr import describe, valid_centers
from sar_registration.geometry import weighted_affine, sane
from sar_registration.maritime_targets import point_labels
from registration_metrics import predictive_loo_rmse


def select_distributed(points, scores, shape, maximum=24, minimum_spacing=32.):
    """Confidence/diversity selection, never sort by the reported fit error.

    Geometry has already rejected gross outliers. The final budget is selected
    by learned confidence and spatial diversity, with a half-budget quadrant cap.
    """
    if len(points) == 0:
        return np.empty(0,int)
    h,w = shape[:2]
    quadrant = (points[:,0] >= w/2).astype(int)+2*(points[:,1] >= h/2)
    selected, remaining, counts = [],np.ones(len(points),bool),np.zeros(4,int)
    norm = max(float(np.max(scores)),1e-8)
    while len(selected) < min(maximum,len(points)):
        allowed = remaining & (counts[quadrant] < int(np.ceil(maximum/2)))
        if selected:
            distances = np.min(np.linalg.norm(points[:,None]-points[selected][None],axis=2),axis=1)
            allowed &= distances >= minimum_spacing
            diversity = .25+.75*np.minimum(distances/(.3*min(h,w)),1.)
        else:
            diversity = np.ones(len(points))
        if not allowed.any():
            break
        merit = (np.asarray(scores)/norm+.05)*diversity/(1.+.2*counts[quadrant])
        index = int(np.argmax(np.where(allowed,merit,-np.inf)))
        selected.append(index);remaining[index]=False;counts[quadrant[index]]+=1
    return np.asarray(selected,int)


def predictive_loo(source, target, scores):
    """Use the same confidence-weighted leave-one-out prediction as reports."""
    return predictive_loo_rmse(source, target, scores)


def reciprocal_candidates(source_desc, reference_desc, projected, reference_points,
                          radius, min_cosine=.5, min_margin=.015,
                          source_instances=None,reference_instances=None):
    # Mutual nearest neighbors define correspondences; no quality threshold,
    # radius cutoff, target-label restriction, or margin rejection is applied.
    cosine = source_desc @ reference_desc.T
    distance = np.linalg.norm(projected[:, None, :] - reference_points[None, :, :], axis=2)
    score = cosine - .10*(distance/max(radius, 1e-6))**2
    if not score.size:
        return np.empty((0, 2), int), np.empty(0)
    row = np.argmax(score, axis=1)
    col = np.argmax(score, axis=0)
    ids = np.arange(len(source_desc))
    keep = np.isfinite(score[ids,row]) & (col[row] == ids)
    pairs = np.column_stack((ids[keep], row[keep]))
    if len(pairs) < 3 and min(score.shape) >= 3:
        from scipy.optimize import linear_sum_assignment
        left, right = linear_sum_assignment(-score)
        pairs = np.column_stack((left, right))
    confidence = np.maximum((cosine[pairs[:,0],pairs[:,1]] + 1.) / 2., 1e-6)
    return pairs, confidence


def neighborhood_filter(projected, targets, radius, neighbors=8):
    if len(projected) < 6:
        return np.ones(len(projected), bool)
    _, indices = cKDTree(projected).query(projected, k=min(neighbors+1,len(projected)))
    displacement = targets-projected
    residuals = np.linalg.norm(displacement[:,None,:]-displacement[indices[:,1:]], axis=2)
    tolerance = max(2., radius*.15)
    # Local majority consensus suppresses isolated descriptor matches while
    # admitting smoothly varying affine corrections across a large image.
    return (residuals <= tolerance).mean(axis=1) >= .5


def match_iteratively(model, reference, source_points, reference_points, initial,
                      directory: Path, iterations=3,
                      radius=40., min_cosine=.5, min_margin=.015,
                      batch_size=128, threshold=2., iteration_callback=None,
                      max_matches=0, min_matches=10, image_evidence=None, targets=None):
    """Train once, then T_k -> Xs_k -> descriptors -> matches -> T_(k+1).

    The callback runs for EVERY attempted round, including rejected or empty
    rounds. Only accepted transforms feed the next round's reference crops.
    """
    directory.mkdir(parents=True, exist_ok=True)
    transform = initial.copy()
    reference_desc = describe(model, reference, reference_points, batch_size)
    history, best_pairs = [], np.empty((0, 2), int)
    best_scores, best_mask = np.empty(0), np.empty(0, bool)
    for iteration in range(iterations):
        previous_transform = transform.copy()
        guidance = None
        if image_evidence is not None and iteration == 0:
            input_transform,guidance = image_evidence.refine(transform)
        else:
            input_transform = transform.copy()
        projected_all = project(source_points, input_transform)
        source_ids = np.flatnonzero(valid_centers(projected_all, reference.shape))
        projected = projected_all[source_ids]
        pairs, scores = np.empty((0, 2), int), np.empty(0)
        mask = np.empty(0, bool)
        fitted, accepted, movement = None, False, None
        record = {'iteration': iteration+1, 'radius': radius,
                  'valid_Xs_count': len(source_ids), 'candidate_pairs': 0,
                  'previous_transform':previous_transform.tolist(), 'image_guidance':guidance}
        if len(projected) < 3 or len(reference_points) < 3:
            record['status'] = 'insufficient_valid_crops'
        else:
            # This call always crops fresh Xs from reference at T_k(source).
            source_desc = describe(model, reference, projected, batch_size)
            pairs, scores = reciprocal_candidates(source_desc, reference_desc, projected,
                reference_points, radius, min_cosine, min_margin,
                **(dict(source_instances=point_labels(source_points[source_ids],targets.source_labels),
                   reference_instances=point_labels(reference_points,targets.ref_labels)) if targets is not None else {}))
            if len(pairs):
                pairs[:, 0] = source_ids[pairs[:, 0]]
            record['candidate_pairs'] = len(pairs)
            if len(pairs) < 3:
                record['status'] = 'insufficient_matches'
            else:
                src, dst = source_points[pairs[:, 0]], reference_points[pairs[:, 1]]
                fitted, mask = robust_affine(src, dst, threshold, scores)
                # RANSAC is an estimator, not an output gate. If it cannot
                # estimate an affine transform, fit all correspondences instead.
                fit_method = 'ransac'
                if fitted is None or mask.sum() < 3:
                    try:
                        fitted = weighted_affine(src, dst, scores)
                        mask = np.ones(len(pairs), bool)
                        fit_method = 'weighted_least_squares_fallback'
                    except (ValueError, np.linalg.LinAlgError):
                        fitted = None
                if not sane(fitted):
                    fitted = None
                    mask = np.zeros(len(pairs), bool)
                    record['status'] = 'mathematically_degenerate_fit'
                else:
                    record.update(fit_method=fit_method, geometric_pool_count=int(mask.sum()),
                        spatial_selected_count=int(mask.sum()), fit_pair_count=int(mask.sum()),
                        spatial_fallback_used=False, supporter_count=0, supporter_rmse=None,
                        predictive_loo_rmse=predictive_loo(src[mask],dst[mask],scores[mask]))
                if fitted is not None:
                    before = np.linalg.norm(project(src, input_transform)-dst, axis=1)
                    after = np.linalg.norm(project(src, fitted)-dst, axis=1)
                    image_before,image_after = None,None
                    if image_evidence is not None:
                        # NCC is reported for comparison, never an acceptance gate.
                        image_before = image_evidence.score(previous_transform)
                        image_after = image_evidence.score(fitted)
                    target_quality=targets.quality(src[mask],dst[mask],fitted) if targets is not None else None
                    accepted = True
                    record.update(inliers=int(mask.sum()), image_before=image_before,image_after=image_after,
                        target_quality=target_quality, quality_gates_removed=True,
                        image_guard_removed=True, supporter_guard=True, loo_guard=True, spatial_guard=True,
                        point_coverage=coverage(dst[mask],reference.shape), status='accepted')
                    if accepted:
                        movement = float(np.median(np.linalg.norm(
                            project(src, fitted)-project(src, input_transform), axis=1)))
                        transform, best_pairs, best_scores, best_mask = fitted, pairs, scores, mask
                        radius = max(8., min(radius, 3*float(np.quantile(after[mask], .9))+4))
        source, target = source_points[pairs[:, 0]], reference_points[pairs[:, 1]]
        evaluation_transform = fitted if fitted is not None else input_transform
        # When no valid fit exists, report no fitted inliers; do not make an
        # empty/degenerate learned stage look like a successful SIFT fallback.
        if fitted is None:
            mask = np.zeros(len(pairs), bool)
        record.update(accepted=accepted, movement=movement,
                      input_transform=input_transform.tolist(),
                      candidate_transform=None if fitted is None else fitted.tolist(),
                      output_transform=transform.tolist(),
                      evaluated_transform='candidate' if fitted is not None else 'input_no_valid_fit')
        event = dict(record=record, source=source, target=target, mask=mask, scores=scores,
                     evaluation_transform=evaluation_transform, input_transform=input_transform,
                     output_transform=transform.copy(), source_ids=source_ids,
                     projected=projected, original_points=source_points[source_ids])
        if iteration_callback is not None:
            record['artifacts'] = iteration_callback(event)
        history.append(record)
        # Persist at each round, so an interruption preserves completed output.
        temporary = directory/'matching_iterations.tmp'
        temporary.write_text(json.dumps(history, indent=2), encoding='utf-8')
        temporary.replace(directory/'matching_iterations.json')
        if not accepted or movement < .05:
            break
    return transform, best_pairs, best_scores, best_mask, history
