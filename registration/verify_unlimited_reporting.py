"""Regression: keep >24 real geometric inliers and export fixed-window NCC."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import cv2
import numpy as np
from sar_registration.geometry import project
from sar_registration.learned_matching import match_iteratively
from sar_registration.maritime_image_validation import MaritimeImageEvidence
from sar_registration.reporting import export_final, IterationRecorder, comparison

def main():
    rng=np.random.default_rng(72)
    sensed=rng.integers(0,256,(256,256),dtype=np.uint8)
    transform=np.eye(3);transform[:2,2]=[2.,1.]
    image=cv2.warpPerspective(sensed,transform,(256,256))
    xx,yy=np.meshgrid(np.linspace(80,176,7),np.linspace(80,176,7))
    source=np.column_stack([xx.ravel(),yy.ravel()])
    reference=project(source,transform)
    class Targets:
        ref_labels=np.ones(image.shape,np.int32)
        source_labels=np.ones(image.shape,np.int32)
        def filter_pairs(self,src,dst):return np.ones(len(src),bool)
        def select(self,*args):raise AssertionError('Unlimited mode must retain the entire geometric pool')
        def quality(self,*args):return dict(same_target_rate=1.,spatially_valid=True)
    windows=SimpleNamespace(active_ids=[1],ref_labels=np.ones(image.shape,np.int32),
        kind={1:'shore'},source_centers=source)
    evidence=MaritimeImageEvidence(image,sensed,np.eye(3),windows)
    with TemporaryDirectory(prefix='sar_unlimited_ncc_') as tmp, \
         patch('sar_registration.learned_matching.describe',side_effect=lambda model,img,centers,batch:np.eye(len(centers),dtype=np.float32)):
        root=Path(tmp)
        recorder=IterationRecorder(root/'result',image,sensed,ncc_evidence=evidence)
        final,pairs,scores,mask,history=match_iteratively(None,image,source,reference,np.eye(3),root/'matching',
            iterations=1,radius=40,threshold=1.3,min_matches=6,targets=Targets(),iteration_callback=recorder)
        assert history[0]['accepted'] and mask.sum()==49 and np.allclose(final,transform), history
        metrics=export_final(root/'result',image,sensed,final,source[pairs[:,0]],reference[pairs[:,1]],mask,
            'learned_update_accepted',scores,ncc_evidence=evidence)
        assert metrics['Nred']==49 and metrics['RMSEall']<1e-6
        expected=evidence.score(final)['holdout_ncc']
        assert np.isclose(metrics['NCC'],expected) and expected>.95
        assert expected>evidence.score(np.eye(3))['holdout_ncc']
        saved=json.loads((root/'result/metrics.json').read_text())
        assert saved['NCC']==metrics['NCC']
        assert 'NCC=' in (root/'result/registration_metrics.txt').read_text()
        assert 'NCC' in (recorder.directory/'iteration_metrics.csv').read_text()
        assert recorder.rows[0]['Nred']==49 and np.isclose(recorder.rows[0]['NCC'],expected)
        frontend=root/'frontend';frontend.mkdir()
        point_metrics={k:metrics[k] for k in ['Nred','RMSEall','RMSEloo','Pquad']}
        (frontend/'fusion_report.json').write_text(json.dumps(dict(representation_labels=['Original','round1','round2'],
            matches=[dict(label='Original',transform=final.tolist()),dict(label='round1',transform=np.eye(3).tolist())],
            sift_metrics={'Original':point_metrics,'round1':point_metrics,
                'round2':dict(Nred=0,RMSEall=None,RMSEloo=None,Pquad=None,status='sift_failed')})))
        rows=comparison(frontend,root/'result',metrics,ncc_evidence=evidence)
        assert rows[-1]['NCC']==expected
        assert rows[0]['NCC']==expected and rows[1]['NCC']<expected
        assert rows[2]['NCC'] is None
        saved_report=json.loads((frontend/'fusion_report.json').read_text())
        assert saved_report['sift_metrics']['Original']['NCC']==expected
    print(json.dumps(dict(passed=True,Nred=49,NCC=expected,
        checks=['unlimited_default_retains_all_49_geometric_inliers','geometry_and_quality_guards_pass',
        'NCC_matches_fixed_window_score','correct_alignment_improves_NCC',
        'final_and_iteration_json_text_csv_include_NCC']),indent=2))

if __name__=='__main__':main()
