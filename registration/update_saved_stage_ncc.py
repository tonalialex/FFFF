"""Backfill comparable SAR-SIFT NCC from saved images, transforms and target masks."""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,json,shutil
import cv2,numpy as np
from PIL import Image
from sar_registration.maritime_image_validation import MaritimeImageEvidence
from sar_registration.reporting import comparison

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def backup(path):
    path=Path(path)
    saved=path.with_name(path.stem+'.before_stage_ncc'+path.suffix)
    if path.exists() and not saved.exists():shutil.copy2(path,saved)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root',type=Path,required=True)
    args=parser.parse_args();root=args.batch_root.resolve()
    cv2.setNumThreads(4)
    rows=[];diagnostics=[]
    for summary_path in sorted(root.glob('*/pair_summary.json')):
        pair=summary_path.parent;summary=load(summary_path)
        frontend=next((pair/'registration/fusion_runs').glob('*/fusion_report.json')).parent
        output=pair/'registration/registration_results'/summary['pair']
        metrics=load(output/'metrics.json')
        ref=cv2.cvtColor(np.asarray(Image.open(summary['reference']).convert('RGB')),cv2.COLOR_RGB2GRAY)
        src=cv2.cvtColor(np.asarray(Image.open(summary['sensed']).convert('RGB')),cv2.COLOR_RGB2GRAY)
        instances=load(frontend/'maritime_targets/target_instances.json')
        masks=np.load(frontend/'maritime_targets/target_instances.npz',allow_pickle=False)
        targets=SimpleNamespace(active_ids=instances['active_ids'],ref_labels=masks['reference_labels'],
            kind={int(k):v for k,v in instances['kinds'].items()})
        bank=np.load(frontend/'patch_coordinates.npz',allow_pickle=False)
        evidence=MaritimeImageEvidence(ref,src,bank['T0'],targets)
        # Reproduce the OLD final NCC before editing derived stage reports.
        assert np.isclose(evidence.score(np.asarray(metrics['final_transform']))['holdout_ncc'],
                          metrics['NCC'],rtol=0,atol=1e-10), 'Saved final NCC does not match fixed-window reconstruction'
        for path in [frontend/'fusion_report.json',output/'four_stage_metrics.csv',output/'four_stage_metrics.json',
                     pair/'results/four_stage_metrics.csv',pair/'results/four_stage_metrics.json']:
            backup(path)
        stage_rows=comparison(frontend,output,metrics,ncc_evidence=evidence)
        for name in ['four_stage_metrics.csv','four_stage_metrics.json']:
            if (pair/'results').is_dir():shutil.copy2(output/name,pair/'results'/name)
        rows.extend([dict(pair=summary['pair'],**r) for r in stage_rows])
        history=load(frontend/'matching_iterations.json')
        detail=dict(pair=summary['pair'],batch_status=summary['status'],final_status=metrics['status'],
            final_Nred=metrics['Nred'],final_NCC=metrics['NCC'],registered_image_exists=(output/'registered_sensed.png').is_file(),
            sift_inliers={r['stage']:r['Nred'] for r in stage_rows[:-1]},iterations=[])
        for item in history:
            entry={k:item.get(k) for k in ['iteration','status','accepted','candidate_pairs','geometric_pool_count',
                'fit_pair_count','inliers','predictive_loo_rmse','loo_guard','supporter_guard','spatial_guard','target_quality']}
            entry['limits']=dict(minimum_geometric_inliers=6,loo_max_pixels=1.3*1.5,minimum_target_relative_coverage=.3)
            detail['iterations'].append(entry)
        diagnostics.append(detail)
        print(summary['pair'],[(r['stage'],r['NCC']) for r in stage_rows],flush=True)
    if not rows:raise ValueError('No saved pair summaries found')
    with (root/'batch_four_stage_metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'registration_failure_audit.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Updated stage NCC; final metrics, models, transforms and acceptance decisions were preserved.')

if __name__=='__main__':main()
