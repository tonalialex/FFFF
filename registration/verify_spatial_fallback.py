"""Regression: estimable transforms survive former quality gates."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from sar_registration.geometry import project, sane
from sar_registration.learned_matching import match_iteratively, reciprocal_candidates

def main():
    image=np.zeros((256,256),np.uint8)
    source=np.array([[80.,80.],[120.,80.],[80.,120.]])
    transform=np.eye(3);transform[:2,2]=[2.,1.]
    reference=project(source,transform)
    class Targets:
        ref_labels=np.ones(image.shape,np.int32)
        source_labels=np.ones(image.shape,np.int32)
        def filter_pairs(self,*args):raise AssertionError('Target rejection is disabled')
        def select(self,*args):raise AssertionError('Spatial thinning is disabled')
        def quality(self,*args):return dict(same_target_rate=0.,spatially_valid=False)
    seen=[]
    def describe(model,img,centers,batch):
        seen.append(np.asarray(centers).copy())
        return np.eye(len(centers),dtype=np.float32)
    with TemporaryDirectory() as tmp, patch('sar_registration.learned_matching.describe',side_effect=describe), patch('sar_registration.learned_matching.predictive_loo',return_value=999.), patch('sar_registration.learned_matching.robust_affine',return_value=(None,np.zeros(3,bool))):
        final,pairs,scores,mask,history=match_iteratively(None,image,source,reference,np.eye(3),Path(tmp),iterations=2,min_matches=10,max_matches=1,targets=Targets())
        assert history[0]['accepted'] and mask.sum()==3 and np.allclose(final,transform)
        assert history[0]['fit_method']=='weighted_least_squares_fallback'
        assert np.allclose(seen[2],reference), 'Estimated transform must feed next reference crops'
        assert history[0]['predictive_loo_rmse']==999.
    reflection=np.diag([-100.,.001,1.]);assert sane(reflection)
    singular=np.diag([0.,1.,1.]);assert not sane(singular)
    pairs,_=reciprocal_candidates(np.eye(3),np.eye(3),source,reference,radius=.01,min_cosine=2.,min_margin=2.,source_instances=np.ones(3),reference_instances=np.zeros(3))
    assert len(pairs)==3, 'Old radius, cosine, margin and target gates must be ignored'
    print('PASS: 3-point least-squares fallback, poor LOO/coverage/target quality accepted, transform feedback, extreme scales/reflection accepted, singular matrix rejected')

if __name__=='__main__':main()
