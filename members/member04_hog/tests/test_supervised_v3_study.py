"""Meaningful invariants for placement, constrained selection and learning isolation."""
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from members.member04_hog.supervised_v3_study.stability import variants
from members.member04_hog.supervised_v3_study.common import threshold_budget,checkpoint_key,sample_weights
from members.member04_hog.supervised_v3_study.train import fit

class StudyTests(unittest.TestCase):
    def test_no_crop_and_same_scale_position(self):
        a=np.arange(256*256,dtype=np.float32).reshape(256,256)/(256*256)
        v=dict(variants(a,127));self.assertEqual(len(v),21);np.testing.assert_array_equal(v['original'],a)
        for s in (224,240):
            gap=256-s;c=gap//2
            center=v[f's{s}_white_center'][c:c+s,c:c+s]
            np.testing.assert_array_equal(center,v[f's{s}_white_left'][c:c+s,:s])
            np.testing.assert_array_equal(center,v[f's{s}_white_right'][c:c+s,gap:])
            np.testing.assert_array_equal(center,v[f's{s}_white_top'][:s,c:c+s])
            np.testing.assert_array_equal(center,v[f's{s}_white_bottom'][gap:,c:c+s])
            np.testing.assert_array_equal(center,v[f's{s}_border_center'][c:c+s,c:c+s])

    def test_budget_ties_and_all_pass(self):
        r=threshold_budget([0,0,1,1],[.9,.1,.8,.2],0)
        self.assertEqual((r['fn'],r['fp']),(2,0));self.assertGreater(r['threshold'],.9)
        r=threshold_budget([0,0,1,1],[.9,.1,.8,.2],1)
        self.assertEqual((r['fn'],r['fp']),(0,1))
        r=threshold_budget([0,1],[0,0],0);self.assertEqual((r['fn'],r['fp']),(1,0))
        with self.assertRaises(ValueError):threshold_budget([0,1],[np.nan,1])
        self.assertLess(checkpoint_key(dict(fn=1,fp=2,epoch=3)),checkpoint_key(dict(fn=1,fp=2,epoch=5)))

    def test_validation_no_gradient_and_saved_best_is_independent(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(12,5)).astype(np.float32);y=np.array([0]*6+[1]*6)
        with threadpool_limits(limits=1):
            a,last,best,h,scores=fit(x,y,sample_weights(y),x,y,layers=(4,2),alpha=.01,seed=42,epochs=4)
            b,last2,_,_,_=fit(x,y,sample_weights(y),x+100,y,layers=(4,2),alpha=.01,seed=42,epochs=4)
        for left,right in zip(last.coefs_,last2.coefs_):np.testing.assert_array_equal(left,right)
        np.testing.assert_allclose(a.predict_proba(x)[:,1],scores[best['epoch']-1],rtol=1e-5,atol=1e-7)
        for left,right in zip(a.coefs_,last.coefs_):self.assertFalse(np.shares_memory(left,right))
        self.assertTrue(all(row['fp']<=5 for row in h))

if __name__=='__main__':unittest.main()
