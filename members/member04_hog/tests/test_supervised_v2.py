import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from members.member04_hog.supervised_v2.common import (sample_weights, choose_threshold,
    prediction_rows, score_metrics, GROUPS)
from members.member04_hog.supervised_v2.train import fit, select_group
from members.member04_hog.supervised_v2.inference import predict


class SupervisedTests(unittest.TestCase):
    def test_balanced_weight_mass_and_label_encoding(self):
        y=np.array([0,0,0,1])
        w=sample_weights(y)
        self.assertAlmostEqual(w[y==0].sum(),2)
        self.assertAlmostEqual(w[y==1].sum(),2)
        with self.assertRaises(ValueError): sample_weights([0,0])
        rows=prediction_rows([SimpleNamespace(image_id='a',label='FAIL')],[.7],.7)
        self.assertEqual(rows[0]['predicted_label'],'FAIL')

    def test_threshold_balanced_accuracy_not_majority_f1(self):
        y=np.array([0,0,1,1,1,1]);scores=np.array([.1,.4,.2,.3,.8,.9])
        t=choose_threshold(y,scores)
        self.assertAlmostEqual(t,.15)
        with self.assertRaises(ValueError): choose_threshold([0,1],[0,np.nan])

    def test_validation_does_not_update_weights_and_reload(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(8,4)).astype(np.float32)
        y=np.array([0]*4+[1]*4);w=sample_weights(y)
        with threadpool_limits(limits=1):
            a,history,scores=fit(x,y,w,seed=42,epochs=2,variants=[x])
            b,_,_=fit(x+100,y,w,seed=42,epochs=2,variants=[x])
        for left,right in zip(a.coefs_,b.coefs_):np.testing.assert_array_equal(left,right)
        self.assertEqual(len(history),2)
        self.assertEqual(a.classes_.tolist(),[0,1])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.joblib';joblib.dump(a,path)
            np.testing.assert_array_equal(scores,joblib.load(path).predict_proba(x)[:,1])

    def test_constant_baselines_and_single_class(self):
        samples=[SimpleNamespace(image_id=str(i),label='FAIL' if i<77 else 'PASS') for i in range(101)]
        m=score_metrics(prediction_rows(samples,np.ones(101),.5))
        self.assertAlmostEqual(m['balanced_accuracy'],.5)
        self.assertAlmostEqual(m['f1'],154/178)
        self.assertEqual(m['false_fail'],24)
        self.assertIsNone(score_metrics(prediction_rows(samples[:1],[1],.5))['balanced_accuracy'])

    def test_same_settings_prefer_single_group(self):
        metric=[dict(balanced_accuracy=.9,false_pass=2)]*3
        self.assertEqual(select_group({'C_without_clahe':metric,'A2':metric}),'A2')
        self.assertEqual(GROUPS['A2'],GROUPS['C_without_clahe'])
        self.assertEqual(GROUPS['B2'],GROUPS['C_without_geometry'])
        self.assertEqual(len(GROUPS),11)

    def test_missing_model_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            r=predict('missing.png',Path(temp)/'missing.joblib')
            self.assertEqual(r['status'],'MODEL_UNAVAILABLE')
            self.assertIsNone(r['predicted_label'])
