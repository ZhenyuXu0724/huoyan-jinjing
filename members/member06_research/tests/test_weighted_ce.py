"""Synthetic-only optimizer tests; historical checkpoint read is inference-only."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from members.member06_research.config import Config
from members.member06_research.train import class_weights, run_epoch
from members.member06_research.weighted_dev import config_diff, verify_archive, ARCHIVE, read
from members.member06_research.compare_dev_runs import metrics, inspect_run, compare

class WeightedTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_config(self):
        self.assertEqual(Config().train_class_weight,'none')
        with self.assertRaises(ValueError):Config(train_class_weight='manual')
        a=Config().to_dict();b={**a,'train_class_weight':'balanced'}
        self.assertEqual(list(config_diff(a,b)),['train_class_weight'])
        with self.assertRaises(ValueError):config_diff(a,{**b,'batch_size':2})
    def test_weights_order_and_missing(self):
        rows=[{'label':'PASS'}]*133+[{'label':'FAIL'}]*52
        c=Config();self.assertTrue(torch.allclose(class_weights(rows,c,'cpu'),torch.tensor([185/266,185/104])))
        c=Config(class_to_idx={'FAIL':0,'PASS':1})
        self.assertTrue(torch.allclose(class_weights(rows,c,'cpu'),torch.tensor([185/104,185/266])))
        with self.assertRaises(ValueError):class_weights(rows[:2],c,'cpu')
    def test_loss_and_unweighted_regression(self):
        x=torch.tensor([[2.,0.],[1.,0.],[0.,2.]]);y=torch.tensor([0,1,0]);loader=[(x[:2],y[:2]),(x[2:],y[2:])]
        m=nn.Linear(2,2,bias=False);m.weight.data.copy_(torch.eye(2))
        stats,p=run_epoch(m,loader,'cpu',1)
        self.assertAlmostEqual(stats['loss'],nn.CrossEntropyLoss()(x,y).item(),places=6)
        criterion=nn.CrossEntropyLoss(weight=torch.tensor([.6,1.8]))
        with self.assertRaises(ValueError):run_epoch(m,loader,'cpu',1,criterion=criterion)
        # lr=0 isolates reduction check on synthetic tensors; real optimization covered below.
        stats,_=run_epoch(m,loader,'cpu',1,torch.optim.SGD(m.parameters(),lr=0),criterion=criterion)
        self.assertAlmostEqual(stats['loss'],criterion(x,y).item(),places=6)
        before=m.weight.detach().clone()
        run_epoch(m,loader,'cpu',1,torch.optim.SGD(m.parameters(),lr=.1),criterion=criterion)
        self.assertFalse(torch.equal(before,m.weight))
        val,_=run_epoch(m,loader,'cpu',1)
        self.assertAlmostEqual(val['loss'],nn.CrossEntropyLoss()(m(x),y).item(),places=6)
    def test_archive_and_old_checkpoint(self):
        self.assertGreater(verify_archive(),0)
        run=inspect_run(ARCHIVE/'run');self.assertEqual(run['metrics']['best_epoch'],5)
        self.assertEqual(run['metrics']['false_pass'],73)
    def test_missing_results_no_output(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'report'
            with self.assertRaisesRegex(ValueError,'工件缺失'):compare(ARCHIVE/'run',Path(d)/'absent',out)
            self.assertFalse(out.exists())
    def test_exact_historical_unweighted_behavior(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('members.member06_research._historical_train',ARCHIVE/'source/train.py')
        historical=importlib.util.module_from_spec(spec);spec.loader.exec_module(historical)
        torch.manual_seed(42)
        a=nn.Linear(3,2);b=copy.deepcopy(a)
        data=[(torch.randn(4,3),torch.tensor([0,1,0,0])),(torch.randn(3,3),torch.tensor([1,0,1]))]
        x,p=historical.run_epoch(a,data,'cpu',1,torch.optim.Adam(a.parameters(),lr=.001))
        y,q=run_epoch(b,data,'cpu',1,torch.optim.Adam(b.parameters(),lr=.001))
        self.assertEqual(x,y);self.assertEqual(p,q)
        for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]))
    def test_weighted_fit_synthetic(self):
        from members.member06_research.smoke_data import generate
        from members.member06_research.train import fit
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);generate(root/'data')
            c=Config(weights='RANDOM_SMOKE_ONLY',data_kind='SMOKE_ONLY',image_size=64,
                     train_manifest=str(root/'data/train.csv'),val_manifest=str(root/'data/val.csv'),
                     epochs=1,batch_size=4,train_class_weight='balanced')
            fit(c,root/'run')
            payload=torch.load(root/'run/best.pt',weights_only=True)
            self.assertEqual(payload['run_metadata']['loss_protocol']['validation'],'unweighted CE mean')
            self.assertEqual(payload['config']['train_class_weight'],'balanced')
            self.assertEqual(payload['val_metrics'],read(root/'run/history.json')[0]['val'])
    def test_shell_preserves_training_exit_and_unique_directories(self):
        import subprocess, shutil
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'members/member06_research'
            (base/'scripts').mkdir(parents=True);(base/'.venv/bin').mkdir(parents=True)
            script=base/'scripts/run_weighted_dev.sh'
            shutil.copy2(Path(__file__).parents[1]/'scripts/run_weighted_dev.sh',script)
            fake=base/'.venv/bin/python'
            fake.write_text('#!/usr/bin/env bash\ncase "$*" in *member06_research.train*) echo synthetic-failure; exit 7;; *) exit 0;; esac\n')
            fake.chmod(0o755)
            for _ in range(2):
                ret=subprocess.run(['bash',str(script)],capture_output=True,text=True)
                self.assertEqual(ret.returncode,7,ret.stderr)
            runs=list((base/'runs').iterdir());self.assertEqual(len(runs),2)
            for run in runs:
                self.assertEqual((run/'train_exit_code.txt').read_text().strip(),'7')
                self.assertIn('synthetic-failure',(run/'console.log').read_text())
                self.assertFalse((run/'comparison').exists())
    def test_auc_ties(self):
        rows=[{'target_index':i,'predicted_index':0,'softmax':[.5,.5]} for i in [0,1]]
        self.assertEqual(metrics(rows)['roc_auc'],.5)
    def test_comparison_synthetic_report_and_no_overwrite(self):
        # No fake real-run files: only mocked in-memory records in a temporary test directory.
        old=inspect_run(ARCHIVE/'run');new=copy.deepcopy(old)
        new['config']['train_class_weight']='balanced'
        new['metadata']['environment']['source_sha256'].update(read(Path(__file__).parents[1]/'configs/weighted_ce_code_identity.json'))
        new['metadata']['loss_protocol']={'train_class_weight':'balanced','validation':'unweighted CE mean','weights':[185/266,185/104]}
        with tempfile.TemporaryDirectory() as d, patch('members.member06_research.compare_dev_runs.inspect_run',side_effect=lambda p: old if str(p)=='old' else new):
            out=Path(d)/'report';compare('old','new',out)
            self.assertTrue((out/'COMPARISON_CN.md').exists())
            self.assertIn('C：',read(out/'comparison_metrics.json')['decision'])
            with self.assertRaisesRegex(ValueError,'already exists'):compare('old','new',out)
            new['metadata']['datasets']['val']['manifest_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'identity mismatch'):compare('old','new',Path(d)/'invalid')

if __name__=='__main__':unittest.main()
