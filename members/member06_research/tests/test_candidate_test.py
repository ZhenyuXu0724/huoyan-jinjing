"""Synthetic fixtures only: no real Test pixels, labels, or optimizer training."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from members.member06_research.config import Config
from members.member06_research.model import FrozenResNet18
from members.member06_research.runtime import file_sha256,write_json
from members.member06_research.data import image_digest
from members.member06_research.candidate_test import evaluate,protocol_check,eligibility,safe_path,summarize

class CandidateTestTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.protocol=self.root/'protocol.json';self.handoff=self.root/'handoff.json'
        c=Config(weights='RANDOM_SMOKE_ONLY',data_kind='SMOKE_ONLY',image_size=32)
        model=FrozenResNet18(c.weights)
        dev={'train':{'images':[{'path':'/synthetic_train.png','sha256_rgb':'a'*64}]},'val':{'images':[{'path':'/synthetic_val.png','sha256_rgb':'b'*64}]}}
        self.p={'schema':'member06.candidate-test.v1','kind':'SMOKE_ONLY','primary':'v2','control':'v1','decision':'argmax; ties PASS=0','test_used_for_selection':False,'expected_counts':[2]*5,'source_sha256':{},'known_exposed_sha256':[],'models':{},'device':'cpu','latency':{'warmup':10}}
        for name in ['v1','v2']:
            cfg=c.to_dict();cfg['train_class_weight']='balanced' if name=='v2' else 'none'
            payload={'format_version':1,'architecture':'frozen_resnet18_linear','config':cfg,'evidence_status':'SMOKE_ONLY','epoch':5,'model_state':copy.deepcopy(model.state_dict()),'run_metadata':{'test_data_used':False,'datasets':dev}}
            if name=='v2':payload['model_state']['head.bias'][1]+=.1
            path=self.root/f'{name}.pt';torch.save(payload,path)
            self.p['models'][name]={'checkpoint':str(path),'sha256':file_sha256(path),'config':cfg,'epoch':5}
        rows=[];labels=[]
        for i in range(4):
            path=self.root/f'i{i}.png';Image.new('RGB',(40+i,35),(i*50,100,200)).save(path)
            rows.append({'image_id':str(i),'image_path':path.name,'image_sha256':file_sha256(path),'rgb_sha256':image_digest(path),'physical_scene_id':f'new-{i}','source_batch':'synthetic','capture_condition':'synthetic'})
            labels.append({'image_id':str(i),'label':'PASS' if i%2==0 else 'FAIL','anomaly_type':'none' if i%2==0 else 'unknown','non_catalogue_count':i%2})
        self.m={'kind':'SMOKE_ONLY','contains_gt':False,'images':rows,'development_scene_ledger':[{'image_path':'/synthetic_train.png','physical_scene_id':'dev-1'},{'image_path':'/synthetic_val.png','physical_scene_id':'dev-2'}]}
        write_json(self.root/'labels.json',{'images':labels})
        self.h={'status':'APPROVED','scope':'INDEPENDENT_HELD_OUT_CANDIDATE_EVALUATION','kind':'SMOKE_ONLY','custodian':'synthetic-only-A','independent_reviewer':'synthetic-only-B','decision_reference':'TEMPORARY_TEST_FIXTURE','dataset_version':'synthetic','approved_at':'synthetic','unsealed_access_history':'synthetic fixture only','expected_counts':[2]*5,'dataset_root':str(self.root),'metadata_file':'metadata.json','labels_file':'labels.json','labels_sha256':file_sha256(self.root/'labels.json'),'n':4,'class_support':{'PASS':2,'FAIL':2}}
        self.h.update({k:True for k in ['labels_reviewed','physical_scene_separation_verified','never_used_for_development','sealed_until_binding','fixed_counts_confirmed','dev_candidate_limitations_accepted']})
        self.save()
    def save(self):
        write_json(self.protocol,self.p);self.protocol.with_suffix('.sha256').write_text(file_sha256(self.protocol))
        self.h['protocol_sha256']=file_sha256(self.protocol)
        write_json(self.root/'metadata.json',self.m);self.h['metadata_sha256']=file_sha256(self.root/'metadata.json');write_json(self.handoff,self.h)
    def run_mode(self,mode):return evaluate(self.protocol,self.handoff,mode=mode,allow_smoke=True)
    def test_metadata_check_and_bind_do_not_open_test_content(self):
        (self.root/'labels.json').write_text('NOT VALID JSON')
        with patch('members.member06_research.candidate_test.image_digest',side_effect=AssertionError('decoded test')):
            self.assertFalse(self.run_mode('check')['test_content_read']);self.assertFalse(self.run_mode('bind')['test_content_read'])
        self.assertFalse((self.root/'results').exists())
    def test_synthetic_dual_execution_and_single_claim(self):
        self.run_mode('bind');result=self.run_mode('execute');self.assertEqual(result['status'],'SMOKE_ONLY')
        import json
        data=json.loads((self.root/'results/metrics.json').read_text())
        for model in ['v1','v2']:
            self.assertEqual(data['models'][model]['n'],4)
            self.assertEqual(sum(map(sum,data['models'][model]['confusion_matrix'])),4)
            self.assertGreater(data['models'][model]['latency_ms']['median'],0)
        self.assertTrue((self.root/'execution_complete.json').exists())
        with self.assertRaises(FileExistsError):self.run_mode('execute')
    def test_unapproved_fails_before_content(self):
        self.h['status']='PENDING_NOT_APPROVED';self.save()
        with patch('members.member06_research.candidate_test.image_digest',side_effect=AssertionError('opened')):
            with self.assertRaisesRegex(ValueError,'handoff absent'):self.run_mode('execute')
        self.assertFalse((self.root/'execution_claim.json').exists())
    def test_requires_binding(self):
        with self.assertRaises(FileNotFoundError):self.run_mode('execute')
        self.assertFalse((self.root/'execution_claim.json').exists())
    def test_scene_and_pixel_overlap(self):
        self.m['images'][0]['physical_scene_id']='dev-1';self.save()
        with self.assertRaisesRegex(ValueError,'scene overlap'):self.run_mode('check')
        self.m['images'][0]['physical_scene_id']='new-0';self.m['images'][0]['rgb_sha256']='a'*64;self.save()
        with self.assertRaisesRegex(ValueError,'pixel overlap'):self.run_mode('check')
    def test_exposure_and_incomplete_ledger(self):
        self.p['known_exposed_sha256']=[self.m['images'][0]['image_sha256']];self.save()
        with self.assertRaisesRegex(ValueError,'exposed'):self.run_mode('check')
        self.p['known_exposed_sha256']=[];self.m['development_scene_ledger'].pop();self.save()
        with self.assertRaisesRegex(ValueError,'complete development'):self.run_mode('check')
    def test_protocol_and_checkpoint_tamper(self):
        self.protocol.write_text(self.protocol.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'protocol changed'):self.run_mode('check')
        self.save();(self.root/'v1.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):self.run_mode('check')
    def test_corrupt_image_aborts_without_exclusion(self):
        self.run_mode('bind');(self.root/'i0.png').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'image hash mismatch'):self.run_mode('execute')
        self.assertTrue((self.root/'execution_failed.json').exists())
        self.assertFalse((self.root/'results/metrics.json').exists())
    def test_gt_not_in_metadata_and_single_class_rejected(self):
        self.m['images'][0]['label']='PASS';self.save()
        with self.assertRaisesRegex(ValueError,'no labels'):self.run_mode('check')
        self.m['images'][0].pop('label');self.h['class_support']={'PASS':4,'FAIL':0};self.save()
        with self.assertRaisesRegex(ValueError,'both class'):self.run_mode('check')
    def test_path_and_duplicate_rejection(self):
        with self.assertRaises(ValueError):safe_path(self.root,'../escape')
        (self.root/'link').symlink_to(self.root/'labels.json')
        with self.assertRaises(ValueError):safe_path(self.root,'link')
        self.m['images'][1]['image_id']='0';self.save()
        with self.assertRaisesRegex(ValueError,'duplicate image_id'):self.run_mode('check')
    def test_smoke_cannot_be_promoted(self):
        with self.assertRaisesRegex(ValueError,'SMOKE_ONLY'):protocol_check(self.protocol)
        self.p['kind']='REAL';self.save()
        with self.assertRaisesRegex(ValueError,'kind mismatch'):self.run_mode('check')
    def test_metric_values_and_missing_class(self):
        rows=[{'target_index':t,'predicted_index':d,'softmax':[1-s,s]} for t,d,s in [(0,0,.1),(0,1,.6),(1,0,.4),(1,1,.9)]]
        m=summarize(rows,[1,2,3,4]);self.assertEqual(m['confusion_matrix'],[[1,1],[1,1]]);self.assertEqual(m['roc_auc'],.75)
        self.assertEqual(m['balanced_accuracy'],.5);self.assertEqual(m['latency_ms']['median'],2.5);self.assertAlmostEqual(m['latency_ms']['p95'],3.85)
        self.assertIsNone(summarize(rows[:2],[1,2])['roc_auc'])

if __name__=='__main__':unittest.main()
