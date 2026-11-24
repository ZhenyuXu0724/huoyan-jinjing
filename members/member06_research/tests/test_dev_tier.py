"""DEV_ONLY safety and grouped metrics; every input is synthetic in temporary directories."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from members.member06_research.tests import test_development_workflow as fixtures
write = fixtures.write
from members.member06_research import development, corpus_audit, experiments
from members.member06_research.config import Config
from members.member06_research.preflight import check_training_inputs
from members.member06_research.train import fit
from members.member06_research.runtime import load_checkpoint, file_sha256
from members.member06_research.adapter import ClassificationAdapter
from members.member06_research.group_metrics import decision_metrics, grouped_validation


class DevTests(unittest.TestCase):
    setUp = fixtures.DevelopmentTests.setUp
    write = staticmethod(write)

    def prepared(self):
        development.prepare_dev(self.sources, self.base/'dev')
        return Config.load(self.base/'dev/config.json')

    def test_dev_prepare_preflight_preserves_limits_and_no_test_access(self):
        original = Path.open
        def guarded(path, *a, **kw):
            if 'test' in path.parts:
                raise AssertionError('test read during DEV preparation')
            return original(path, *a, **kw)
        with patch.object(Path, 'open', guarded):
            c = self.prepared()
            train, val, r = check_training_inputs(c)
        self.assertEqual(r['status'], 'DEV_ONLY')
        self.assertEqual(r['formal_readiness'], 'NOT_READY')
        self.assertFalse(r['weights_updated'])
        self.assertFalse(r['scene_independence_verified'])
        self.assertTrue(all('anomaly_type' in row for row in train.rows+val.rows))
        self.assertTrue(all(row['scene_id'].startswith('UNVERIFIED:') for row in train.rows))

    def test_fit_requires_opt_in_before_model_or_output(self):
        c=self.prepared()
        with patch('members.member06_research.train.FrozenResNet18', side_effect=AssertionError('model initialized')):
            with self.assertRaisesRegex(ValueError, 'allow-dev-only'):
                fit(c,self.base/'run')
        self.assertFalse((self.base/'run').exists())

    def test_config_cannot_promote_development_data(self):
        c=self.prepared();c.training_tier='FORMAL_READY';c.counts_rule_status='CONFIRMED'
        with self.assertRaisesRegex(ValueError,'cannot be promoted'):
            check_training_inputs(c)

    def test_manifest_tamper_rejected(self):
        c=self.prepared()
        with Path(c.train_manifest).open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'manifest changed'):development.verify_dev(c)

    def test_known_scene_overlap_and_unknown_annotation_rejected(self):
        r=development.audit(self.sources)
        ledger={'schema_version':'member06.scene-ledger.v1','images':[
            {'image_path':row['image_path'],'scene_id':'same_scene'} for row in r['images']]}
        with self.assertRaisesRegex(ValueError,'known scene crosses'):development.dev_rows(r,[1]*5,ledger)
        r['images'][0]['unresolved_ignore_or_judgment']=True
        with self.assertRaisesRegex(ValueError,'unresolved'):development.dev_rows(r,[1]*5)

    def test_formal_truth_export_refuses_dev(self):
        from members.member06_research.handoff import export
        self.prepared()
        with self.assertRaisesRegex(ValueError,'DEV_ONLY'):export(self.base/'dev/config.json',self.base,self.base/'out')

    def test_checkpoint_and_freeze_require_correct_evidence(self):
        c=Config(training_tier='DEV_ONLY')
        payload={'format_version':1,'architecture':'frozen_resnet18_linear','config':c.to_dict(),'evidence_status':'DEV_ONLY'}
        with patch('members.member06_research.runtime.torch.load',return_value=payload):
            with self.assertRaisesRegex(ValueError,'allow-dev-only'):load_checkpoint('synthetic.pt')
        p=self.base/'selection.json';write(p,{'schema_version':'member06.selection.v1','test_used':False,'selected':{'evidence_status':'DEV_ONLY'}})
        with self.assertRaisesRegex(ValueError,'DEV_ONLY'):experiments.freeze(p,self.base/'freeze.json')

    def test_flat_labelme_hash_count_and_inventory(self):
        root=self.base/'flat';root.mkdir();image=root/'sample.png'
        Image.new('RGB',(30,20),'white').save(image)
        write(image.with_suffix('.json'),{'imagePath':image.name,'imageWidth':30,'imageHeight':20,'shapes':[
            {'label':'outside','shape_type':'rectangle','points':[[1,1],[5,5]],'flags':{}}]})
        index={'schema_version':'member06.flat-labelme.v1','root':str(root),'split':'train','files':{
            p.name:{'sha256':file_sha256(p),'bytes':p.stat().st_size} for p in root.iterdir()}}
        p=self.base/'flat.json';write(p,index)
        rows,errors=corpus_audit.audit_flat(p,{'outside':0})
        self.assertFalse(errors);self.assertEqual(rows[0]['actual_counts']['0'],1)
        image.write_bytes(b'corrupt')
        self.assertTrue(corpus_audit.audit_flat(p,{'outside':0})[1])

    def test_proposed_mapping_records_history_without_approving_reuse(self):
        import csv
        r=corpus_audit.audit_corpus(self.sources)
        self.assertFalse(r['screenshot_split_verified'])
        entries=[{'image_id':x['image_id'],'split':x['split'],'label':x['annotation_label']} for x in r['images']]
        target=next(x for x in entries if x['split']=='val');target['split']='train'
        p=self.base/'proposal.csv'
        def save():
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['image_id','split','label']);w.writeheader();w.writerows(entries)
        save();r=corpus_audit.audit_corpus(self.sources,proposed_split=p)
        self.assertTrue(r['role_transition_review_required'])
        self.assertFalse(r['proposed_split_training_approved'])
        self.assertEqual(r['role_transitions'][0]['from'],'val')
        target['label']='FAIL' if target['label']=='PASS' else 'PASS'
        save();r=corpus_audit.audit_corpus(self.sources,proposed_split=p)
        self.assertTrue(r['errors']);self.assertFalse(r['screenshot_split_verified'])

    def test_public_test_requires_explicit_audit_and_is_never_in_dev(self):
        from members.member06_research.team_data import audit_dataset
        self.assertFalse(audit_dataset(self.root,self.catalogue_path)['errors'])
        r=audit_dataset(self.root,self.catalogue_path,include_public_test=True)
        self.assertTrue(r['errors']);self.assertTrue(r['test_content_read'])


class GroupTests(unittest.TestCase):
    def test_coverage_and_undefined_fail_metrics(self):
        rows=[{'label':'FAIL','predicted_label':'PASS','status':'DEV_ONLY'},
              {'label':'PASS','predicted_label':None,'status':'INPUT_ERROR'}]
        self.assertEqual(decision_metrics(rows)['coverage'],0)
        r=decision_metrics(rows,allow_dev_only=True)
        self.assertEqual(r['coverage'],0.5);self.assertEqual(r['false_pass'],1)
        self.assertIsNone(r['fail_precision']);self.assertEqual(r['fail_recall'],0)
        self.assertIsNone(decision_metrics([])['coverage'])

    def test_groups_respect_reversed_label_mapping(self):
        c=Config(training_tier='DEV_ONLY',class_to_idx={'FAIL':0,'PASS':1})
        rows=[{'path':'synthetic.png','scene_id':'fixture','label':'FAIL','non_catalogue_count':3,'anomaly_type':'non_catalogue'}]
        r=grouped_validation(rows,[{'target_index':0,'predicted_index':0,'softmax':[0.8,0.2]}],c)
        self.assertEqual(r['groups']['non_catalogue_count']['3']['fail_recall'],1)
        self.assertEqual(r['cases'][0]['fail_softmax'],0.8)
        with self.assertRaises(ValueError):grouped_validation(rows,[],c)

    def test_adapter_dev_status_never_ok_and_no_fabricated_counts(self):
        class Backend:
            evidence_status='DEV_ONLY';class_to_idx={'PASS':0,'FAIL':1}
            def score_batch(self,images):return [[0.2,0.8] for _ in images]
        a=ClassificationAdapter(Backend(),[2]*5,'PROVISIONAL_PENDING_PILOT')
        r=a.predict(Image.new('RGB',(20,20)),[2]*5)
        self.assertEqual(r['status'],'DEV_ONLY');self.assertEqual(r['predicted_label'],'FAIL')
        self.assertNotIn('actual_counts',r);self.assertNotIn('boxes',r)
        self.assertEqual(a.predict('missing',[3]*5)['status'],'UNSUPPORTED_COUNTS')

if __name__=='__main__':unittest.main()
