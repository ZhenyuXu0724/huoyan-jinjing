"""Small synthetic regression tests: no real data mutation or model execution."""
import json,tempfile,unittest,random
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scene_audit import layout_similarity,rotate
from build_safety import validate_plan,refuse_existing
from independent_acceptance import ROOT,sha,require
import build_dataset_v2 as builder

class SafetyTests(unittest.TestCase):
 def test_existing_directory_preserved(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'freeze_v1') as d:
   root=Path(d);(root/'train').mkdir();f=root/'train'/'precious';f.write_bytes(b'keep')
   with self.assertRaisesRegex(ValueError,'EXISTING_DATASET_PROTECTED'):refuse_existing(root)
   self.assertEqual(f.read_bytes(),b'keep')
 def test_partial_empty_directory_refused(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'freeze_v1') as d:
   root=Path(d);(root/'test').mkdir()
   with self.assertRaises(ValueError):refuse_existing(root)
 def test_missing_changed_extra_sources_and_plan(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'freeze_v1') as d:
   root=Path(d);a=root/'a.jpg';b=root/'a.json';a.write_bytes(b'image');b.write_bytes(b'{}')
   r=dict(src_img=a,src_json=b,image_filename='a.jpg',split='train',label='PASS',data_tier='original',scene_group_id='g',parent_image_id='',augmentation_type='none')
   lock={'files':{p.name:{'sha256':sha(p)} for p in (a,b)},'assignments':{'a.jpg':{k:r[k] for k in ('split','label','data_tier','scene_group_id','parent_image_id','augmentation_type')}}}
   lp=root/'lock.json';lp.write_text(json.dumps(lock));validate_plan([r],root,lp)
   a.write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'SOURCE_SNAPSHOT_CHANGED'):validate_plan([r],root,lp)
   a.write_bytes(b'image');r['split']='test'
   with self.assertRaisesRegex(ValueError,'APPROVED_PLAN_CHANGED'):validate_plan([r],root,lp)
   with self.assertRaisesRegex(ValueError,'SOURCE_SNAPSHOT_CHANGED'):validate_plan([],root,lp)
   a.unlink()
   with self.assertRaises(FileNotFoundError):validate_plan([r],root,lp)
 def test_explicit_checks_not_assert(self):
  with self.assertRaises(ValueError):require(False,'bad')
 def test_plan_independent_of_glob_order(self):
  before,_=builder.collect_and_assign();old=Path.glob
  def reverse_glob(p,pattern):return iter(reversed(list(old(p,pattern))))
  with patch.object(Path,'glob',reverse_glob):after,_=builder.collect_and_assign()
  keys=lambda rs:sorted((r['image_filename'],r['split'],r['scene_group_id']) for r in rs)
  self.assertEqual(keys(before),keys(after))

class GeometryTests(unittest.TestCase):
 def setUp(self):self.b=[('a',[.1,.15,.2,.3]),('b',[.5,.6,.7,.8])]
 def test_identity_rotation_landscape_portrait(self):
  for k in range(4):
   rb=rotate([x[1] for x in self.b],k)
   self.assertAlmostEqual(layout_similarity(self.b,list(zip(['a','b'],rb)))[0],1)
  # Pixel coordinates after width/height swap give same normalized rotated box.
  x=np.array([100,300,200,600])/np.array([1000,2000,1000,2000])
  y=np.array([300,800,600,900])/np.array([2000,1000,2000,1000])
  np.testing.assert_allclose(rotate([x],1)[0],y)
 def test_one_to_one_no_reused_object(self):
  a=[('a',[.1,.1,.2,.2])];b=a+a
  self.assertAlmostEqual(layout_similarity(a,b)[0],.5)
  self.assertAlmostEqual(layout_similarity(a,b)[1],1.)
 def test_added_removed_objects_containment(self):
  self.assertAlmostEqual(layout_similarity(self.b,self.b[:1])[0],.5)
  self.assertAlmostEqual(layout_similarity(self.b,self.b[:1])[1],1.)
 def test_empty_not_independent(self):self.assertIsNone(layout_similarity([],self.b))
 def test_translation_scale_screening(self):
  other=[(l,(np.array(b)*.6+.2).tolist()) for l,b in self.b]
  self.assertLess(layout_similarity(self.b,other)[0],.5)
  self.assertAlmostEqual(layout_similarity(self.b,other,True)[0],1.)
 def test_different_labels_negative(self):
  self.assertEqual(layout_similarity(self.b,[('c',x[1]) for x in self.b])[0],0)
 def test_cross_source_screened(self):
  records=[{'data_tier':'original','source_batch':batch,'norm_boxes':self.b,'split':sp,'image_filename':batch,'scene_group_id':batch} for batch,sp in [('a','train'),('b','val')]]
  result=builder.check_cross_split_similarity(records)
  self.assertEqual(result['__ALL_BATCHES_SCREENING_ONLY__']['cross_split_pairs_gt_035_count'],1)

class FinalRevisionTests(unittest.TestCase):
 def setUp(self):
  import csv
  from final_revision_checks import policy
  self.rows=list(csv.DictReader((ROOT/'dataset_manifest.csv').open()))
  self.policy=policy()
 def test_qualified_status_not_inferred_from_counts(self):
  from final_revision_checks import check_rows
  self.policy['formal_test_status']='READY'
  with self.assertRaisesRegex(ValueError,'qualification'):check_rows(self.rows,self.policy)
 def test_no_silent_train_to_test_promotion(self):
  from final_revision_checks import check_rows
  name=self.policy['six_pass_candidates'][2]
  next(r for r in self.rows if r['image_filename']==name)['split']='test'
  with self.assertRaisesRegex(ValueError,'unapproved promotion'):check_rows(self.rows,self.policy)
 def test_all_three_children_keep_restriction(self):
  from final_revision_checks import check_rows
  name=next(n for n in self.policy['core_family'] if '__' in n)
  next(r for r in self.rows if r['image_filename']==name)['quality_notes']=''
  with self.assertRaisesRegex(ValueError,'missing review restriction'):check_rows(self.rows,self.policy)

if __name__=='__main__':unittest.main()
