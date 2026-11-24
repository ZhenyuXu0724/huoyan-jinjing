"""One-time authorized two-pair move and shared metadata serialization. No image/JSON edits."""
import csv,json,hashlib,sys,os
from pathlib import Path
D=Path(__file__).resolve().parents[1];O=D/'freeze_v1';sys.path.insert(0,str(D))
assert (O/'before.json').exists() and (O/'management_before/dataset_manifest.csv').exists()
assert not (O/'moves.json').exists(),'already applied; do not repeat'
rows=list(csv.DictReader((D/'dataset_manifest.csv').open()));six=sorted(r['image_filename'] for r in rows if r['source_batch']=='11/23_PASS' and r['split']=='train');assert len(six)==6
selected=six[:2]
confirm={'source':'用户本轮聊天中的四项明确事实与裁决；非Agent自行推断、非模型预测','scope':'Dataset v2 final freeze','facts':['同意新版划分及旧验证/历史评价数据用于新版开发训练','11/23正常照片每次拍摄前均重新打乱零件，属于不同物理摆放；允许从Train六张选两张','171811与172012不属同一关联摆放，保留它们及其增强分区','11/20、11/23新增数据未被参评模型用于训练、验证、选型、调参或模型错误分析'],'limitations':['不把共享背景会话等同同一摆放','本轮未获对11/20全部16张Test候选互相独立及与全部开发场景无关联的逐场景确认'],'selection_rule':'sort exact filenames of the six current Nov23 Train originals ascending; take first two','eligible_six':six,'selected':selected}
(D/'USER_CONFIRMATIONS.json').write_text(json.dumps(confirm,ensure_ascii=False,indent=2)+'\n')
old=json.loads((D/'final_review_policy.json').read_text());pair=old['core_pair'];core=old['core_family'];scope=old['conservative_review_scope'];newbatch=[r['image_filename'] for r in rows if r['source_batch'] in ('11/20','11/20_augmented','11/23_PASS')];legacy=[r['image_filename'] for r in rows if r['source_batch'] not in ('11/20','11/20_augmented','11/23_PASS')]
summary={'version':'freeze_v1','allocation_changed':True,'moved_images':2,'actual_test_original_pass':15,'actual_test_original_fail':16,'actual_test_originals':31,'confirmed_distinct_nov23_test_scenes':15,'nov20_test_physical_family_count':'UNVERIFIED','formal_test_status':'NOT_READY','scene_isolation':'NOV23_AND_SPECIFIED_PAIR_USER_CONFIRMED_OTHER_NOV20_UNVERIFIED','report':'FINAL_FREEZE_CN.md','confirmation_source':'USER_CONFIRMATIONS.json'}
p={'summary':summary,'core_pair':pair,'core_family':core,'six_pass_candidates':six,'selected':selected,'review_tags':{'user_confirmed_no_model_development_use':newbatch,'user_approved_legacy_reallocation':legacy,'user_confirmed_pair_distinct_layouts':core,'user_confirmed_nov23_reshuffled_scenes':[r['image_filename'] for r in rows if r['source_batch']=='11/23_PASS']},'retained_splits':{r['image_filename']:('test' if r['image_filename'] in selected else r['split']) for r in rows},'formal_test_status':'NOT_READY','partition_status':'PROVISIONAL_HOLDOUT','resolved_previous_restrictions':list(old['review_tags']),'confirmation_source':'USER_CONFIRMATIONS.json'}
(D/'final_review_policy.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
# Only four data files change path. Validate all before starting, rollback on move error.
moves=[]
for n in selected:
 for name in (n,Path(n).with_suffix('.json').name):
  src=D/'train/PASS/original'/name;dst=D/'test/PASS/original'/name
  assert src.is_file() and not dst.exists()
  moves.append({'from':str(src.relative_to(D)),'to':str(dst.relative_to(D)),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
done=[]
try:
 for m in moves:(D/m['from']).rename(D/m['to']);done.append(m)
except Exception:
 for m in reversed(done):(D/m['to']).rename(D/m['from'])
 raise
(O/'moves.json').write_text(json.dumps(moves,ensure_ascii=False,indent=2)+'\n')
import build_dataset_v2 as b
records,groups=b.collect_and_assign()
records.sort(key=lambda r:({'train':0,'val':1,'test':2}[r['split']],{'PASS':0,'FAIL':1}[r['label']],{'original':0,'augmented':1}[r['data_tier']],r['image_filename']))
by={r['image_filename']:r for r in rows}
for i,r in enumerate(records,1):
 m=by[r['image_filename']];r.update(sample_id=f'DSV2_{i:04d}',partition_status='PROVISIONAL_HOLDOUT' if r['split']=='test' else 'DEVELOPMENT_'+r['split'].upper(),original_relpath=str(r['src_img'].relative_to(D.parent)),original_json_relpath=str(r['src_json'].relative_to(D.parent)),v2_image_relpath=f"{r['split']}/{r['label']}/{r['data_tier']}/{r['image_filename']}",v2_json_relpath=f"{r['split']}/{r['label']}/{r['data_tier']}/{r['json_filename']}",image_sha256=m['image_sha256'],json_sha256=m['json_sha256'])
lock=json.loads((D/'source_snapshot_lock.json').read_text());lock['assignments']={r['image_filename']:{k:r[k] for k in ('split','label','data_tier','scene_group_id','parent_image_id','augmentation_type')} for r in records};lock['allocation_authority']='USER_CONFIRMATIONS.json';(D/'source_snapshot_lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
# Historical screening retained explicitly as pre-freeze evidence; do not rerun image audit.
historic=json.loads((O/'management_before/scene_groups.json').read_text())['cross_split_layout_similarity_audit']
b.write_metadata(records,groups,historic)
e={'schema':'dataset_v2.eligibility.freeze_v1','engineering_status':'READY','formal_training_approval':'USER_APPROVED_NEW_SPLIT_AND_LEGACY_REUSE','new_batch_model_development_history':'USER_CONFIRMED_NOT_USED','test_status':'PROVISIONAL_HOLDOUT','formal_independent_test':'NOT_READY','resolved':['legacy_reuse_approved','nov23_distinct_placements','171811_172012_distinct_placements','new_batch_no_model_development_use'],'remaining':['11/20 Test原图16张的物理家族数量及与其余开发场景的关联尚未逐场景确认；不能将31张照片直接当作31独立场景'],'support':{'test_originals':31,'test_PASS_originals':15,'test_FAIL_originals':16,'confirmed_nov23_test_placements':15,'nov20_test_family_count':'UNKNOWN','augmented_test_images':48},'all_test_files':[r['image_filename'] for r in records if r['split']=='test'],'final_revision':summary,'authority':'USER_CONFIRMATIONS.json'}
(D/'evaluation_eligibility.json').write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n')
print('Moved data files:',len(moves),'selected:',selected)
