"""Record scoped human/agent visual observations and candidate-family blocks; no data edits."""
import csv,json
from pathlib import Path
D=Path(__file__).resolve().parents[1];O=D/'acceptance_v1'
rs=json.loads((O/'independent_records.json').read_text());by={r['name']:r for r in rs}
rows=list(csv.DictReader((O/'layout_pairs.csv').open()))
parent={r['name']:r['name'] for r in rs}
def find(n):
 while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
 return n
def join(a,b):parent[find(a)]=find(b)
links=[]
for p in rows:
 if float(p['iou_1to1'])>.35:
  join(p['a'],p['b']);links.append({'a':p['a'],'b':p['b'],'basis':'one_to_one_iou_gt_035_candidate'})
audit=json.loads((O/'scene_audit.json').read_text())
for p in audit['known_pairs']:
 join(p['a'],p['b']);links.append({'a':p['a'],'b':p['b'],'basis':'historical_candidate_not_all_confirmed'})
a,b='MVIMG_20261120_171811.jpg','MVIMG_20261120_172012.jpg'
join(a,b);links.append({'a':a,'b':b,'basis':'targeted_visual_partial_layout_and_changed_unknown_objects; UNCONFIRMED_CAPTURE_HISTORY'})
for r in rs:
 if r['tier']=='augmented':join(r['name'],r['name'].split('__')[0]+'.jpg')
families={}
for n in parent:families.setdefault(find(n),[]).append(n)
payload=[]
for i,names in enumerate(sorted(families.values(),key=lambda x:sorted(x)[0]),1):
 if len(names)<2:continue
 payload.append({'candidate_family_id':f'CF{i:03d}','members':sorted(names),'splits':sorted({by[n]['split'] for n in names}),'original_group_ids':sorted({by[n]['group'] for n in names}),'status':'CANDIDATE_NOT_PHYSICAL_CERTIFICATION'})
(O/'candidate_families.json').write_text(json.dumps({'links':links,'families':payload,'cross_partition_families':[x for x in payload if len(x['splits'])>1]},ensure_ascii=False,indent=2)+'\n')
block_names=[r['name'] for r in rs if r['group'] in ('G_1120_B08','G_1120_B09')]
flags={'schema':'dataset_v2.eligibility.v1','engineering_status':'READY','formal_training_approval':'TEAM_SIGNOFF_PENDING; user approved construction, not evidence of whole-team historical-use signoff','test_status':'PROVISIONAL_HOLDOUT','formal_independent_test':'NOT_READY','all_test_files':[r['name'] for r in rs if r['split']=='test'],'reasons':['test_originals_29_PASS13_FAIL16_not_77_independent_scenes','historical_use_by_all_members_unknown','physical_scene_ledger_missing','nov23_shared_session','nov20_B08_B09_partial_layout_candidate'],'high_risk_pair':[a,b],'high_risk_pair_and_children':[n for n in by if n in (a,b) or any(n.startswith(Path(x).stem+'__') for x in (a,b))],'conservative_review_scope_B08_B09':block_names,'action':'Keep approved paths; do not execute formal Test; custodian must adjudicate full related families before any repartition. No automatic Test-to-Train move.'}
(D/'evaluation_eligibility.json').write_text(json.dumps(flags,ensure_ascii=False,indent=2)+'\n')
# Explicitly distinguish legacy scores from corrected supplemental screening.
p=D/'scene_groups.json';g=json.loads(p.read_text());g['disclaimer']='Candidate time/sequence blocks, NOT confirmed physical families. Low IoU cannot prove different physical scenes. Original audit below used same-batch many-to-one matching; retained as historical evidence only.';g['independent_audit']={'report':'INDEPENDENT_ACCEPTANCE_CN.md','one_to_one_all_batches':'acceptance_v1/scene_audit.json','transitive_candidate_families':'acceptance_v1/candidate_families.json','eligibility':'evaluation_eligibility.json','known_scene_isolation':'INCONCLUSIVE'};p.write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n')
p=D/'dataset_statistics.json';g=json.loads(p.read_text());g['integrity_summary']['legacy_layout_audit_status']='HISTORICAL_SAME_BATCH_MANY_TO_ONE_SCREENING_NOT_INDEPENDENCE_PROOF';g['independent_acceptance_reference']='INDEPENDENT_ACCEPTANCE_CN.md';p.write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n')
print('candidate families',len(payload),'cross',len([x for x in payload if len(x['splits'])>1]),'review block files',len(block_names))
