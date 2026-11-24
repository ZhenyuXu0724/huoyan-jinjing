"""User-confirmed frozen allocation and reproducible management annotations."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def policy():return json.loads((ROOT/'final_review_policy.json').read_text())
def append_note(notes,tag):
 parts=[x for x in notes.split(';') if x]
 if tag not in parts:parts.append(tag)
 return ';'.join(parts)

def apply_allocation(records,groups):
 p=policy();nov23=sorted([r for r in records if r['source_batch']=='11/23_PASS'],key=lambda r:r['image_filename'])
 six=[r['image_filename'] for r in nov23 if r['split']=='train']
 if len(six)!=6 or six[:2]!=p['selected']:raise ValueError('pre-freeze six or deterministic selection drift')
 groups[:]=[g for g in groups if g['source_batch']!='11/23_PASS']
 for r in nov23:
  if r['image_filename'] in p['selected']:r['split']='test'
  r['scene_group_id']='U1123_'+r['stem'];r['scene_independence_status']='DISTINCT_PLACEMENT_USER_CONFIRMED'
  groups.append({'scene_group_id':r['scene_group_id'],'source_batch':'11/23_PASS','subcondition':r['source_subcondition'],'assigned_split':r['split'],'grouping_basis':'user confirms each photograph followed reshuffling; filename-order first two of prior Train6 promoted','independence_status':'DISTINCT_PLACEMENT_USER_CONFIRMED','confirmation_source':'USER_CONFIRMATIONS.json','original_count':1,'augmented_count':0,'original_images':[r['image_filename']],'augmented_images':[]})

def apply_notes(records):
 p=policy()
 for r in records:
  parts=[x for x in r['quality_notes'].split(';') if x and not x.startswith('final_review_') and x!='reassigned_from_legacy_nov14_val_only_pending_team_signoff']
  r['quality_notes']=';'.join(parts)
  for tag,names in p['review_tags'].items():
   if r['image_filename'] in names:r['quality_notes']=append_note(r['quality_notes'],tag)

def check_rows(rows,p):
 def need(ok,msg):
  if not ok:raise ValueError(msg)
 by={r['image_filename']:r for r in rows};need(len(by)==len(rows),'duplicate review record')
 for n,sp in p['retained_splits'].items():need(n in by and by[n]['split']==sp,'unapproved promotion or reassignment: '+n)
 for tag,names in p['review_tags'].items():
  for n in names:need(n in by and tag in by[n]['quality_notes'].split(';'),'missing review restriction or confirmation: '+n)
 for parent in p['core_pair']:
  family={n for n in by if n==parent or n.startswith(Path(parent).stem+'__')}
  need(len(family)==4 and family.issubset(set(p['core_family'])),'incomplete parent family')
  need(len({by[n]['split'] for n in family})==1,'augmentation split leak')
 need(p['formal_test_status']=='NOT_READY' and p['partition_status']=='PROVISIONAL_HOLDOUT','unsupported test qualification')
 need(p['selected']==sorted(p['six_pass_candidates'])[:2],'non-deterministic selection')
 need(not any('final_review_' in r['quality_notes'] for r in rows),'stale resolved restriction')
 return {'review_status':'PASS','core_pair_distinct':'USER_CONFIRMED','moves':2,'formal_test_status':'NOT_READY','remaining':'OTHER_NOV20_PHYSICAL_FAMILY_ISOLATION'}

def run():
 p=policy();rows=list(csv.DictReader((ROOT/'dataset_manifest.csv').open()));result=check_rows(rows,p)
 for name in ('scene_groups.json','dataset_statistics.json','evaluation_eligibility.json'):
  if json.loads((ROOT/name).read_text()).get('final_revision')!=p['summary']:raise ValueError('inconsistent freeze metadata: '+name)
 (ROOT/'freeze_v1/restriction_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return result
if __name__=='__main__':run()
