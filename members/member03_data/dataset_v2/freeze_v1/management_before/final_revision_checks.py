"""Keep final eligibility annotations reproducible; never move or change data bytes."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def policy():
 return json.loads((ROOT/'final_review_policy.json').read_text())

def append_note(notes,tag):
 parts=[x for x in notes.split(';') if x]
 if tag not in parts:parts.append(tag)
 return ';'.join(parts)

def apply_notes(records):
 p=policy()
 for r in records:
  for tag,names in p['review_tags'].items():
   if r['image_filename'] in names:r['quality_notes']=append_note(r['quality_notes'],tag)

def check_rows(rows,p):
 def need(ok,msg):
  if not ok:raise ValueError(msg)
 by={r['image_filename']:r for r in rows}
 need(len(by)==len(rows),'duplicate review record')
 for name,sp in p['retained_splits'].items():need(name in by and by[name]['split']==sp,'unapproved promotion or family reassignment: '+name)
 for tag,names in p['review_tags'].items():
  for name in names:need(name in by and tag in by[name]['quality_notes'].split(';'),'missing review restriction: '+name)
 for parent in p['core_pair']:
  family={n for n in by if n==parent or n.startswith(Path(parent).stem+'__')}
  need(len(family)==4 and family.issubset(set(p['core_family'])),'incomplete parent family')
  need(len({by[n]['split'] for n in family})==1,'augmentation split leak')
 need(p['formal_test_status']=='NOT_READY' and p['partition_status']=='PROVISIONAL_HOLDOUT','unsupported test qualification')
 return {'review_status':'PASS','core_family_images':len(p['core_family']),'moves':0,'formal_test_status':'NOT_READY','scene_isolation':'INCONCLUSIVE'}

def run():
 p=policy();rows=list(csv.DictReader((ROOT/'dataset_manifest.csv').open()))
 result=check_rows(rows,p)
 for file in ('scene_groups.json','dataset_statistics.json','evaluation_eligibility.json'):
  obj=json.loads((ROOT/file).read_text())
  if obj.get('final_revision')!=p['summary']:raise ValueError('inconsistent final revision: '+file)
 (ROOT/'final_revision_v1/restriction_checks.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
 return result
if __name__=='__main__':run()
