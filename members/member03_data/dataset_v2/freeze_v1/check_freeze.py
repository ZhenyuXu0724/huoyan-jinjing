import hashlib,json,os,csv,sys
from pathlib import Path
D=Path(__file__).resolve().parents[1];R=D.parents[2];O=D/'freeze_v1';sys.path.insert(0,str(D))
def h(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):x.update(b)
 return x.hexdigest()
before=json.loads((O/'before.json').read_text());moves=json.loads((O/'moves.json').read_text());expected=dict(before['data'])
for m in moves:
 assert expected.pop(m['from'])==m['sha256'];expected[m['to']]=m['sha256']
actual={str(p.relative_to(D)):h(p) for sp in ('train','val','test') for p in (D/sp).rglob('*') if p.is_file()}
assert actual==expected,'data mutation beyond authorized four moves'
now={}
for base,dirs,files in os.walk(R,followlinks=False):
 dirs[:]=[n for n in dirs if Path(base)/n!=D]
 for n in files:
  p=Path(base)/n;now[str(p.relative_to(R))]={'link':os.readlink(p)} if p.is_symlink() else {'sha256':h(p)}
assert now==before['outside'],'outside change'
assert all((D/n).is_file() and h(D/n)==v for n,v in before['history'].items()),'history change'
from build_dataset_v2 import collect_and_assign
rs,groups=collect_and_assign();manifest=list(csv.DictReader((D/'dataset_manifest.csv').open()));by={r['image_filename']:r for r in manifest}
fields=['split','label','data_tier','scene_group_id','scene_independence_status','parent_image_id','augmentation_type','quality_notes']
assert all(all(str(r[k])==by[r['image_filename']][k] for k in fields) for r in rs),'builder/manifest mismatch'
actualgroups=json.loads((D/'scene_groups.json').read_text())['groups'];assert groups==actualgroups,'builder group mismatch'
oldlock=json.loads((O/'management_before/source_snapshot_lock.json').read_text());lock=json.loads((D/'source_snapshot_lock.json').read_text());assert oldlock['files']==lock['files'],'source bytes lock changed'
result={'result':'PASS','authorized_data_files_moved':len(moves),'data_files':len(actual),'other_data_changes':0,'outside_files_unchanged':len(now),'historical_files_unchanged':len(before['history']),'builder_records_match':len(rs),'builder_groups_match':len(groups),'source_lock_bytes_unchanged':True}
(O/'preservation_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
