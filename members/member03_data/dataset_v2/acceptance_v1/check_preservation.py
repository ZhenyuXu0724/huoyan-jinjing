import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parents[1];R=D.parents[2];O=D/'acceptance_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
before=json.loads((O/'outside_before.json').read_text());now={}
for base,dirs,files in os.walk(R,followlinks=False):
 dirs[:]=[n for n in dirs if Path(base)/n != D]
 for n in files:
  p=Path(base)/n
  now[str(p.relative_to(R))]={'symlink':os.readlink(p)} if p.is_symlink() else {'sha256':sha(p),'size':p.stat().st_size}
changed=[p for p in before if p in now and before[p]!=now[p]];added=sorted(set(now)-set(before));missing=sorted(set(before)-set(now))
data=json.loads((O/'data_before.json').read_text());current={str(p.relative_to(D)):sha(p) for sp in ('train','val','test') for p in (D/sp).rglob('*') if p.is_file()}
result={'outside_files_before':len(before),'outside_files_after':len(now),'outside_changed':changed,'outside_added':added,'outside_missing':missing,'dataset_files_before':len(data),'dataset_files_after':len(current),'dataset_data_bytes_unchanged':data==current,'scope':'repository files incl ignored files and .git; regular-file bytes/size and file symlink targets; not filesystem timestamps; proves this run only','result':'PASS' if not changed and not added and not missing and data==current else 'FAIL'}
(O/'preservation_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if result['result']!='PASS':raise SystemExit(1)
