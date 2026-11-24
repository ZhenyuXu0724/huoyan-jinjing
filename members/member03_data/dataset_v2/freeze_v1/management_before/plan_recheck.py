"""Recompute assignments independently of builder code, using reviewed sequence/block policy."""
import json,collections,csv
from pathlib import Path
from independent_acceptance import OUT,ROOT,require

def run():
 rs=json.loads((OUT/'independent_records.json').read_text());expected={}
 def assign_blocks(items,gap,splits):
  previous=None;block=1
  for r in sorted(items,key=lambda r:r['name']):
   hms=r['name'].split('_')[2][:6];t=int(hms[:2])*3600+int(hms[2:4])*60+int(hms[4:])
   if previous is not None and t-previous>=gap:block+=1
   expected[r['name']]=splits(block);previous=t
 for batch in {r['batch'] for r in rs if r['tier']=='original'}:
  items=[r for r in rs if r['batch']==batch]
  if batch=='11/11_parts_v1':
   for cond in {Path(r['source']).parent.name for r in items}:
    sub=sorted([r for r in items if Path(r['source']).parent.name==cond],key=lambda r:r['name'])
    for i,r in enumerate(sub):expected[r['name']]='train' if i<(31 if cond=='标准光正视' else 32) else 'val'
  elif batch=='11/14_PASS':
   for i,r in enumerate(sorted(items,key=lambda r:r['name'])):expected[r['name']]='train' if i<16 else 'val'
  elif batch=='11/14_FAIL':assign_blocks(items,30,lambda b:'val' if b in (1,3,8,11) else 'train')
  elif batch=='11/18_FAIL':assign_blocks(items,25,lambda b:'val' if b==3 else 'train')
  elif batch=='11/20':assign_blocks(items,60,lambda b:'train' if b in (1,2,3,4,5,6,8) else 'val' if b==7 else 'test')
  elif batch=='11/23_PASS':assign_blocks(items,60,lambda b:'train' if b==1 else 'test')
 for r in rs:
  if r['tier']=='augmented':expected[r['name']]=expected[r['name'].split('__')[0]+'.jpg']
 require(len(expected)==609 and all(expected[r['name']]==r['split'] for r in rs),'independent plan != disk')
 lock=json.loads((ROOT/'source_snapshot_lock.json').read_text())
 require(all(lock['assignments'][n]['split']==sp for n,sp in expected.items()),'lock != independent plan')
 result={'result':'PASS','assignments_checked':len(expected),'method':'separate reimplementation of recorded sequence/block policy; not proof policy establishes scene independence'}
 (OUT/'plan_recheck.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':run()
