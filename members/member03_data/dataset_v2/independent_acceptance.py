#!/usr/bin/env python3
"""Independent disk/source audit; never imports the builder or changes data files.
Run with /usr/bin/python3. Outputs only inside freeze_v1; earlier audit directories are historical.
"""
import collections as C
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent
REPO = ROOT.parents[2]
OUT = ROOT / 'freeze_v1'
VERSION = '755e038cc8f05b9701cceb28e3381d5fa3bc6a81'
LABELS = ['non_catalogue', 'M4x25', 'M5x10', 'M4x10', 'M4_hex_nut', 'flat_washer']
SOURCES = {
 '11/11_parts_v1': ('full_dataset_upload/datasets/parts_v1',160),
 '11/14_PASS': ('11.14训练补强＋干扰验证/补强数据(训练）',21),
 '11/14_FAIL': ('11.14训练补强＋干扰验证/验证集_不要训练',77),
 '11/18_FAIL': ('数据11.18_补含干扰件训练集',52),
 '11/20': ('11.20',70),
 '11/20_augmented': ('11.20_泛化增强',210),
 '11/23_PASS': ('1123pass/11.23补pass',19),
}
MATRIX = {'11/11_parts_v1':[127,0,32,1,0,0], '11/14_PASS':[16,0,5,0,0,0],
 '11/14_FAIL':[0,59,0,18,0,0], '11/18_FAIL':[0,40,0,12,0,0],
 '11/20':[2,46,0,6,0,16], '11/23_PASS':[4,0,0,0,15,0]}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def require(ok,msg):
 if not ok: raise ValueError(msg)
def write(name,x):
 (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def enumerate_sources():
 found=[]
 for batch,(rel,n) in SOURCES.items():
  images=sorted(p for p in (SRC/rel).rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png'))
  require(len(images)==n,f'{batch}: {len(images)} != {n}')
  jsons={p for p in (SRC/rel).rglob('*.json') if '统一GT' not in p.parts}
  require(jsons=={p.with_suffix('.json') for p in images},f'orphan JSON in {batch}')
  found.extend((batch,p) for p in images)
 return found

def anomaly(c):
 if c==[0,2,2,2,2,2]:return 'none'
 if not sum(c):return 'empty_scene'
 nc,miss,extra=c[0]>0,any(x<2 for x in c[1:]),any(x>2 for x in c[1:])
 if nc and miss and not extra:return 'non_catalogue_plus_missing'
 if nc and not miss and not extra:return f'non_catalogue_only_{c[0]}'
 if not nc and miss and not extra:return 'pure_missing'
 return '+'.join(([f'nc_{c[0]}'] if nc else [])+(['missing'] if miss else [])+(['extra'] if extra else [])) or 'other_fail'

def parse(p):
 raw=json.loads(p.with_suffix('.json').read_text())
 require(isinstance(raw.get('shapes'),list),f'missing shapes {p}')
 with Image.open(p) as im:
  size=im.size
  orientation=im.getexif().get(274)
  im.load()
 with Image.open(p) as check: check.verify()
 require(size==(raw['imageWidth'],raw['imageHeight']),f'dimensions {p}')
 c=[0]*6; boxes=[]
 for i,s in enumerate(raw['shapes']):
  require(s['label'] in LABELS,f'unknown label {p}:{i}')
  require(s.get('shape_type')=='rectangle',f'shape type {p}:{i}')
  pts=s['points']; require(len(pts)==2 and all(len(q)==2 for q in pts),f'points {p}:{i}')
  x1,x2=sorted(q[0] for q in pts); y1,y2=sorted(q[1] for q in pts)
  require(all(math.isfinite(v) for v in (x1,x2,y1,y2)) and 0<=x1<x2<=size[0] and 0<=y1<y2<=size[1],f'box bounds {p}:{i}')
  c[LABELS.index(s['label'])]+=1
  boxes.append((s['label'],[x1/size[0],y1/size[1],x2/size[0],y2/size[1]]))
 image_path=str(raw.get('imagePath','')).replace('\\','/')
 return raw,c,boxes,size,orientation,image_path

def run():
 OUT.mkdir(exist_ok=True)
 sources=enumerate_sources()
 # Git blob identity is independent of existing manifests and hashes.
 tree={}
 for entry in subprocess.check_output(['git','ls-tree','-rz',VERSION],cwd=REPO).split(b'\0'):
  if entry:
   meta,path=entry.split(b'\t',1);tree[path.decode()]=meta.decode().split()[2]
 manifest=list(csv.DictReader((ROOT/'dataset_manifest.csv').open()))
 require(len(manifest)==609,'manifest length')
 by_name={r['image_filename']:r for r in manifest}
 require(len(by_name)==609,'duplicate manifest image')
 disk=[p for sp in ('train','val','test') for p in (ROOT/sp).rglob('*') if p.is_file() or p.is_symlink()]
 require(len(disk)==1218,'extra/missing disk data')
 by_disk=C.defaultdict(list)
 for p in disk:by_disk[p.name].append(p)
 checks=[]; records=[]; warnings=[]; identities={}; inode_seen=set()
 for batch,p in sources:
  require(p.name in by_name,f'missing manifest {p}')
  m=by_name[p.name]
  raw,c,boxes,size,orientation,ip=parse(p)
  if ip.split('/')[-1]!=p.name:warnings.append({'file':p.name,'issue':'imagePath_basename_mismatch','value':ip})
  if orientation not in (None,1):warnings.append({'file':p.name,'issue':'EXIF_orientation','value':orientation})
  label='PASS' if c==[0,2,2,2,2,2] else 'FAIL'
  tier='augmented' if batch=='11/20_augmented' else 'original'
  for src,field,sfield,hfield in [(p,'v2_image_relpath','original_relpath','image_sha256'),(p.with_suffix('.json'),'v2_json_relpath','original_json_relpath','json_sha256')]:
   require(len(by_disk[src.name])==1,f'nonunique destination {src.name}')
   dst=by_disk[src.name][0]
   sp,dl,dt,_=dst.relative_to(ROOT).parts
   require(dl==label and dt==tier,f'wrong directory {dst}')
   require(m[field]==str(dst.relative_to(ROOT)) and m[sfield]==str(src.relative_to(SRC)),f'manifest path {src}')
   require(not dst.is_symlink() and not src.is_symlink() and not any(q.is_symlink() for q in dst.parents) and not any(q.is_symlink() for q in src.parents),f'symlink path {dst}')
   st=dst.stat(); inode=(st.st_dev,st.st_ino)
   require(st.st_nlink==1 and inode not in inode_seen and not os.path.samefile(src,dst),f'not independent copy {dst}')
   inode_seen.add(inode)
   hs,hd=sha(src),sha(dst)
   require(src.stat().st_size==st.st_size and hs==hd==m[hfield],f'bytes mismatch {src}')
   rel=str(src.relative_to(REPO))
   blob=subprocess.check_output(['git','hash-object','--',str(src)],cwd=REPO).decode().strip()
   require(tree.get(rel)==blob,f'approved Git source mismatch: {rel}')
   identities[str(src.relative_to(SRC))]={'sha256':hs,'git_blob':blob,'size':st.st_size}
   checks.append({'source':str(src.relative_to(SRC)),'destination':str(dst.relative_to(ROOT)),'size':st.st_size,'sha256':hs,'git_blob':blob,'byte_equal':True,'independent_inode':True})
  require(m['label']==label and m['data_tier']==tier and m['source_batch']==batch and m['split']==sp,f'manifest semantics {p}')
  require([int(m[f'count_{i}_{LABELS[i]}']) for i in range(6)]==c,f'manifest counts {p}')
  require((int(m['width']),int(m['height']))==size and int(m['total_objects'])==sum(c) and m['anomaly_category']==anomaly(c),f'manifest derived fields {p}')
  require(m['partition_status']==('PROVISIONAL_HOLDOUT' if sp=='test' else 'DEVELOPMENT_'+sp.upper()),f'partition status {p}')
  records.append({'name':p.name,'source':str(p.relative_to(SRC)),'batch':batch,'split':sp,'label':label,'tier':tier,'counts':c,'boxes':boxes,'size':size,'group':m['scene_group_id'],'anomaly':anomaly(c),'image_sha256':m['image_sha256']})
 require({str(p.relative_to(ROOT)) for p in disk}=={r['destination'] for r in checks},'disk bijection')
 require(len({r['image_sha256'] for r in records})==609,'duplicate image bytes')
 # Independent filename parent mapping, raw shape geometry and class comparison.
 by_record={r['name']:r for r in records}
 aug=list(csv.DictReader((ROOT/'augmentation_mapping.csv').open()))
 amap={r['aug_filename']:r for r in aug};require(len(aug)==len(amap)==210,'augmentation map bijection')
 parent_counts=C.Counter(); augtypes=C.Counter()
 for r in records:
  if r['tier']!='augmented':continue
  stem,suffix=Path(r['name']).stem.split('__'); pn=stem+'.jpg'; a=amap[r['name']]; parent=by_record[pn]
  require(suffix in ('A01_lowlight','A02_warm','A03_softnoise'),'augmentation type')
  require(r['split']==parent['split'] and r['label']==parent['label'] and r['group']==parent['group'],f'parent isolation {r["name"]}')
  require(r['boxes']==parent['boxes'] and r['size']==parent['size'] and r['counts']==parent['counts'],f'nongeometric geometry drift {r["name"]}')
  m,pm=by_name[r['name']],by_name[pn]
  require(m['parent_image_id']==pn and m['augmentation_type']==suffix,'manifest parent')
  expected={'aug_sample_id':m['sample_id'],'aug_filename':r['name'],'augmentation_type':suffix,'split':r['split'],'partition_status':m['partition_status'],'label':r['label'],'aug_v2_image_relpath':m['v2_image_relpath'],'aug_v2_json_relpath':m['v2_json_relpath'],'aug_image_sha256':m['image_sha256'],'aug_json_sha256':m['json_sha256'],'parent_sample_id':pm['sample_id'],'parent_filename':pn,'parent_split':parent['split'],'parent_label':parent['label'],'parent_v2_image_relpath':pm['v2_image_relpath'],'parent_image_sha256':pm['image_sha256'],'scene_group_id':parent['group'],'same_split_verified':'True','same_label_verified':'True'}
  require(a==expected,f'augmentation metadata {r["name"]}')
  parent_counts[pn]+=1;augtypes[suffix]+=1
 require(len(parent_counts)==70 and set(parent_counts.values())=={3},'three children per parent')
 # Statistics computed from actual source annotations + unique disk locations.
 stats=json.loads((ROOT/'dataset_statistics.json').read_text()); groups=json.loads((ROOT/'scene_groups.json').read_text())
 matrix={}
 for batch,expected in MATRIX.items():
  br=[r for r in records if r['batch']==batch]
  counts=[sum(r['split']==sp and r['label']==l for r in br) for sp in ('train','val','test') for l in ('PASS','FAIL')]
  require(counts==expected,f'approved matrix {batch}: {counts}')
  matrix[batch]=dict(zip(('train_PASS','train_FAIL','val_PASS','val_FAIL','test_PASS','test_FAIL'),counts));matrix[batch]['total_originals']=len(br)
 require(matrix==stats['original_source_allocation_matrix'],'statistics matrix')
 expected_totals={'all_images':len(records),'original_images':sum(r['tier']=='original' for r in records),'augmented_images':sum(r['tier']=='augmented' for r in records),'total_copied_files_images_plus_jsons':len(checks)}
 for tier in ('original','augmented'):
  for label in ('PASS','FAIL'):expected_totals[tier+'_'+label]=sum(r['tier']==tier and r['label']==label for r in records)
 require(stats['totals']==expected_totals,'statistics totals')
 split_stats={}
 for sp in ('train','val','test'):
  rs=[r for r in records if r['split']==sp];orig=[r for r in rs if r['tier']=='original'];ar=[r for r in rs if r['tier']=='augmented']
  x={'partition_status':'PROVISIONAL_HOLDOUT' if sp=='test' else 'DEVELOPMENT_'+sp.upper(),'total_images':len(rs),'original_images':len(orig),'augmented_images':len(ar)}
  for tier in ('original','augmented'):
   for l in ('PASS','FAIL'):x[tier+'_'+l]=sum(r['tier']==tier and r['label']==l for r in rs)
  for l in ('PASS','FAIL'):x['total_'+l]=sum(r['label']==l for r in rs)
  x.update(original_by_source_batch=dict(C.Counter(r['batch'] for r in orig)),original_anomaly_categories=dict(C.Counter(r['anomaly'] for r in orig)),augmented_by_type=dict(C.Counter(Path(r['name']).stem.split('__')[1] for r in ar)),object_instances_original_only={str(i):sum(r['counts'][i] for r in orig) for i in range(6)},object_instances_including_augmented={str(i):sum(r['counts'][i] for r in rs) for i in range(6)})
  require(x==stats['splits'][sp],f'statistics {sp}');split_stats[sp]=x
 require(len(groups['groups'])==groups['total_candidate_groups'],'group total')
 seen=[]
 for g in groups['groups']:
  rs=[r for r in records if r['group']==g['scene_group_id']]
  require(rs and all(r['split']==g['assigned_split'] for r in rs),'scene split')
  for tier in ('original','augmented'):
   names=[r['name'] for r in rs if r['tier']==tier];seen+=names
   require(len(names)==g[tier+'_count'] and sorted(names)==sorted(g[tier+'_images']),'scene membership')
 require(len(seen)==len(set(seen))==609,'scene group bijection')
 sums={}
 for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
  h,path=line.split('  ',1);require(path not in sums,'duplicate SHA line');sums[path]=h
 require(sums=={r['destination']:r['sha256'] for r in checks},'SHA256SUMS complete bijection')
 correction=json.loads((SRC/by_record['IMG_20261114_162432.jpg']['source']).with_suffix('.json').read_text())
 require(correction['shapes'][15]['label']=='non_catalogue' and correction['shapes'][16]['label']=='M4x10','162432 corrected indices')
 rel=str((SRC/by_record['IMG_20261114_162432.jpg']['source']).with_suffix('.json').relative_to(REPO))
 historical=subprocess.check_output(['git','show','16d9239aae5360134aae4ae6eb0b9a19703364ba:'+rel],cwd=REPO)
 require(hashlib.sha256(historical).hexdigest()==sha(REPO/rel),'correction commit identity')
 write('source_destination_checks.json',checks);write('independent_records.json',records)
 write('source_identity_verified.json',{'approved_commit':VERSION,'files':identities})
 conditions=C.Counter((Path(r['source']).parent.name,r['split'],r['label']) for r in records if r['batch']=='11/11_parts_v1')
 nc=C.Counter((r['batch'],r['split'],r['counts'][0]) for r in records if r['tier']=='original')
 summary={'result':'PASS','source_images':609,'source_jsons':609,'checked_files':len(checks),'approved_git_blob_matches':len(checks),'physical_independent_files':len(inode_seen),'matrix':matrix,'splits':split_stats,'augmentation_types':dict(augtypes),'parents':len(parent_counts),'parent_split_mismatches':0,'nongeometric_coordinate_mismatches':0,'warnings':warnings,'conditions_1111':{'|'.join(k):v for k,v in conditions.items()},'non_catalogue_distribution':{'|'.join(map(str,k)):v for k,v in nc.items()},'nov20_fail_washer_distribution':dict(C.Counter(r['counts'][5] for r in records if r['batch']=='11/20' and r['label']=='FAIL')),'nov18_compound':[r['name']+':'+r['split'] for r in records if r['batch']=='11/18_FAIL' and r['counts'][1]<2],'historical_IMG_7209':by_record['IMG_7209.png']}
 write('independent_summary.json',summary)
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
