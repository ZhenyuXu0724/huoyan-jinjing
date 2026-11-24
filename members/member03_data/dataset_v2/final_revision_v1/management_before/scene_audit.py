#!/usr/bin/env python3
"""Candidate screening only, never proof of scene independence or automatic repartition."""
import csv,json,itertools,collections
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw
from independent_acceptance import ROOT,SRC,OUT,require

def rotate(box,k):
 a=np.asarray(box,dtype=float).reshape((-1,4))
 if k==0:return a
 x1,y1,x2,y2=a.T
 return np.array(([y1,1-x2,y2,1-x1],[1-x2,1-y2,1-x1,1-y1],[1-y2,x1,1-y1,x2])[k-1]).T

def ious(a,b):
 lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
 inter=np.prod(np.maximum(hi-lo,0),axis=-1)
 area=lambda x:np.prod(x[:,2:]-x[:,:2],axis=1)
 return inter/(area(a)[:,None]+area(b)[None,:]-inter)

def layout_similarity(bx1,bx2,relative=False):
 """One-to-one same-label matching; return global/min-count containment scores.
 None for empty annotations: absence of geometry is not evidence of independence.
 relative=True removes each layout envelope translation/axis scale (heuristic).
 """
 if not bx1 or not bx2:return None
 la=np.array([x[0] for x in bx1]);lb=np.array([x[0] for x in bx2])
 a=np.array([x[1] for x in bx1]);b=np.array([x[1] for x in bx2]);best=(0.,0.)
 def normalize(x):
  lo=x[:,:2].min(0);hi=x[:,2:].max(0)
  return (x-np.tile(lo,2))/np.tile(hi-lo,2)
 if relative:a=normalize(a)
 for k in range(4):
  rb=rotate(b,k)
  if relative:rb=normalize(rb)
  mat=ious(a,rb)*(la[:,None]==lb[None,:])
  i,j=linear_sum_assignment(-mat);s=float(mat[i,j].sum())
  best=max(best,(s/max(len(a),len(b)),s/min(len(a),len(b))))
 return best

def sheet(name,records):
 thumb=(360,330);canvas=Image.new('RGB',(thumb[0]*4,thumb[1]*((len(records)+3)//4)), 'white');draw=ImageDraw.Draw(canvas)
 for i,r in enumerate(records):
  with Image.open(SRC/r['source']) as im:
   im=im.convert('RGB');im.thumbnail((350,275));x=(i%4)*360;y=(i//4)*330;canvas.paste(im,(x,y))
  draw.text((x,y+278),r['name'],fill='black');draw.text((x,y+294),f"{r['split']} {r['label']} counts={r['counts']}",fill='black')
 canvas.save(OUT/name)

def run():
 records=json.loads((OUT/'independent_records.json').read_text());orig=[r for r in records if r['tier']=='original'];by={r['name']:r for r in orig}
 pairs=[];edges=[];empty=[]
 for a,b in itertools.combinations(orig,2):
  v=layout_similarity(a['boxes'],b['boxes'])
  if v is None:
   empty.append([a['name'],b['name']]);continue
  rel=layout_similarity(a['boxes'],b['boxes'],True) if a['split']!=b['split'] else (None,None)
  p={'a':a['name'],'b':b['name'],'split_a':a['split'],'split_b':b['split'],'batch_a':a['batch'],'batch_b':b['batch'],'iou_1to1':v[0],'containment':v[1],'relative_iou':rel[0],'relative_containment':rel[1]}
  pairs.append(p)
  if v[0]>.35:edges.append(p)
 with (OUT/'layout_pairs.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
 # Connected components explicitly include edges across sources + parent links.
 uf={r['name']:r['name'] for r in records}
 def find(n):
  while uf[n]!=n:uf[n]=uf[uf[n]];n=uf[n]
  return n
 def union(a,b):uf[find(a)]=find(b)
 for p in edges:union(p['a'],p['b'])
 for r in records:
  if r['tier']=='augmented':union(r['name'],r['name'].split('__')[0]+'.jpg')
 components=collections.defaultdict(list)
 for r in records:components[find(r['name'])].append(r)
 crossing=[[r['name'] for r in rs] for rs in components.values() if len({r['split'] for r in rs})>1]
 known=[('IMG_20261114_160444.jpg','IMG_20261114_160608.jpg'),('IMG_7247.png','IMG_7223.png'),('IMG_7285.png','IMG_7277.png'),('IMG_7284.png','IMG_7273.png'),('IMG_7246.png','IMG_7233.png'),('IMG_7244.png','IMG_7243.png'),('IMG_20261114_162451.jpg','IMG_20261114_162500.jpg'),('IMG_20261114_161242.jpg','IMG_20261114_161317.jpg'),('IMG_20261114_161714.jpg','IMG_20261114_161719.jpg')]
 known_rows=[]
 for an,bn in known:
  a,b=by[an],by[bn]
  known_rows.append({'a':an,'b':bn,'split_a':a['split'],'split_b':b['split'],'group_a':a['group'],'group_b':b['group'],'score':layout_similarity(a['boxes'],b['boxes'])})
 cross=[p for p in pairs if p['split_a']!=p['split_b']];crossbatch=[p for p in cross if p['batch_a']!=p['batch_b']]
 result={'all_original_pairs':len(orig)*(len(orig)-1)//2,'scored_pairs':len(pairs),'empty_pairs_not_evaluable':len(empty),'cross_split_pairs':len(cross),'cross_batch_cross_split_pairs':len(crossbatch),'high_absolute_cross_pairs':[p for p in cross if p['iou_1to1']>.35],'crossing_candidate_components':crossing,'top_cross_batch':sorted(crossbatch,key=lambda p:p['iou_1to1'],reverse=True)[:10],'top_cross_relative':sorted(cross,key=lambda p:p['relative_containment'],reverse=True)[:10],'known_pairs':known_rows,'qualification':'INCONCLUSIVE; scores are screening heuristics, no automatically confirmed physical families'}
 (OUT/'scene_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 sheet('known_pairs.jpg',[by[n] for pair in known for n in pair])
 sheet('nov23_all.jpg',[r for r in orig if r['batch']=='11/23_PASS'])
 nov20=sorted([r for r in orig if r['batch']=='11/20'],key=lambda r:r['name']);boundary=[]
 for a,b in zip(nov20,nov20[1:]):
  if a['split']!=b['split']:boundary.extend([a,b])
 sheet('nov20_boundaries.jpg',boundary)
 empties=[r for r in records if r['name'].startswith('MVIMG_20261120_172557')];sheet('empty_scene_four.jpg',empties)
 top=[]
 for p in result['top_cross_batch'][:4]+result['top_cross_relative'][:4]:top.extend([by[p['a']],by[p['b']]])
 sheet('screening_candidates.jpg',top)
 print(json.dumps({k:v for k,v in result.items() if k not in ('top_cross_batch','top_cross_relative')},ensure_ascii=False,indent=2))
if __name__=='__main__':run()
