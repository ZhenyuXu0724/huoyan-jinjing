"""Factor-separated whole-image placement/scale diagnostics, never used for learning."""
import hashlib
from PIL import Image,ImageOps
import numpy as np
from threadpoolctl import threadpool_limits
from ..features import read_image,extract
from ..model import load_model,predict_with_model
from ..one_class.scoring import reconstruction_scores
from .common import ROOT,OUT,read,write,prepare,prediction_rows,score_metrics

def border_gray(path):
    with Image.open(path) as im:
        a=np.asarray(ImageOps.exif_transpose(im).convert('L'))
    h,w=a.shape;dy=max(1,int(np.ceil(h*.05)));dx=max(1,int(np.ceil(w*.05)))
    mask=np.zeros_like(a,dtype=bool);mask[:dy]=True;mask[-dy:]=True;mask[:,:dx]=True;mask[:,-dx:]=True
    return int(np.median(a[mask]))

def variants(image,gray):
    yield 'original',image
    raw=Image.fromarray(np.rint(image*255).astype('uint8'))
    for size in (240,224):
        small=np.asarray(raw.resize((size,size),Image.Resampling.LANCZOS),dtype=np.float32)/255
        gap=256-size;center=gap//2
        positions={'center':(center,center),'left':(0,center),'right':(gap,center),'top':(center,0),'bottom':(center,gap)}
        for fill,value in [('white',255),('border',gray)]:
            for pos,(x,y) in positions.items():
                canvas=np.full((256,256),np.float32(value/255),dtype=np.float32)
                canvas[y:y+size,x:x+size]=small
                yield f's{size}_{fill}_{pos}',canvas

def run():
    destination=OUT/'stability'
    if destination.exists():raise ValueError('Refuse to overwrite stability run')
    samples=prepare();destination.mkdir(parents=True)
    models={'old':load_model(ROOT/'weights/member04_hog/baseline1.joblib'),
            'v2':load_model(ROOT/'weights/member04_hog/supervised_v2/baseline1_supervised_v2.joblib')}
    results={}; rows=[]
    with threadpool_limits(limits=1):
        for split,items in samples.items():
            for n,s in enumerate(items):
                image=read_image(s.path,256);gray=border_gray(s.path);views=list(variants(image,gray))
                hashes={name:hashlib.sha256(a.tobytes()).hexdigest() for name,a in views}
                aliases={name:next(k for k,v in hashes.items() if v==h) for name,h in hashes.items()}
                for model_name,bundle in models.items():
                    # Single-image scoring matches production, avoiding batch numerical drift.
                    scores=[]
                    for name,a in views:
                        x=bundle['scaler'].transform(extract(a,bundle['config']).reshape(1,-1))
                        score=float(reconstruction_scores(bundle['model'],x)[0]) if model_name=='old' else float(bundle['model'].predict_proba(x)[0,1])
                        scores.append(score)
                    production=predict_with_model(s.path,bundle)
                    if production['prediction_score']!=scores[0]:raise ValueError('Original differs from production')
                    threshold=bundle['config']['threshold'];base=scores[0]>=threshold
                    for (name,_),score in zip(views,scores,strict=True):
                        rows.append(dict(split=split,model=model_name,image_id=s.image_id,label=s.label,status='OK',
                          variant=name,predicted_label='FAIL' if score>=threshold else 'PASS',prediction_score=score,
                          score_type=production['score_type'],threshold=threshold,flipped=bool((score>=threshold)!=base),
                          original_prediction=production['predicted_label'],duplicate_of=aliases[name],border_gray=gray))
                if (n+1)%20==0:print(f'Stability {split} {n+1}/{len(items)}',flush=True)
            for model_name in models:
                selected=[r for r in rows if r['split']==split and r['model']==model_name]
                groups={}
                for variant,_ in views:
                    group=[r for r in selected if r['variant']==variant]
                    stats=score_metrics(group)
                    stats.update(flips=sum(r['flipped'] for r in group),
                       correct_to_wrong=sum(r['original_prediction']==r['label'] and r['predicted_label']!=r['label'] for r in group),
                       wrong_to_correct=sum(r['original_prediction']!=r['label'] and r['predicted_label']==r['label'] for r in group))
                    groups[variant]=stats
                paired={}
                lookup={(r['image_id'],r['variant']):r for r in selected}
                for size in (240,224):
                    for fill in ('white','border'):
                        base=f's{size}_{fill}_center'
                        for pos in ('left','right','top','bottom'):
                            alt=f's{size}_{fill}_{pos}'
                            paired[alt+'_vs_'+base]=sum(lookup[(s.image_id,alt)]['predicted_label']!=lookup[(s.image_id,base)]['predicted_label'] for s in items)
                    for pos in ('center','left','right','top','bottom'):
                        paired[f's{size}_border_vs_white_{pos}']=sum(lookup[(s.image_id,f's{size}_white_{pos}')]['predicted_label']!=lookup[(s.image_id,f's{size}_border_{pos}')]['predicted_label'] for s in items)
                results[split+'_'+model_name]=dict(n=len(items),any_flip=sum(any(r['flipped'] for r in selected if r['image_id']==s.image_id) for s in items),variants=groups,paired_flips=paired)
            write(destination/'predictions.json',rows);write(destination/'summary.json',results)
    prepare()
    print('Stability complete; original inference matched for every image/model.',flush=True)

if __name__=='__main__':run()
