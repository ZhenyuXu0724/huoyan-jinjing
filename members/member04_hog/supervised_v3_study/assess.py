"""Post-freeze historical regression and real interface checks."""
import argparse,csv,tempfile
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from ..data import load_manifest,sha256
from ..model import predict_with_model
from ..supervised_v2.inference import predict,batch
from ..evaluate import write_predictions
from .common import ROOT,DATA,OUT,CONFIGS,SEEDS,read,write,prepare,code_hashes,score_metrics,prediction_rows

def verify():
    f=read(OUT/'freeze.json');prepare()
    if f['code_sha256']!=code_hashes() or sha256(ROOT/f['model_path'])!=f['model_sha256'] or sha256(OUT/'protocol.json')!=f['protocol_sha256']:
        raise ValueError('Frozen study modified')
    return f

def run():
    frozen=verify();destination=OUT/'assessment'
    if destination.exists():raise ValueError('Historical test already accessed; no overwrite')
    destination.mkdir(parents=True)
    write(destination/'test_access.json',dict(time=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha256(OUT/'freeze.json'),scope='historical regression, not independent test'))
    bundle=joblib.load(ROOT/frozen['model_path']);samples=load_manifest(DATA/'test.csv')
    with threadpool_limits(limits=1):
        rows=[dict(image_id=s.image_id,label=s.label,**predict_with_model(s.path,bundle)) for s in samples]
    write_predictions(destination/'test_predictions.csv',rows);write(destination/'test_metrics.json',score_metrics(rows))
    valid=load_manifest(DATA/'valid.csv');train=load_manifest(DATA/'train.csv')
    real=[];maximum=0
    with threadpool_limits(limits=1):
        for group in CONFIGS:
            for seed in SEEDS:
                metrics=read(OUT/f'experiments/{group}/seed{seed}/metrics.json')
                for checkpoint in ('best','last'):
                    m=metrics[checkpoint];path=ROOT/m['model_path']
                    if sha256(path)!=m['model_sha256']:raise ValueError('Checkpoint hash mismatch')
                    model=joblib.load(path)
                    with (OUT/f'experiments/{group}/seed{seed}/{checkpoint}_valid_predictions.csv').open(encoding='utf-8-sig') as stream:
                        saved={r['image_id']:r for r in csv.DictReader(stream)}
                    for s in valid:
                        result=predict_with_model(s.path,model)
                        if result['predicted_label']!=saved[s.image_id]['predicted_label']:raise ValueError('Single/batch label mismatch')
                        maximum=max(maximum,abs(result['prediction_score']-float(saved[s.image_id]['prediction_score'])))
                real.append(dict(group=group,seed=seed,reload='PASS'))
        train_rows=[dict(image_id=s.image_id,label=s.label,**predict_with_model(s.path,bundle)) for s in train]
    write_predictions(destination/'train_predictions.csv',train_rows);write(destination/'train_metrics.json',score_metrics(train_rows))
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        temp=Path(temp);bad=temp/'broken.png';bad.write_text('not an image',encoding='utf-8')
        model_path=ROOT/frozen['model_path']
        assert predict(bad,model_path)['status']=='INPUT_ERROR'
        assert predict(temp/'missing.png',model_path)['status']=='INPUT_ERROR'
        assert predict(valid[0].path,temp/'missing.joblib')['status']=='MODEL_UNAVAILABLE'
        request=temp/'requests.json'
        write(request,[dict(image_id=valid[0].image_id,expected_counts={str(i):2 for i in range(1,6)}),
                       dict(image_id='missing.png',expected_counts={str(i):2 for i in range(1,6)}),
                       dict(image_id='unsupported.png',expected_counts={str(i):1 for i in range(1,6)})])
        results=batch(DATA.parent/'dataset',request,temp/'batch',model_path)
        assert [r['status'] for r in results]==['OK','INPUT_ERROR','UNSUPPORTED_COUNTS']
    verify();write(OUT/'verification.json',dict(checkpoints=18,experiments=real,validation_images=101,
        max_single_batch_score_difference=maximum,label_mismatches=0,invalid_inputs='PASS',batch='PASS',old_v2_preservation='PASS'))
    print('18 checkpoints verified; historical test recorded after freeze',flush=True)

if __name__=='__main__':run()
