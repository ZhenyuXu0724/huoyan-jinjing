"""Nine fixed-budget runs; validation chooses checkpoints, never updates weights."""
import copy
from datetime import datetime,timezone
from importlib.metadata import version
from time import perf_counter
import csv
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ..features import read_image,extract
from ..data import sha256
from ..evaluate import write_predictions
from .common import (ROOT,OUT,WEIGHTS,CONFIGS,SEEDS,prepare,read,write,code_hashes,
                     sample_weights,prediction_rows,score_metrics,threshold_budget,checkpoint_key,group_key)

def fit(x,y,w,v,vy,*,layers,alpha,seed,epochs=200):
    model=MLPClassifier(hidden_layer_sizes=layers,activation='relu',solver='adam',alpha=alpha,
        learning_rate_init=.001,batch_size=min(32,len(y)),shuffle=False,max_iter=1,early_stopping=False,random_state=seed)
    order_rng,view_rng=np.random.default_rng(seed),np.random.default_rng(seed)
    history=[];best=None;best_model=None;epoch_scores=[]
    for epoch in range(1,epochs+1):
        order=order_rng.permutation(len(y));view_rng.integers(1,size=len(order))
        # Preserve v2's stack layout and optimizer updates exactly.
        xx=np.stack([x[i] for i in order])
        model.partial_fit(xx,y[order],classes=np.array([0,1]),sample_weight=w[order])
        scores=model.predict_proba(v)[:,1]
        if not np.array_equal(model.classes_,[0,1]):raise ValueError('Wrong probability column encoding')
        choice=threshold_budget(vy,scores)
        row=dict(epoch=epoch,train_loss=float(model.loss_),**choice)
        history.append(row);epoch_scores.append(scores.tolist())
        if best is None or checkpoint_key(row)<checkpoint_key(best):
            best=row.copy();best_model=copy.deepcopy(model)
    return best_model,model,best,history,epoch_scores

def run():
    if not (OUT/'stability/summary.json').exists():raise ValueError('Complete stability first')
    diagnostic=read(OUT/'stability/summary.json')
    if set(diagnostic)!={'train_old','train_v2','valid_old','valid_v2'}:raise ValueError('Incomplete stability run')
    samples=prepare();signature=dict(configs={k:dict(layers=list(v[0]),alpha=v[1]) for k,v in CONFIGS.items()},
        seeds=list(SEEDS),epochs=200,budget_fp=5,code_sha256=code_hashes(),
        threshold_rule='min FN then FP then largest representative threshold',
        checkpoint_rule='min FN then FP then earliest epoch',
        group_rule='mean FN then mean FP then smaller network then smaller alpha',
        data_audit_sha256=sha256(ROOT/'outputs/member04_hog/supervised_v2/data_audit.json'))
    if (OUT/'protocol.json').exists() and read(OUT/'protocol.json')!=signature:raise ValueError('Study protocol changed')
    write(OUT/'protocol.json',signature)
    config=dict(size=256,cell=16,epochs=200,geometry='A0',gamma=False,clahe=True,expected_counts={str(i):2 for i in range(1,6)})
    t=perf_counter();x0=np.stack([extract(read_image(s.path,256),config) for s in samples['train']]);scaler=StandardScaler().fit(x0)
    x=scaler.transform(x0);v=scaler.transform(np.stack([extract(read_image(s.path,256),config) for s in samples['valid']]))
    y=np.array([int(s.label=='FAIL') for s in samples['train']]);vy=np.array([int(s.label=='FAIL') for s in samples['valid']]);w=sample_weights(y)
    feature_seconds=perf_counter()-t;results={}
    with threadpool_limits(limits=1):
        for group,(layers,alpha) in CONFIGS.items():
            results[group]=[]
            for seed in SEEDS:
                destination=OUT/'experiments'/group/f'seed{seed}';weights=WEIGHTS/'experiments'/group/f'seed{seed}'
                if destination.exists() or weights.exists():raise ValueError('Run already exists; do not overwrite')
                destination.mkdir(parents=True);weights.mkdir(parents=True)
                print(f'Train {group} seed{seed} started',flush=True);started=perf_counter()
                best,last,choice,history,epoch_scores=fit(x,y,w,v,vy,layers=layers,alpha=alpha,seed=seed)
                reproduction=None
                if group=='R0':
                    reference=joblib.load(ROOT/f'weights/member04_hog/supervised_v2/experiments/B2/seed{seed}/model.joblib')
                    reproduction=dict(coefs_exact=all(np.array_equal(a,b) for a,b in zip(last.coefs_,reference['model'].coefs_,strict=True)),
                        intercepts_exact=all(np.array_equal(a,b) for a,b in zip(last.intercepts_,reference['model'].intercepts_,strict=True)),
                        scaler_exact=np.array_equal(scaler.scale_,reference['scaler'].scale_) and np.array_equal(scaler.mean_,reference['scaler'].mean_),
                        max_probability_difference=float(np.max(np.abs(last.predict_proba(v)-reference['model'].predict_proba(v)))))
                    write(destination/'reproduction.json',reproduction)
                    if not all(reproduction[k] for k in ('coefs_exact','intercepts_exact','scaler_exact')) or reproduction['max_probability_difference']!=0:
                        raise ValueError('R0 does not exactly reproduce v2; stop and investigate')
                metrics={}
                for name,model,ch in [('best',best,choice),('last',last,history[-1])]:
                    bundle=dict(kind='hog_supervised_classifier',version='Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection',role='local',group=group,seed=seed,
                        model=model,scaler=scaler,config={**config,'threshold':ch['threshold'],'selected_epoch':ch['epoch']},
                        classes={'PASS':0,'FAIL':1},capabilities=dict(image_level=True,objects=False,arbitrary_counts=False),
                        dependencies={k:version(k) for k in ('numpy','Pillow','scikit-image','scikit-learn','joblib')},
                        study_config=dict(layers=list(layers),alpha=alpha),code_sha256=signature['code_sha256'])
                    path=weights/f'{name}.joblib';joblib.dump(bundle,path)
                    scores=model.predict_proba(v)[:,1];predicted=prediction_rows(samples['valid'],scores,ch['threshold'])
                    write_predictions(destination/f'{name}_valid_predictions.csv',predicted)
                    metrics[name]={**score_metrics(predicted),**ch,'model_sha256':sha256(path),'model_path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size}
                metrics.update(group=group,seed=seed,fit_seconds=perf_counter()-started,feature_seconds=feature_seconds,reproduction=reproduction)
                write(destination/'metrics.json',metrics);write(destination/'history.json',history)
                write(destination/'validation_scores_by_epoch.json',dict(image_ids=[s.image_id for s in samples['valid']],scores=epoch_scores))
                write(destination/'config.json',dict(group=group,seed=seed,layers=list(layers),alpha=alpha,**config))
                results[group].append(metrics['best']);write(OUT/'progress.json',results)
                print(f"Done {group}/{seed}: epoch={choice['epoch']} FN={choice['fn']} FP={choice['fp']} time={metrics['fit_seconds']:.1f}s",flush=True)
    selected=min(results,key=lambda name:group_key(name,results[name]));representative=results[selected][0]
    write(OUT/'summary.json',dict(groups=results,selected_group=selected,representative_seed=42))
    prepare()
    write(OUT/'freeze.json',dict(created_at=datetime.now(timezone.utc).isoformat(),group=selected,seed=42,
        model_path=representative['model_path'],model_sha256=representative['model_sha256'],epoch=representative['epoch'],
        threshold=representative['threshold'],code_sha256=code_hashes(),protocol_sha256=sha256(OUT/'protocol.json'),
        scope='development validation selection; historical test only after freeze; no default replacement'))
    print(f'Frozen {selected} seed42 epoch{representative["epoch"]}',flush=True)

if __name__=='__main__':run()
