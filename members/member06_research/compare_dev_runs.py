"""Read-only comparison of two completed DEV runs; only writes a new report directory."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from .config import Config
from .runtime import file_sha256, load_checkpoint, write_json
from .weighted_dev import ARCHIVE, BASE, read, verify_archive, config_diff


def metrics(predictions):
    if not predictions: return {'n':0,'coverage':None}
    y=np.array([p['target_index'] for p in predictions]); s=np.array([p['softmax'][1] for p in predictions])
    d=np.array([p['predicted_index'] for p in predictions]);tn=int(((y==0)&(d==0)).sum());fp=int(((y==0)&(d==1)).sum());fn=int(((y==1)&(d==0)).sum());tp=int(((y==1)&(d==1)).sum())
    pos=s[y==1];neg=s[y==0]
    return {'n':len(y),'support_PASS':tn+fp,'support_FAIL':tp+fn,'coverage':1.,
            'accuracy':(tn+tp)/len(y),'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if len(pos) and len(neg) else None,
            'fail_precision':tp/(tp+fp) if tp+fp else None,'fail_recall':tp/(tp+fn) if tp+fn else None,
            'fail_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'false_pass':fn,'false_fail':fp,
            'roc_auc':float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))) if len(pos) and len(neg) else None,
            'score_median_PASS':float(np.median(neg)) if len(neg) else None,'score_median_FAIL':float(np.median(pos)) if len(pos) else None}


def inspect_run(path):
    path=Path(path)
    required=['config.json','run_config.json','history.json','best.pt','checkpoint_manifest.json','best_val_predictions.json','best_metrics.json','best_val_groups.json']
    missing=[x for x in required if not (path/x).is_file()]
    if missing: raise ValueError(f'训练尚未完成或工件缺失: {path}: {missing}')
    config=Config.load(path/'config.json'); meta=read(path/'run_config.json'); history=read(path/'history.json')
    if config.training_tier!='DEV_ONLY' or config.class_to_idx!={'PASS':0,'FAIL':1}:raise ValueError('DEV_ONLY / positive class mismatch')
    digest=file_sha256(path/'best.pt')
    if read(path/'checkpoint_manifest.json')['sha256']!=digest: raise ValueError('checkpoint hash mismatch')
    model,loaded,_,payload=load_checkpoint(path/'best.pt',device='cpu',allow_dev_only=True)
    if loaded.to_dict()!=config.to_dict() or payload['run_metadata']!=meta: raise ValueError('checkpoint/config/metadata disagreement')
    if 'optimizer_state' not in payload:raise ValueError('incomplete checkpoint')
    if [r['epoch'] for r in history]!=list(range(1,config.epochs+1)):raise ValueError('incomplete epoch history')
    if not all(math.isfinite(r[s]['loss']) for r in history for s in ('train','val')):raise ValueError('nonfinite history')
    selected=min(history,key=lambda r:r['val']['loss'])
    if payload['epoch']!=selected['epoch'] or payload['val_metrics']!=selected['val']:raise ValueError('best epoch not minimum unweighted Val loss')
    bm=read(path/'best_metrics.json')
    if bm['epoch']!=payload['epoch'] or any(bm[k]!=v for k,v in payload['val_metrics'].items()):raise ValueError('best metrics mismatch')
    predictions=read(path/'best_val_predictions.json'); rows=meta['datasets']['val']['images']
    if len(predictions)!=len(rows) or len({p['image_path'] for p in predictions})!=len(rows):raise ValueError('Val coverage/duplicate mismatch')
    for p,row in zip(predictions,rows):
        if p['image_path']!=row['path'] or p['scene_id']!=row['scene_id'] or p['target_index']!=config.class_to_idx[row['label']]:raise ValueError('Val prediction identity mismatch')
        scores=p['softmax']
        if len(scores)!=2 or any(not math.isfinite(v) or not 0<=v<=1 for v in scores) or abs(sum(scores)-1)>1e-5:raise ValueError('invalid softmax')
        if p['predicted_index']!=int(np.argmax(scores)):raise ValueError('decision rule changed')
    m=metrics(predictions)
    for key in ('accuracy','false_pass','false_fail','fail_precision','fail_recall','fail_f1'):
        a,b=m[key],payload['val_metrics'][key]
        if (a is None)!=(b is None) or a is not None and abs(a-b)>1e-7:raise ValueError(f'prediction metrics differ: {key}')
    ce=sum(-math.log(max(p['softmax'][p['target_index']],1e-300)) for p in predictions)/len(predictions)
    if abs(ce-payload['val_metrics']['loss'])>1e-5:raise ValueError('Val loss is not unweighted CE from saved predictions')
    if meta.get('test_data_used') is not False:raise ValueError('Test exclusion missing')
    m.update(best_epoch=payload['epoch'],val_loss=payload['val_metrics']['loss'])
    return {'config':config.to_dict(),'metadata':meta,'metrics':m,'predictions':predictions,'rows':rows,'history':history,
            'identity':{x:file_sha256(path/x) for x in required}}


def compare(old_path,new_path,output,fp_budget=None):
    verify_archive()
    old,new=inspect_run(old_path),inspect_run(new_path)
    diff=config_diff(old['config'],new['config'])
    a,b=old['metadata'],new['metadata']
    for key in ('datasets','data_provenance_sha256','pretrained_source','decision_rule','selection'):
        if a.get(key)!=b.get(key):raise ValueError(f'D: experimental identity mismatch: {key}')
    for key in ('python','packages','device','torch_cuda'):
        if a['environment'][key]!=b['environment'][key]:raise ValueError(f'D: environment mismatch: {key}')
    protocol=read(BASE/'configs/weighted_ce_code_identity.json')
    for name,digest in a['environment']['source_sha256'].items():
        expected=protocol.get(name,digest)
        if b['environment']['source_sha256'].get(name)!=expected:raise ValueError(f'D: code mismatch: {name}')
    lp=b.get('loss_protocol',{})
    counts=[sum(r['label']==label for r in b['datasets']['train']['images']) for label in ('PASS','FAIL')]
    expected=[sum(counts)/(2*c) for c in counts]
    if lp.get('train_class_weight')!='balanced' or lp.get('validation')!='unweighted CE mean' or not np.allclose(lp.get('weights',[]),expected):raise ValueError('D: loss protocol mismatch')
    groups=[]
    for field in ('non_catalogue_count','source_package'):
        for value in sorted({str(r.get(field,'UNAVAILABLE')) for r in old['rows']}):
            ids=[i for i,r in enumerate(old['rows']) if str(r.get(field,'UNAVAILABLE'))==value]
            for version,run in [('v1',old),('v2',new)]:
                groups.append({'field':field,'group':value,'version':version,**metrics([run['predictions'][i] for i in ids])})
    x,y=old['metrics'],new['metrics'];group_better=any(g['version']=='v2' and g.get('fail_recall') is not None and g['fail_recall']>next(o['fail_recall'] for o in groups if o['field']==g['field'] and o['group']==g['group'] and o['version']=='v1') for g in groups)
    if y['fail_recall']>x['fail_recall'] and y['false_fail']>x['false_fail']:
        decision='B：FAIL检出增加，同时PASS误拒增加；可能是工作点变化，不能称为解决迁移。'
    elif y['roc_auc']<=x['roc_auc'] and y['balanced_accuracy']<=x['balanced_accuracy'] and not group_better:
        decision='C：AUC、Balanced Accuracy与异常分组均未改善，停止继续搜索CE权重，保留负结果。'
    elif y['roc_auc']>x['roc_auc'] and y['balanced_accuracy']>x['balanced_accuracy'] and y['fail_recall']>x['fail_recall']:
        decision='A候选：多项开发指标改善；改善是否明显、误拒代价是否可接受仍须团队确认，并需独立场景确认。'
    else:decision='混合结果：证据不足以判定明显改善，不自动选优或继续搜索。'
    if fp_budget is None:decision+=' 未提供团队允许的Val PASS误拒数，不能判定代价可接受。'
    else:decision+=f' 团队传入误拒预算={fp_budget}，本次FP={y["false_fail"]}，'+('预算内。' if y['false_fail']<=fp_budget else '超过预算。')
    output=Path(output)
    if output.exists():raise ValueError('output already exists; choose a new directory')
    output.mkdir(parents=True)
    identity={'DEV_ONLY':True,'config_diff':diff,'v1':old['identity'],'v2':new['identity'],'data_provenance_sha256':a['data_provenance_sha256'],'test_used':False,'fp_budget':fp_budget,'source_limitations':'Historical source recovered and hashed; config.py/train.py changes reviewed. Single seed, reused Val, provisional scene/annotation review.'}
    write_json(output/'experiment_identity.json',identity)
    write_json(output/'comparison_metrics.json',{'v1':x,'v2':y,'delta':{k:y[k]-x[k] if x[k] is not None and y[k] is not None else None for k in x},'decision':decision,'DEV_ONLY':True})
    with (output/'comparison_by_group.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for g in groups for k in g)));w.writeheader();w.writerows(groups)
    labels={'best_epoch':'最佳Epoch','val_loss':'无权Val Loss','accuracy':'Accuracy','balanced_accuracy':'Balanced Accuracy','fail_precision':'FAIL Precision','fail_recall':'FAIL Recall','fail_f1':'FAIL F1','false_pass':'False PASS / FN','false_fail':'False FAIL / FP','roc_auc':'ROC-AUC','coverage':'验证覆盖率'}
    fmt=lambda v:'未定义' if v is None else f'{v:.6g}'
    lines=['# DEV_ONLY：Weighted CE受控开发对照','',decision,'','| 指标 | DEV v1 | Weighted CE v2 | 差值 |','|---|---:|---:|---:|']
    for k,label in labels.items():lines.append(f'|{label}|{fmt(x[k])}|{fmt(y[k])}|{fmt(y[k]-x[k] if x[k] is not None and y[k] is not None else None)}|')
    lines+=['','## 真实标注分组','目录外数量来自清单标注，不是预测数量。完整来源分组见comparison_by_group.csv。','| 目录外数量 | 版本 | 支持数 | FAIL数 | 检出率 | FN | FP |','|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['field']=='non_catalogue_count':lines.append(f'|{g["group"]}|{g["version"]}|{g["n"]}|{g["support_FAIL"]}|{fmt(g["fail_recall"])}|{g["false_pass"]}|{g["false_fail"]}|')
    lines+=['','### 来源分组','| 来源 | FAIL支持 | v1召回 | v2召回 | v2漏放 |','|---|---:|---:|---:|---:|']
    for g in groups:
        if g['field']=='source_package' and g['version']=='v2':
            prev=next(o for o in groups if o['field']==g['field'] and o['group']==g['group'] and o['version']=='v1')
            lines.append(f'|{g["group"]}|{g["support_FAIL"]}|{fmt(prev["fail_recall"])}|{fmt(g["fail_recall"])}|{g["false_pass"]}|')
    lines+=['','## 分数与迁移解释',f'PASS分数中位数 {x["score_median_PASS"]:.6f} → {y["score_median_PASS"]:.6f}；FAIL中位数 {x["score_median_FAIL"]:.6f} → {y["score_median_FAIL"]:.6f}。',f'AUC变化 {y["roc_auc"]-x["roc_auc"]:+.6f}；FAIL Recall变化 {y["fail_recall"]-x["fail_recall"]:+.6f}；仍漏放 {y["false_pass"]}/{y["support_FAIL"]}。', '若召回增加而AUC未改善，支持主要是分数/工作点移动的解释，不能证明对象表示或跨来源迁移已修复。即使多指标改善，也只支持本次固定种子、既有Val上的开发收益。来源分组与异常数量仍有混杂，不能作因果归因。','', '## 比较边界','仅Train CE权重变化；Val始终无权CE、argmax默认规则。新旧Train Loss定义不同，不直接比较大小。Val曾选epoch且反复诊断，非独立盲测。标签、九张少件图及跨分区场景尚待成员2/3确认。没有Test访问、阈值搜索或自动选优。数据/配置不一致时工具以D失败退出，不生成指标报告。','', '图表：curves.png、score_distributions.png。历史源码和哈希见归档，身份见experiment_identity.json。']
    (output/'COMPARISON_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for label,run in [('v1',old),('weighted v2',new)]:
        axes[0].plot([h['epoch'] for h in run['history']],[h['val']['loss'] for h in run['history']],label=label)
        axes[1].plot([h['epoch'] for h in run['history']],[h['val']['accuracy'] for h in run['history']],label=label)
    for ax,title in zip(axes,['Unweighted validation CE','Validation accuracy']):ax.set_title(title);ax.set_xlabel('epoch');ax.legend()
    fig.tight_layout();fig.savefig(output/'curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(label,run) in zip(axes,[('v1',old),('weighted v2',new)]):
        for cls,name in [(0,'PASS'),(1,'FAIL')]:ax.hist([p['softmax'][1] for p in run['predictions'] if p['target_index']==cls],bins=np.linspace(0,1,21),alpha=.5,label=name)
        ax.axvline(.5,color='black',linestyle='--');ax.set_title(label);ax.set_xlabel('uncalibrated FAIL softmax');ax.legend()
    fig.tight_layout();fig.savefig(output/'score_distributions.png');plt.close(fig)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-run',required=True);p.add_argument('--output',required=True);p.add_argument('--max-false-fail',type=int)
    args=p.parse_args()
    if args.max_false_fail is not None and args.max_false_fail<0:p.error('max-false-fail must be nonnegative')
    try:print(compare(ARCHIVE/'run',args.new_run,args.output,args.max_false_fail))
    except (OSError,ValueError,KeyError,RuntimeError) as e:p.exit(2,f'COMPARISON_NOT_READY / D: {e}\n')
if __name__=='__main__':main()
