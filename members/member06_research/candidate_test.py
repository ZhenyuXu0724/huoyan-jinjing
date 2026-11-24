"""Locked v2-primary/v1-control evaluation; metadata checks never open test images/GT."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import Config
from .data import Preprocess, image_digest
from .runtime import file_sha256, load_checkpoint, seed_everything, environment, write_json
from .compare_dev_runs import metrics

BASE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = BASE/'final_evaluation_v1/protocol.json'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def new_json(p, obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def safe_path(root, relative):
    root=Path(root).resolve();rel=Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts,'relative path required')
    p=root/rel
    require(not any(x.is_symlink() for x in [p,*p.parents] if x!=root.parent),'symlinks forbidden')
    require(p.resolve().is_relative_to(root),'path escapes dataset root')
    return p


def protocol_check(path, *, allow_smoke=False):
    path=Path(path).resolve();p=read(path)
    require(file_sha256(path)==path.with_suffix('.sha256').read_text().strip(),'protocol changed')
    require(p['schema']=='member06.candidate-test.v1','protocol schema')
    require(p['primary']=='v2' and p['control']=='v1','fixed candidate roles changed')
    require(p['decision']=='argmax; ties PASS=0' and p['test_used_for_selection'] is False,'decision/selection changed')
    require(p['kind']=='REAL' or allow_smoke and p['kind']=='SMOKE_ONLY','SMOKE_ONLY needs explicit flag')
    for filename,sha in p['source_sha256'].items():
        require(file_sha256(BASE/filename)==sha,f'evaluator code changed: {filename}')
    payloads={}
    for name,item in p['models'].items():
        require(file_sha256(item['checkpoint'])==item['sha256'],f'checkpoint changed: {name}')
        state=torch.load(item['checkpoint'],weights_only=True,map_location='cpu')
        c=Config(**state['config'])
        require(state['format_version']==1 and state['architecture']=='frozen_resnet18_linear','model format')
        require(c.to_dict()==item['config'] and state['epoch']==item['epoch'],'model/config/epoch mismatch')
        require(c.class_to_idx=={'PASS':0,'FAIL':1} and c.supported_counts==p['expected_counts'],'class/count rule mismatch')
        require(c.data_kind==p['kind'] and state['evidence_status']==c.evidence_status,'evidence kind mismatch')
        require(state['run_metadata']['test_data_used'] is False,'model has Test selection history')
        payloads[name]=state
    a=Config(**payloads['v1']['config']).to_dict();b=Config(**payloads['v2']['config']).to_dict()
    require({k:(a[k],b[k]) for k in a if a[k]!=b[k]}=={'train_class_weight':('none','balanced')},'not the fixed single-factor pair')
    require(payloads['v1']['run_metadata']['datasets']==payloads['v2']['run_metadata']['datasets'],'development identity mismatch')
    require(all(torch.equal(v,payloads['v2']['model_state'][k]) for k,v in payloads['v1']['model_state'].items() if k.startswith('backbone.')),'backbone changed')
    if p['kind']=='REAL':
        current=environment(torch.device(p['device']))
        require(current['packages']==p['packages'] and current['python']==p['python'],'frozen dependency environment changed')
    return p,payloads


def eligibility(protocol_path, handoff_path, *, allow_smoke=False):
    p,payloads=protocol_check(protocol_path,allow_smoke=allow_smoke)
    h=read(handoff_path)
    require(h.get('status')=='APPROVED','BLOCKED: qualified custodial handoff absent')
    require(h.get('protocol_sha256')==file_sha256(protocol_path),'handoff does not bind protocol')
    require(h.get('scope')=='INDEPENDENT_HELD_OUT_CANDIDATE_EVALUATION','public partition not eligible')
    require(h.get('kind')==p['kind'],'handoff evidence kind mismatch')
    for key in ['custodian','independent_reviewer','decision_reference','dataset_version','approved_at','unsealed_access_history']:
        require(isinstance(h.get(key),str) and bool(h[key].strip()),f'missing {key}')
    require(h['custodian']!=h['independent_reviewer'],'independent reviewer required')
    for key in ['labels_reviewed','physical_scene_separation_verified','never_used_for_development',
                'sealed_until_binding','fixed_counts_confirmed','dev_candidate_limitations_accepted']:
        require(h.get(key) is True,f'{key} not confirmed')
    require(h.get('expected_counts')==p['expected_counts'],'counts differ; no rule change allowed')
    # Metadata-only documents, explicitly supplied by data custodian; no image/GT hashes read here.
    root=Path(h['dataset_root'])
    meta_path=safe_path(root,h['metadata_file'])
    require(file_sha256(meta_path)==h['metadata_sha256'],'metadata changed')
    m=read(meta_path);rows=m['images']
    require(m.get('contains_gt') is False and m.get('kind')==p['kind'],'metadata must not contain GT')
    require(len(rows)>0,'empty evaluation cohort')
    dev=payloads['v1']['run_metadata']['datasets']
    development=[r for split in ('train','val') for r in dev[split]['images']]
    ledger=m['development_scene_ledger']
    require({r['image_path'] for r in ledger}=={r['path'] for r in development} and len(ledger)==len(development),'complete development physical-scene ledger required')
    require(all(set(r)=={'image_path','physical_scene_id'} for r in ledger),'scene ledger must contain only path and physical-scene ID')
    for r in [*ledger,*rows]:
        require(isinstance(r.get('physical_scene_id'),str) and r['physical_scene_id'].strip() and not r['physical_scene_id'].upper().startswith(('UNVERIFIED','UNKNOWN','PENDING')),'trusted physical scene IDs required')
    require(not {r['physical_scene_id'] for r in rows}&{r['physical_scene_id'] for r in ledger},'physical scene overlap')
    allowed={'image_id','image_path','image_sha256','rgb_sha256','physical_scene_id','source_batch','capture_condition'}
    require(all(set(r)==allowed for r in rows),'metadata row schema (no labels/GT)')
    for key in ['image_id','image_path','image_sha256','rgb_sha256']:
        require(len({r[key] for r in rows})==len(rows),f'duplicate {key}')
    dev_rgb={r['sha256_rgb'] for r in development}
    require(not {r['rgb_sha256'] for r in rows}&dev_rgb,'pixel overlap with development')
    require(not {r['image_sha256'] for r in rows}&set(p['known_exposed_sha256']),'known exposed image forbidden')
    for r in rows:
        require(str(safe_path(root,r['image_path']).resolve()) not in {r['path'] for r in development},'development path overlap')
        for key in ['image_sha256','rgb_sha256']:
            require(len(r[key])==64 and all(c in '0123456789abcdef' for c in r[key]),'SHA256 required')
    require(h['n']==len(rows),'handoff count mismatch')
    require(set(h['class_support'])=={'PASS','FAIL'} and all(type(n) is int and n>0 for n in h['class_support'].values()) and sum(h['class_support'].values())==len(rows),'both class supports required')
    require(len(h['labels_sha256'])==64,'sealed labels hash required')
    safe_path(root,h['labels_file']) # no stat/open/hash of labels
    return p,h,m,payloads


def summarize(predictions, times):
    out=metrics(predictions)
    matrix=[[0,0],[0,0]]
    for r in predictions:matrix[r['target_index']][r['predicted_index']]+=1
    out.update(confusion_matrix=matrix,matrix_order='rows=true, columns=predicted; PASS,FAIL',
               latency_ms={'median':float(np.median(times)),'p95':float(np.percentile(times,95,method='linear')),'n':len(times)})
    return out


def evaluate(protocol_path,handoff_path,*,mode='check',allow_smoke=False):
    protocol_path=Path(protocol_path).resolve()
    p,h,m,payloads=eligibility(protocol_path,handoff_path,allow_smoke=allow_smoke)
    binding={'protocol_sha256':file_sha256(protocol_path),'handoff_sha256':file_sha256(handoff_path),
             'metadata_sha256':h['metadata_sha256'],'labels_sha256':h['labels_sha256'],'n':h['n'],
             'primary':'v2','control':'v1','status':'DATA_AND_PROTOCOL_BOUND_BEFORE_TEST_ACCESS'}
    bind_path=protocol_path.parent/'dataset_binding.json'
    if mode=='check':return {'status':'ELIGIBLE_METADATA_ONLY','n':h['n'],'test_content_read':False}
    if mode=='bind':
        new_json(bind_path,binding)
        return {'status':binding['status'],'test_content_read':False}
    require(mode=='execute','unknown mode')
    require(read(bind_path)==binding,'binding missing/changed: bind qualified metadata before Test access')
    # One claim per protocol, before any test content access. Failed attempts stay recorded.
    claim=protocol_path.parent/'execution_claim.json'
    new_json(claim,{**binding,'status':'CLAIMED_NO_AUTOMATIC_RETRY'})
    out=protocol_path.parent/'results'
    try:
        require(not out.exists(),'output exists; preserve previous results')
        out.mkdir()
        root=Path(h['dataset_root']);rows=m['images']
        paths=[]
        for row in rows:
            path=safe_path(root,row['image_path'])
            require(file_sha256(path)==row['image_sha256'],'test image hash mismatch')
            require(image_digest(path)==row['rgb_sha256'],'test RGB hash mismatch')
            paths.append(path)
        predictions={name:[] for name in ('v1','v2')};times={name:[] for name in predictions};models={}
        for name in predictions:
            model,c,device,_=load_checkpoint(p['models'][name]['checkpoint'],p['device'],allow_dev_only=True,allow_smoke=allow_smoke)
            require(str(device)==p['device'],'device changed')
            seed_everything(c.seed,c.cpu_threads)
            models[name]=(model,Preprocess(c),device)
            with torch.no_grad():
                for _ in range(p['latency']['warmup']):model(torch.zeros(1,3,c.image_size,c.image_size,device=device))
        def sync(device):
            if device.type=='cuda':torch.cuda.synchronize(device)
        with torch.no_grad():
            for i,(row,path) in enumerate(zip(rows,paths)):
                # Counterbalance order; both evaluate the exact same cohort, batch size one.
                for name in (('v2','v1') if i%2==0 else ('v1','v2')):
                    model,transform,device=models[name];sync(device);start=time.perf_counter_ns()
                    scores=model(transform(path).unsqueeze(0).to(device)).softmax(1)[0].cpu().tolist()
                    predicted=int(np.argmax(scores));sync(device);elapsed=(time.perf_counter_ns()-start)/1e6
                    require(len(scores)==2 and np.isfinite(scores).all(),'invalid prediction; abort entire evaluation')
                    times[name].append(elapsed)
                    predictions[name].append({'image_id':row['image_id'],'predicted_index':predicted,'softmax':scores,'latency_ms':elapsed})
        # GT released only after both predictions complete; GT never enters the model.
        labels_path=safe_path(root,h['labels_file'])
        require(file_sha256(labels_path)==h['labels_sha256'],'GT release hash mismatch')
        labels=read(labels_path)['images'];by_id={r['image_id']:r for r in labels}
        require(len(by_id)==len(labels)==len(rows) and set(by_id)=={r['image_id'] for r in rows},'GT coverage mismatch')
        require(Counter(r['label'] for r in labels)==h['class_support'],'class support differs from predeclared metadata')
        for row in labels:
            require(row['label'] in {'PASS','FAIL'},'invalid label')
            require(type(row['non_catalogue_count']) is int and row['non_catalogue_count']>=0,'reviewed anomaly count required')
            require(isinstance(row['anomaly_type'],str) and bool(row['anomaly_type']),'anomaly type required')
        result={'status':'SMOKE_ONLY' if p['kind']=='SMOKE_ONLY' else 'INDEPENDENT_HELD_OUT_DEV_CANDIDATE_EVALUATION',
                'formal_model_promoted':False,'primary':'v2','control':'v1','binding':binding,'models':{},'groups':[],'latency_protocol':p['latency'],
                'environment':environment(models['v2'][2]),'test_used_for_selection':False}
        for name,values in predictions.items():
            for row, pred in zip(rows,values):
                label=by_id[row['image_id']]
                pred.update(target_index=int(label['label']=='FAIL'),physical_scene_id=row['physical_scene_id'],
                            source_batch=row['source_batch'],capture_condition=row['capture_condition'],
                            anomaly_type=label['anomaly_type'],non_catalogue_count=label['non_catalogue_count'])
            result['models'][name]=summarize(values,times[name])
            for field in ['source_batch','capture_condition','anomaly_type','non_catalogue_count']:
                for value in sorted({str(x[field]) for x in values}):
                    selected=[x for x in values if str(x[field])==value]
                    result['groups'].append({'model':name,'field':field,'value':value,**summarize(selected,[x['latency_ms'] for x in selected])})
            write_json(out/f'{name}_predictions.json',values)
        result['paired_changes']={'v1_wrong_v2_right':sum(a['predicted_index']!=a['target_index'] and b['predicted_index']==b['target_index'] for a,b in zip(predictions['v1'],predictions['v2'])), 'v1_right_v2_wrong':sum(a['predicted_index']==a['target_index'] and b['predicted_index']!=b['target_index'] for a,b in zip(predictions['v1'],predictions['v2']))}
        result['development_reference']=p.get('development_reference',{})
        result['physical_scenes']=len({r['physical_scene_id'] for r in rows})
        result['independence_note']='Image metrics, not independent scene confidence intervals; repeated images in one scene remain correlated.'
        write_json(out/'metrics.json',result)
        with (out/'groups.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(result['groups'][0]));writer.writeheader();writer.writerows(result['groups'])
        lines=['# 固定候选独立评价','',result['status'],'','v2为预先指定主候选，v1仅作对照；不依据本次成绩换模型或调阈值。','', '|指标|v1对照|v2主候选|','|---|---:|---:|']
        for key in ['n','accuracy','balanced_accuracy','fail_precision','fail_recall','fail_f1','false_pass','false_fail','roc_auc','coverage','confusion_matrix','latency_ms']:
            lines.append(f'|{key}|{result["models"]["v1"][key]}|{result["models"]["v2"][key]}|')
        lines+=['','分组见groups.csv，逐图预测与延迟见v1/v2_predictions.json。缺类指标为null，不伪造零。',
                'Val参考指标保存在metrics.json/development_reference；其与Test的差异同时受选择偏差、类别/异常构成和来源分布影响，不能仅凭性能差宣布加权CE过拟合。',
                '这是已冻结DEV候选的独立评价，不改变checkpoint等级，不等于课程正式模型认证。任何后续基于Test的修改都需要新版本与新的未接触Test。',
                '延迟为batch1离线磁盘解码+预处理+设备传输+前向+softmax/argmax；不含模型加载、哈希、GT读取、摄像头和UI。缓存已受完整性检查影响，不是冷盘测量；两模型交替先后运行，CUDA同步，10次零张量预热。',
                '只有单次每图计时，没有硬件重复实验，不把相关场景帧当独立样本；泛化范围取决于实际条件和异常支持。']
        (out/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write_json(out/'SHA256.json',{f.name:file_sha256(f) for f in out.iterdir() if f.is_file()})
        new_json(protocol_path.parent/'execution_complete.json',{'status':'COMPLETE','binding':binding,'results_sha256':file_sha256(out/'SHA256.json')})
        return {'status':result['status'],'output':str(out)}
    except Exception as exc:
        new_json(protocol_path.parent/'execution_failed.json',{'status':'FAILED_NO_RETRY','error':str(exc)})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',default=str(DEFAULT_PROTOCOL));p.add_argument('--handoff')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--check-only',action='store_true');mode.add_argument('--bind',action='store_true');mode.add_argument('--execute',action='store_true')
    p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args()
    try:
        if not a.handoff:
            protocol,_=protocol_check(a.protocol,allow_smoke=a.allow_smoke)
            require(not a.execute and not a.bind,'qualified --handoff required')
            print(json.dumps({'status':'BLOCKED_DATA_HANDOFF','protocol_valid':True,'primary':protocol['primary'],'test_content_read':False}));return
        print(json.dumps(evaluate(a.protocol,a.handoff,mode='execute' if a.execute else 'bind' if a.bind else 'check',allow_smoke=a.allow_smoke),ensure_ascii=False))
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as e:p.exit(2,f'FINAL_TEST_BLOCKED: {e}\n')

if __name__=='__main__':main()
