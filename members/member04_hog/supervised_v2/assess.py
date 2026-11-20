"""Historical regression and paired offline timing after the v2 freeze."""
import csv
from datetime import datetime, timezone
import platform
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from ..data import load_manifest, sha256, identity, reject_overlap
from ..evaluate import write_predictions
from ..model import load_model, predict_with_model
from .common import ROOT, DATA, OUT, read, write, code_hashes, check_old, score_metrics


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_freeze():
    frozen=read(OUT/'freeze.json')
    if frozen['code_sha256'] != code_hashes() or sha256(ROOT/frozen['model_path']) != frozen['sha256']:
        raise ValueError('Frozen code/model changed')
    if sha256(OUT/'training_protocol.json') != frozen['protocol_sha256']:
        raise ValueError('Protocol changed')
    for name, digest in frozen['manifests'].items():
        if sha256(DATA/f'{name}.csv') != digest:
            raise ValueError('Manifest changed')
    check_old()
    return frozen


def run():
    frozen=verify_freeze()
    destination=OUT/'assessment'
    destination.mkdir(parents=True,exist_ok=False)
    write(destination/'test_access.json',dict(accessed_at=datetime.now(timezone.utc).isoformat(),
          freeze_sha256=sha256(OUT/'freeze.json'),scope='historical public regression; already exposed'))
    samples=load_manifest(DATA/'test.csv')
    audit=read(OUT/'data_audit.json')
    if identity(samples)!=audit['identities']['test']:
        raise ValueError('Test changed since audit')
    for split in ('train','valid'):
        reject_overlap(identity(samples),audit['identities'][split])
    old=load_model(ROOT/'weights/member04_hog/baseline1.joblib')
    new=load_model(ROOT/frozen['model_path'])
    with threadpool_limits(limits=1):
        test_rows=[dict(image_id=s.image_id,label=s.label,**predict_with_model(s.path,new)) for s in samples]
    write_predictions(destination/'new_test_predictions.csv',test_rows)
    old_valid=csv_rows(ROOT/'outputs/member04_hog/experiments/A2/seed42/valid_predictions.csv')
    old_test=csv_rows(ROOT/'outputs/member04_hog/assessment/test/A2/seed42/predictions.csv')
    new_valid=csv_rows(OUT/'experiments'/frozen['selected_group']/'seed42/valid_predictions.csv')
    data={'old_valid':old_valid,'new_valid':new_valid,'old_test':old_test,'new_test':test_rows}
    comparison={key:score_metrics(rows) for key,rows in data.items()}
    for split in ('valid','test'):
        previous={r['image_id']:r for r in data['old_'+split]}
        current={r['image_id']:r for r in data['new_'+split]}
        if set(previous)!=set(current) or any(previous[k]['label']!=current[k]['label'] for k in previous):
            raise ValueError('Versions evaluated on different images or truth')
        old_errors=set(comparison['old_'+split]['errors']);new_errors=set(comparison['new_'+split]['errors'])
        comparison[split+'_error_changes']=dict(common=sorted(old_errors&new_errors),
            fixed=sorted(old_errors-new_errors),introduced=sorted(new_errors-old_errors))
    provenance=read(DATA/'provenance.json')['sources']
    valid_meta={r['image_path'].replace('\\','/').split('/')[-1]:r for r in provenance if r['split']=='valid'}
    grouped={}
    for key in ('old_valid','new_valid'):
        groups={}
        for row in data[key]:
            meta=valid_meta[row['image_id']]
            if row['label']=='PASS':
                source=meta['source_session'];condition=source
                categories=['PASS/source/'+source.split('_',1)[0], 'PASS/condition/'+condition,
                     'PASS/light/'+('dark' if '暗光' in source else 'standard' if '标准光' in source else 'unrecorded'),
                     'PASS/view/'+('side' if '侧视' in source else 'front' if '正视' in source else 'unrecorded')]
            else:
                categories=['FAIL/distractors/'+str(meta['counts']['0'])]
            for category in categories:groups.setdefault(category,[]).append(row)
        grouped[key]={k:score_metrics(rows) for k,rows in groups.items()}
    comparison['groups']=grouped
    constants={}
    for label in ('PASS','FAIL'):
        constants[label]=score_metrics([{**r,'predicted_label':label} for r in old_valid])
    comparison['constant_valid']=constants
    write(destination/'comparison.json',comparison)
    # Use a fixed, balanced 20-image validation subset for both systems, three passes.
    valid_samples=load_manifest(DATA/'valid.csv')
    rng=np.random.default_rng(42)
    timing_samples=[]
    for label in ('PASS','FAIL'):
        eligible=sorted([s for s in valid_samples if s.label==label],key=lambda s:s.image_id)
        timing_samples.extend(eligible[i] for i in rng.choice(len(eligible),10,replace=False))
    timings={'old':[],'new':[]}
    raw=[]
    with threadpool_limits(limits=1):
        for bundle in (old,new):
            for sample in timing_samples[:2]:predict_with_model(sample.path,bundle)
        for repeat in range(3):
            for name,bundle in ([('old',old),('new',new)] if repeat%2==0 else [('new',new),('old',old)]):
                for sample in timing_samples:
                    start=perf_counter();result=predict_with_model(sample.path,bundle)
                    duration=(perf_counter()-start)*1000
                    if result['status']!='OK':raise ValueError('Timing input failed')
                    timings[name].append(duration)
                    raw.append(dict(version=name,repeat=repeat,image_id=sample.image_id,milliseconds=duration))
    write_predictions(destination/'timings.csv',raw)
    write(destination/'timing_summary.json',dict(platform=platform.platform(),processor=platform.processor(),
          cpu_threads=1,repeat_count=3,images=[s.image_id for s in timing_samples],
          scope='preloaded model; image read, preprocessing, HOG and inference; excludes camera/UI/model load',
          sampling='seed42: 10 PASS + 10 FAIL validation images; same samples, alternating model order',
          results={k:dict(median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),observations=len(v)) for k,v in timings.items()}))
    verify_freeze()
    print({k:v for k,v in comparison.items() if k in ('old_valid','new_valid','old_test','new_test')},flush=True)


if __name__=='__main__':run()
