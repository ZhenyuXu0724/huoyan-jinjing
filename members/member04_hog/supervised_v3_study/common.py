"""Shared study contract, immutable inputs and deterministic budget selection."""
from pathlib import Path
import numpy as np
from ..supervised_v2.common import ROOT, DATA, read, write, sample_weights, prediction_rows, score_metrics
from ..supervised_v2.assess import verify_freeze
from ..data import sha256, load_manifest, identity, reject_overlap
OUT = ROOT/'outputs/member04_hog/supervised_v3_study'
WEIGHTS = ROOT/'weights/member04_hog/supervised_v3_study'
CONFIGS = {'R0': ((64,16), .0001), 'R1': ((32,8), .0001), 'R2': ((64,16), .01)}
SEEDS = (42,43,44)

def code_hashes():
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}

def prepare():
    verify_freeze()
    frozen_paths=[]
    for folder in ('weights/member04_hog/supervised_v2','outputs/member04_hog/supervised_v2/experiments'):
        frozen_paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    frozen_paths.append(ROOT/'outputs/member04_hog/supervised_v2/freeze.json')
    snapshot={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(frozen_paths)}
    target=OUT/'v2_preservation.json'
    if target.exists():
        if read(target)!=snapshot:raise ValueError('v2 frozen experiment artifacts changed')
    else:write(target,snapshot)
    audit=read(ROOT/'outputs/member04_hog/supervised_v2/data_audit.json')
    samples={s:load_manifest(DATA/f'{s}.csv') for s in ('train','valid')}
    for s,items in samples.items():
        if identity(items)!=audit['identities'][s]:raise ValueError('Data changed: '+s)
        reject_overlap(identity(items),audit['identities']['test'])
    reject_overlap(identity(samples['train']),identity(samples['valid']))
    return samples

def threshold_budget(y,scores,budget=5):
    y=np.asarray(y); scores=np.asarray(scores,dtype=np.float64)
    if set(y)!={0,1} or y.shape!=scores.shape or not np.isfinite(scores).all() or budget<0:
        raise ValueError('Both classes, finite matching scores and nonnegative budget required')
    u=np.unique(scores)
    # Same finite representative cuts as v2; includes all PASS above maximum.
    cuts=np.r_[u[0],u[:-1]+np.diff(u)/2,np.nextafter(u[-1],np.inf)]
    candidates=[]
    for t in cuts:
        pred=scores>=t;fn=int(np.sum(~pred & (y==1)));fp=int(np.sum(pred & (y==0)))
        if fp<=budget:candidates.append((fn,fp,-float(t)))
    fn,fp,negative=min(candidates)
    return dict(threshold=-negative,fn=fn,fp=fp)

def checkpoint_key(row):return (row['fn'],row['fp'],row['epoch'])

def group_key(name, rows):
    layers,alpha=CONFIGS[name]
    return (np.mean([r['fn'] for r in rows]),np.mean([r['fp'] for r in rows]),sum(layers),alpha)
