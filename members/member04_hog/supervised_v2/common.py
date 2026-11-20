"""Versioned paths and metrics for the supervised baseline; v1 stays immutable."""
import json
from pathlib import Path
import numpy as np
from ..data import sha256
from ..evaluate import metrics
from ..one_class.settings import GEOMETRY

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / 'members/member04_hog'
DATA = ROOT / 'data/member04_hog/supervised_v2'
OUT = ROOT / 'outputs/member04_hog/supervised_v2'
WEIGHTS = ROOT / 'weights/member04_hog/supervised_v2'
COMMIT = '12c85c23bc3a57dea51f39915278f31d9133ce74'
SEEDS = (42, 43, 44)
GROUPS = {k: dict(geometry=k, gamma=False, clahe=False) for k in GEOMETRY}
GROUPS.update(B1=dict(geometry='A0', gamma=True, clahe=False),
              B2=dict(geometry='A0', gamma=False, clahe=True),
              C0=dict(geometry='A2', gamma=False, clahe=True),
              C_without_geometry=dict(geometry='A0', gamma=False, clahe=True),
              C_without_clahe=dict(geometry='A2', gamma=False, clahe=False))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def score_metrics(rows):
    m = metrics(rows)
    m['accuracy'] = (m['tp'] + m['tn']) / m['decided'] if m['decided'] else None
    m['specificity'] = m['tn'] / (m['tn'] + m['fp']) if m['tn'] + m['fp'] else None
    m['balanced_accuracy'] = ((m['recall'] + m['specificity']) / 2
                              if all(m['support'].values()) and m['coverage'] == 1 else None)
    return m


def sample_weights(labels):
    labels = np.asarray(labels)
    if set(labels) != {0, 1}:
        raise ValueError('Training must contain PASS=0 and FAIL=1')
    return np.asarray([len(labels) / (2 * np.sum(labels == x)) for x in labels])


def choose_threshold(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if set(labels) != {0, 1} or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Both labels and finite scores required')
    unique = np.unique(scores)
    thresholds = np.r_[unique[0], unique[:-1] + np.diff(unique) / 2,
                       np.nextafter(unique[-1], np.inf)]
    ranked = []
    for t in thresholds:
        pred = scores >= t
        tp, fp = int(np.sum(pred & (labels == 1))), int(np.sum(pred & (labels == 0)))
        fn, tn = int(np.sum(~pred & (labels == 1))), int(np.sum(~pred & (labels == 0)))
        ba = (tp / (tp + fn) + tn / (tn + fp)) / 2
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        ranked.append((ba, -fn, f1, float(t)))
    return max(ranked)[-1]


def prediction_rows(samples, scores, threshold):
    return [dict(image_id=s.image_id, label=s.label, status='OK',
                 predicted_label='FAIL' if float(v) >= threshold else 'PASS',
                 prediction_score=float(v), score_type='uncalibrated_fail_probability')
            for s, v in zip(samples, scores, strict=True)]


def code_hashes():
    paths = [MODULE / 'supervised_v2' / p for p in ('common.py', 'train.py', 'inference.py')]
    paths += [MODULE / p for p in ('features.py', 'model.py', 'data.py', 'evaluate.py')]
    paths += [MODULE / 'configs/groups/geometry.json', MODULE / 'one_class/settings.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def preserve_old():
    target = OUT / 'old_preservation.json'
    if target.exists():
        check_old()
        return
    paths = [p for p in MODULE.rglob('*.py') if 'supervised_v2' not in p.parts
             and p.name != 'test_supervised_v2.py']
    for folder in ('weights/member04_hog/experiments', 'outputs/member04_hog/experiments',
                   'outputs/member04_hog/assessment'):
        paths += [p for p in (ROOT / folder).rglob('*') if p.is_file()]
    paths.append(ROOT / 'weights/member04_hog/baseline1.joblib')
    write(target, {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})


def check_old():
    for path, checksum in read(OUT / 'old_preservation.json').items():
        if sha256(ROOT / path) != checksum:
            raise ValueError(f'Old artifact changed: {path}')
