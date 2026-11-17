"""Read-only model diagnosis; never fit models or replace frozen thresholds."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .data import load_manifest, sha256
from .evaluate import metrics, validate_freeze, write_json, write_predictions
from .model import predict_with_model

ROOT = Path(__file__).resolve().parents[2]


def summary(rows):
    result = metrics(rows)
    tn, fp = result['tn'], result['fp']
    result['specificity'] = tn / (tn + fp) if tn + fp else None
    result['balanced_accuracy'] = ((result['recall'] + result['specificity']) / 2
                                   if all(result['support'].values()) and result['coverage'] == 1 else None)
    scores = [row['prediction_score'] for row in rows if row['status'] == 'OK']
    result['score_quantiles'] = dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                                        map(float, np.quantile(scores, [0, .25, .5, .75, 1])))) if scores else {}
    return result


def threshold_curve(rows):
    scores = np.unique([r['prediction_score'] for r in rows])
    candidates = np.r_[scores[0], scores[:-1] + np.diff(scores) / 2,
                       np.nextafter(scores[-1], np.inf)]
    output = []
    for threshold in candidates:
        measured = summary([{**r, 'predicted_label': 'FAIL' if r['prediction_score'] >= threshold
                             else 'PASS'} for r in rows])
        output.append({'threshold': float(threshold), **{k: measured[k] for k in
                       ('false_pass', 'false_fail', 'f1', 'balanced_accuracy', 'coverage')}})
    return output


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    frozen = ROOT / 'outputs/member04_hog/assessment/freeze.json'
    record, models = validate_freeze(frozen)
    data = ROOT / 'data/member04_hog/dataset'
    selection = json.loads((ROOT / 'outputs/member04_hog/experiments/selection.json').read_text())
    assert sha256(ROOT / 'weights/member04_hog/baseline1.joblib') == selection['model_sha256']
    for split in ('train', 'valid'):
        assert sha256(data / f'{split}.csv') == selection[f'{split}_manifest_sha256']
    audited = []
    for path in sorted((ROOT / 'outputs/member04_hog/experiments').glob('*/seed*/metrics.json')):
        saved = json.loads(path.read_text())
        relative = path.parent.relative_to(ROOT / 'outputs/member04_hog/experiments')
        weight = ROOT / 'weights/member04_hog/experiments' / relative / 'model.joblib'
        assert sha256(weight) == saved['model_sha256'], str(relative)
        history = json.loads((path.parent / 'history.json').read_text())
        assert len(history) == 200
        audited.append(relative.as_posix())
    assert len(audited) == 33
    results = {}
    with threadpool_limits(limits=1):
        for entry, bundle in models:
            name = entry['name'].replace('/', '_')
            combined = []
            for split in ('train', 'valid'):
                samples = load_manifest(data / f'{split}.csv')
                if split == 'train':
                    assert len(samples) == 133 and all(s.label == 'PASS' for s in samples)
                for sample in samples:
                    relative = sample.path.relative_to(data)
                    condition = relative.parent.name
                    combined.append({'image_id': sample.image_id, 'split': split,
                                     'source': relative.parts[0], 'condition': condition,
                                     'lighting': 'dark' if '暗光' in condition else 'standard' if '标准光' in condition else 'unrecorded',
                                     'view': 'side' if '侧视' in condition else 'front' if '正视' in condition else 'unrecorded',
                                     'image_path': relative.as_posix(), 'label': sample.label,
                                     **predict_with_model(sample.path, bundle)})
            write_predictions(output / f'{name}_scores.csv', combined)
            grouped = defaultdict(list)
            for row in combined:
                for field in ('source', 'condition', 'lighting', 'view', 'label'):
                    grouped[f"{row['split']}/{field}/{row[field]}"].append(row)
            valid = [r for r in combined if r['split'] == 'valid']
            train = [r for r in combined if r['split'] == 'train']
            write_predictions(output / f'{name}_thresholds.csv', threshold_curve(valid))
            results[name] = {'threshold': bundle['config']['threshold'], 'train': summary(train),
                             'valid': summary(valid), 'groups': {k: summary(v) for k, v in grouped.items()}}
    write_json(output / 'diagnosis.json', {'freeze_sha256': sha256(frozen), 'audited_runs': audited,
               'usage': 'development diagnosis only; no model or threshold changes', 'models': results})
    print(json.dumps({k: {s: v[s] for s in ('threshold', 'train', 'valid')} for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
