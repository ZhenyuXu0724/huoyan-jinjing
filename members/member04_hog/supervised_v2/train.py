"""All 33 supervised experiments; weights learn from train only."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter
import shutil
import traceback

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..data import identity, load_manifest, reject_overlap, sha256
from ..evaluate import write_predictions
from ..features import read_image, extract
from ..one_class.settings import GEOMETRY
from .common import (ROOT, DATA, OUT, WEIGHTS, GROUPS, SEEDS, read, write,
                     sample_weights, choose_threshold, prediction_rows,
                     score_metrics, code_hashes, check_old)


def select_group(results):
    def key(name):
        rows = results[name]
        options = GROUPS[name]
        enabled = (options['geometry'] != 'A0') + int(options['gamma']) + int(options['clahe'])
        return (np.mean([r['balanced_accuracy'] for r in rows]),
                -np.mean([r['false_pass'] for r in rows]), -enabled,
                -len(GEOMETRY[options['geometry']]), not name.startswith('C'))
    return max(results, key=key)


def fit(valid_vectors, labels, weights, *, seed, epochs, variants):
    model = MLPClassifier(hidden_layer_sizes=(64, 16), activation='relu', solver='adam',
                          alpha=.0001, learning_rate_init=.001, batch_size=min(32, len(labels)),
                          shuffle=False, max_iter=1, early_stopping=False, random_state=seed)
    order_rng, view_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = order_rng.permutation(len(labels))
        views = view_rng.integers(len(variants), size=len(order))
        x = np.stack([variants[v][i] for i, v in zip(order, views, strict=True)])
        model.partial_fit(x, labels[order], classes=np.array([0, 1]), sample_weight=weights[order])
        history.append(dict(epoch=epoch, train_loss=float(model.loss_)))
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError('Unexpected probability column order')
    return model, history, model.predict_proba(valid_vectors)[:, 1]


def run():
    check_old()
    audit = read(OUT / 'data_audit.json')
    for name in ('train', 'valid', 'test'):
        if sha256(DATA / f'{name}.csv') != audit['manifests'][name]:
            raise ValueError('Manifest changed after audit')
    train, valid = [load_manifest(DATA / f'{name}.csv') for name in ('train', 'valid')]
    reject_overlap(identity(train), identity(valid))
    if identity(train) != audit['identities']['train'] or identity(valid) != audit['identities']['valid']:
        raise ValueError('Image content changed after audit')
    for samples in (train, valid):
        reject_overlap(identity(samples), audit['identities']['test'])
    labels = np.asarray([int(s.label == 'FAIL') for s in train])
    valid_labels = np.asarray([int(s.label == 'FAIL') for s in valid])
    balanced_weights = sample_weights(labels)
    config = dict(size=256, cell=16, epochs=200, expected_counts={str(i): 2 for i in range(1, 6)})
    source_hashes = code_hashes()
    signature = dict(code_sha256=source_hashes, manifests=audit['manifests'], groups=GROUPS,
                     config=config, hidden_layers=[64,16], sample_weight='n/(2*n_class)', seeds=list(SEEDS))
    signature_file = OUT / 'training_protocol.json'
    if signature_file.exists() and read(signature_file) != signature:
        raise ValueError('Protocol changed; new run directory required')
    write(signature_file, signature)
    started = perf_counter()
    images = {name: [read_image(s.path, 256) for s in samples] for name, samples in [('train', train), ('valid', valid)]}
    shared_decode = perf_counter() - started
    results = {}
    with threadpool_limits(limits=1):
        for group, options in GROUPS.items():
            prep_start = perf_counter()
            settings = {**config, **options}
            x = np.stack([extract(im, settings) for im in images['train']])
            scaler = StandardScaler().fit(x)
            original = scaler.transform(x)
            v = scaler.transform(np.stack([extract(im, settings) for im in images['valid']]))
            variants = [scaler.transform(np.stack([extract(im, settings, turns, mirror, gamma)
                        for im in images['train']]))
                        for turns, mirror in GEOMETRY[options['geometry']]
                        for gamma in ((.8, 1., 1.2) if options['gamma'] else (1.,))]
            prep_seconds = perf_counter() - prep_start
            rows = []
            for seed in SEEDS:
                destination = OUT / 'experiments' / group / f'seed{seed}'
                artifact = WEIGHTS / 'experiments' / group / f'seed{seed}' / 'model.joblib'
                metrics_path = destination / 'metrics.json'
                if metrics_path.exists():
                    saved = read(metrics_path)
                    if sha256(artifact) != saved['model_sha256']:
                        raise ValueError('Existing model changed')
                    rows.append(saved)
                    print(f'Reused {group} seed{seed}', flush=True)
                    continue
                if destination.exists() or artifact.exists():
                    raise ValueError(f'Incomplete run requires explicit review: {group}/seed{seed}')
                print(f'Training {group} seed{seed}: 185 images, 200 epochs', flush=True)
                run_start = perf_counter()
                model, history, scores = fit(v, labels, balanced_weights,
                                              seed=seed, epochs=200, variants=variants)
                threshold = choose_threshold(valid_labels, scores)
                bundle = dict(kind='hog_supervised_classifier', version='baseline1_supervised_v2',
                              role='local', group=group, seed=seed, model=model, scaler=scaler,
                              config={**settings, 'threshold': threshold}, classes={'PASS': 0, 'FAIL': 1},
                              capabilities=dict(image_level=True, objects=False, arbitrary_counts=False),
                              code_sha256=source_hashes, manifests=audit['manifests'],
                              dependencies={k: version(k) for k in ('numpy','Pillow','scikit-image','scikit-learn','joblib')})
                destination.mkdir(parents=True)
                artifact.parent.mkdir(parents=True)
                with artifact.open('xb') as stream:
                    joblib.dump(bundle, stream)
                predicted = prediction_rows(valid, scores, threshold)
                metric = score_metrics(predicted)
                metric.update(seed=seed, group=group, threshold=threshold, model_sha256=sha256(artifact),
                              model_bytes=artifact.stat().st_size, fit_validation_seconds=perf_counter()-run_start,
                              feature_preparation_seconds=prep_seconds, shared_decode_seconds=shared_decode,
                              seconds=shared_decode+prep_seconds+perf_counter()-run_start)
                write_predictions(destination / 'valid_predictions.csv', predicted)
                write(destination / 'history.json', history)
                write(destination / 'config.json', {**bundle['config'], 'hidden_layers':[64,16],
                      'alpha':.0001, 'learning_rate':.001, 'batch_size':32,
                      'class_weights':{str(i):float(balanced_weights[labels==i][0]) for i in (0,1)}})
                write(metrics_path, metric)
                rows.append(metric)
                print(f"Finished {group}/{seed}: BA={metric['balanced_accuracy']:.4f}, F1={metric['f1']:.4f}, FN={metric['fn']} FP={metric['fp']}", flush=True)
            results[group] = rows
            write(OUT / 'progress.json', results)
    selected = select_group(results)
    write(OUT / 'summary.json', dict(groups=results, selected_group=selected, integration_seed=42,
                                     usage='development validation; not independent performance'))
    chosen = WEIGHTS / 'experiments' / selected / 'seed42/model.joblib'
    default = WEIGHTS / 'baseline1_supervised_v2.joblib'
    if default.exists() and sha256(default) != sha256(chosen):
        raise ValueError('Refuse to overwrite a different selected model')
    if not default.exists():
        shutil.copyfile(chosen, default)
    frozen = OUT / 'freeze.json'
    if not frozen.exists():
        write(frozen, dict(frozen_at=datetime.now(timezone.utc).isoformat(), selected_group=selected,
                           model_path=default.relative_to(ROOT).as_posix(), sha256=sha256(default),
                           code_sha256=source_hashes, manifests=audit['manifests'],
                           protocol_sha256=sha256(signature_file),
                           test_scope='previously exposed public test; historical regression only'))
    check_old()
    print(f'Selected {selected} seed42; frozen. Old version unchanged.', flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception:
        write(OUT / ('failure_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'),
              {'traceback': traceback.format_exc()})
        raise
