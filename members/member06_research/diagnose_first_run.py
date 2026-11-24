"""Read-only Train/Val diagnosis. New outputs only; no optimization or Test access.

Threshold sweeps reuse existing Val scores. Features are frozen backbone outputs;
PCA fits Train features only and is descriptive, never a causal/probe experiment.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def table(path, rows):
    if not rows:
        return
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_image(path, root, split):
    if split not in {'train', 'val'}:
        raise ValueError('only Train/Val allowed')
    p = Path(path).resolve()
    if not p.is_relative_to(root.resolve()) or any(
        x.lower().startswith('test') or '测试' in x for x in p.parts
    ):
        raise ValueError('image escapes development root or enters Test')
    return p


def binary_metrics(labels, scores, threshold=0.5):
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('invalid binary labels/scores')
    if not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError('scores must be finite softmax values')
    pred = s > threshold  # PASS index 0 wins ties, matching this run.
    tp = int(((y == 1) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    positive, negative = tp + fn, tn + fp
    div = lambda a, b: a / b if b else None
    recall, specificity = div(tp, positive), div(tn, negative)
    return dict(threshold=float(threshold), n=len(y), fail_support=positive, pass_support=negative,
                tp=tp, fn=fn, fp=fp, tn=tn, false_pass=fn, false_fail=fp,
                false_pass_rate=div(fn, positive), false_fail_rate=div(fp, negative),
                precision=div(tp, tp + fp), recall=recall, specificity=specificity,
                f1=div(2 * tp, 2 * tp + fp + fn), accuracy=div(tp + tn, len(y)),
                balanced_accuracy=(recall + specificity) / 2 if positive and negative else None,
                coverage=1.0 if len(y) else None)


def ranking(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    binary_metrics(y, s)
    pos, neg = s[y == 1], s[y == 0]
    auc = float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg))) if len(pos) and len(neg) else None
    points = [{'recall': 0.0, 'precision': 1.0}]
    ap, previous = 0.0, 0.0
    for score in sorted(set(s), reverse=True):
        included = s >= score
        tp = int(((y == 1) & included).sum())
        recall = tp / len(pos) if len(pos) else 0.0
        precision = tp / int(included.sum())
        ap += (recall - previous) * precision
        previous = recall
        points.append({'recall': recall, 'precision': precision})
    return {'roc_auc': auc, 'average_precision': ap if len(pos) else None,
            'pr_prevalence_baseline': len(pos) / len(y) if len(y) else None,
            'pr_points': points, 'definition': 'ROC pairwise with half-credit ties; AP step integral, ties grouped'}


def variance_fraction(features, groups):
    """Descriptive between-group / total scatter, not a classifier or causal test."""
    x = np.asarray(features, dtype=float)
    center = x.mean(0)
    total = float(((x - center) ** 2).sum())
    between = sum(len(x[np.asarray(groups) == g]) * float(((x[np.asarray(groups) == g].mean(0) - center) ** 2).sum()) for g in set(groups))
    return between / total if total else None


def diagnose(run, output, features=False):
    import torch
    from PIL import Image
    from .config import Config
    from .data import Preprocess
    from .runtime import load_checkpoint
    from .corpus_audit import describe
    from .team_data import unresolved_flags

    root = Path.cwd().resolve()
    member = Path(__file__).resolve().parent
    run, output = Path(run).resolve(), Path(output).resolve()
    if not run.is_relative_to(member / 'runs') or not output.is_relative_to(member / 'diagnostics'):
        raise ValueError('run must be member06/runs; output must be member06/diagnostics')
    if output.exists():
        raise ValueError('new output directory required')
    files = {p: digest(p) for p in run.iterdir() if p.is_file()}
    config = Config.load(run / 'config.json')
    if config.data_kind != 'REAL' or config.class_to_idx != {'PASS': 0, 'FAIL': 1}:
        raise ValueError('requires REAL run with PASS0/FAIL1')
    validation, best, history, metadata = [read(run / n) for n in
        ['validation.json', 'best_metrics.json', 'history.json', 'run_config.json']]
    provenance = read(config.data_provenance)
    files[Path(config.data_provenance).resolve()] = digest(config.data_provenance)
    if files[Path(config.data_provenance).resolve()] != metadata['data_provenance_sha256']:
        raise ValueError('provenance changed since training')
    cases = read(run / 'best_val_groups.json')['cases']
    if validation['groups']['cases'] != cases or validation['metrics'] != {k: best[k] for k in validation['metrics']}:
        raise ValueError('reloaded Val differs from saved best')
    checkpoint_hash = files[run / 'best.pt']
    if checkpoint_hash != validation['checkpoint_sha256'] or checkpoint_hash != read(run / 'checkpoint_manifest.json')['sha256']:
        raise ValueError('checkpoint hash conflict')
    for split in ['train', 'val']:
        p = Path(getattr(config, split + '_manifest')).resolve()
        files[p] = digest(p)
        if files[p] != metadata['datasets'][split]['manifest_sha256']:
            raise ValueError('manifest changed since training')
    case_map = {x['image_path']: x for x in cases}
    if len(case_map) != len(cases):
        raise ValueError('duplicate Val identity')
    source_map = {x['image_path']: x for x in provenance['audit']['images']}
    rows = []
    protected = dict(files)
    mapping_path = Path(provenance['audit']['sources']['legacy_root']) / '统一GT/class_mapping.json'
    label_ids = {x['labelme_label']: str(x['id']) for x in read(mapping_path)['classes']}
    transform = Preprocess(config)
    if features:
        model, loaded, device, payload = load_checkpoint(run / 'best.pt', 'cpu', allow_dev_only=True)
        if loaded.to_dict() != config.to_dict() or payload['epoch'] != best['epoch']:
            raise ValueError('checkpoint/config/epoch conflict')
        model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
    vectors, batch, ids = [], [], []
    output.mkdir(parents=True, exist_ok=False)
    issues = []
    for split in ['train', 'val']:
        for index, recorded in enumerate(metadata['datasets'][split]['images']):
            p = safe_image(recorded['path'], root / 'members/member03_data', split)
            source = source_map[str(p)]
            if source['split'] != split:
                raise ValueError('source partition conflict')
            annotation = p.with_suffix('.json')
            for f, expected in [(p, source['image_sha256']), (annotation, source['annotation_sha256'])]:
                protected[f] = digest(f)
                if protected[f] != expected:
                    raise ValueError(f'source changed: {f}')
            raw = read(annotation)
            counts = {str(i): 0 for i in range(6)}
            for shape in raw['shapes']:
                counts[label_ids[shape['label']]] += 1
            if counts != source['actual_counts'] or unresolved_flags(raw):
                raise ValueError(f'annotation conflict/unknown: {p}')
            derived = describe(source, config.supported_counts)
            if derived['annotation_label'] != recorded['label']:
                raise ValueError('binary label contradicts objects')
            with Image.open(p) as original:
                image = original.convert('RGB')
            rgb_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if rgb_hash != recorded['sha256_rgb']:
                raise ValueError('decoded image pixels changed')
            pixels = np.asarray(image.resize((64, 64)), dtype=float)
            luma = pixels @ np.array([.299, .587, .114])
            mask = np.ones((64, 64), dtype=bool)
            sides = []
            for shape in raw['shapes']:
                if shape['shape_type'] != 'rectangle':
                    raise ValueError('nonrectangle annotation')
                (x, y), (xx, yy) = shape['points']
                x, xx = sorted([x, xx]); y, yy = sorted([y, yy])
                if not (0 <= x < xx <= image.width and 0 <= y < yy <= image.height):
                    raise ValueError('bad ROI')
                mask[int(y / image.height * 64):int(np.ceil(yy / image.height * 64)), int(x / image.width * 64):int(np.ceil(xx / image.width * 64))] = False
                sides.append(min(xx - x, yy - y) * config.image_size / max(image.size))
            kind = source['source_kind']
            capture_batch = '2026-11-11' if kind == 'legacy' else '2026-11-18' if kind == 'flat_labelme' else '2026-11-14'
            row = dict(image_path=str(p), split=split, label=recorded['label'], scene_id=recorded['scene_id'],
                       scene_verified=not recorded['scene_id'].startswith('UNVERIFIED'), capture_batch=capture_batch,
                       capture_condition=source.get('capture_condition', 'UNRECORDED'),
                       background_material='UNVERIFIED', light_setup='UNVERIFIED',
                       source_package=source['source_package'], total_object_count=sum(counts.values()),
                       catalogue_count=sum(counts[str(i)] for i in range(1, 6)),
                       annotation_review='AUTO_PENDING' if source.get('raw_auto_review_pending') else 'NO_APPROVAL_EVIDENCE',
                       mean_luma=float(luma.mean()), background_luma_median=float(np.median(luma[mask])) if mask.any() else None,
                       background_red_minus_blue=float(np.median((pixels[:, :, 0] - pixels[:, :, 2])[mask])) if mask.any() else None,
                       object_short_side_at_input_median=float(np.median(sides)), width=image.width, height=image.height,
                       rgb_sha256=rgb_hash, **derived)
            if split == 'val':
                case = case_map[str(p)]
                if case['label'] != row['label']:
                    raise ValueError('Val identity/label mismatch')
                row['fail_softmax'] = case['fail_softmax']
                row['predicted_label'] = case['predicted_label']
            else:
                row['fail_softmax'] = None
                row['predicted_label'] = None
            if row['missing_class_ids'] or row['extra_class_ids']:
                issues.append({'image_path': str(p), 'issue': 'COUNT_ANOMALY_REQUIRES_PHYSICAL_REVIEW',
                               'missing': row['missing_class_ids'], 'extra': row['extra_class_ids'],
                               'binary_label': row['label'], 'is_confirmed_annotation_error': False})
            rows.append(row)
            if features:
                batch.append(transform(image))
                if len(batch) == 16:
                    with torch.inference_mode():
                        vectors.append(model.extract_features(torch.stack(batch).to(device)).cpu().numpy())
                    batch = []
            if (index + 1) % 50 == 0:
                print(f'diagnosis {split}: {index+1} images checked; no optimization', flush=True)
    if set(case_map) != {x['image_path'] for x in rows if x['split'] == 'val'}:
        raise ValueError('incomplete Val predictions')
    if features and batch:
        with torch.inference_mode():
            vectors.append(model.extract_features(torch.stack(batch).to(device)).cpu().numpy())
    if features and any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise ValueError('feature extraction changed model')
    # Brightness and chroma are image-derived proxies, not measured lighting/camera metadata.
    bins = {}
    for field in ['mean_luma', 'background_luma_median', 'background_red_minus_blue']:
        reference = [r[field] for r in rows if r['split'] == 'train' and r[field] is not None]
        cuts = np.quantile(reference, [1/3, 2/3]).tolist()
        bins[field] = cuts
        for r in rows:
            r[field + '_bin'] = 'MISSING' if r[field] is None else ['low', 'middle', 'high'][np.searchsorted(cuts, r[field], side='right')]
    val = [r for r in rows if r['split'] == 'val']
    y = [int(r['label'] == 'FAIL') for r in val]; scores = [r['fail_softmax'] for r in val]
    thresholds = sorted(set([-1e-12, .1, .2, .3, .4, .5, 1.0] + scores))
    sweep = [binary_metrics(y, scores, t) for t in thresholds]
    ranks = ranking(y, scores)
    groups = {}
    fields = ['capture_batch', 'capture_condition', 'background_material', 'light_setup', 'total_object_count', 'catalogue_count',
              'anomaly_type', 'non_catalogue_count', 'mean_luma_bin', 'background_luma_median_bin', 'background_red_minus_blue_bin']
    for field in fields:
        groups[field] = {}
        for value in sorted({str(r[field]) for r in val}):
            subset = [r for r in val if str(r[field]) == value]
            groups[field][value] = binary_metrics([int(r['label'] == 'FAIL') for r in subset], [r['fail_softmax'] for r in subset])
    support = defaultdict(Counter)
    for row in rows:
        support[(row['split'], row['capture_batch'], row['capture_condition'])][row['label']] += 1
    duplicates = []
    for key in ['image_path', 'rgb_sha256', 'scene_id']:
        bucket = defaultdict(list)
        for r in rows:
            bucket[r[key]].append(r)
        for value, items in bucket.items():
            if len(items) > 1:
                duplicates.append({'key': key, 'value': value, 'paths': [r['image_path'] for r in items],
                                   'cross_split': len({r['split'] for r in items}) > 1})
    results = dict(scope='DEV_DIAGNOSTIC_ONLY_NOT_THRESHOLD_SELECTION', test_content_read=False,
                   real_training_executed=False, features_extracted=features, model_updated=False,
                   best_epoch=best['epoch'], last_epoch=history[-1]['epoch'],
                   last_epoch_train=history[-1]['train'], best_epoch_train_online=history[best['epoch']-1]['train'],
                   train_metric_note='online minibatch metrics while head changes, not evaluation of final/best checkpoint',
                   default=binary_metrics(y, scores), ranking=ranks, thresholds_rule='FAIL iff stored FAIL softmax > threshold; PASS wins exact ties',
                   best_balanced_accuracy_on_reused_val=max(sweep, key=lambda r: r['balanced_accuracy']),
                   best_f1_on_reused_val=max(sweep, key=lambda r: r['f1']),
                   groups=groups, bin_cutpoints_train_only=bins,
                   source_support=[dict(split=k[0], batch=k[1], condition=k[2], support=dict(v)) for k, v in support.items()],
                   review_support=dict(Counter(r['annotation_review'] for r in rows)),
                   scene_verified_count=sum(r['scene_verified'] for r in rows), duplicates=duplicates,
                   issues=issues, val_reload_matches=True)
    if features:
        x = np.concatenate(vectors).astype(np.float64)
        ntrain = sum(r['split'] == 'train' for r in rows)
        train = x[:ntrain]; mean = train.mean(0)
        _, singular, axes = np.linalg.svd(train - mean, full_matrices=False)
        coordinates = (x - mean) @ axes[:2].T
        np.savez_compressed(output / 'frozen_features.npz', features=x, pca_coordinates=coordinates,
                            pca_train_mean=mean, pca_axes=axes[:2], image_paths=np.array([r['image_path'] for r in rows]))
        labels = np.array([r['label'] for r in rows]); batches = np.array([r['capture_batch'] for r in rows])
        norms = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        cosine = norms[ntrain:] @ norms[:ntrain].T
        nearest = cosine.argmax(axis=1)
        neighbors = [dict(val_image=rows[ntrain+i]['image_path'], train_image=rows[j]['image_path'],
                          cosine=float(cosine[i, j]), val_label=str(labels[ntrain+i]), train_label=str(labels[j]),
                          is_confirmed_same_scene=False) for i, j in enumerate(nearest)]
        table(output / 'nearest_train_features.csv', neighbors)
        feature_result = {'pca_fit_split': 'train_only', 'dimension': int(x.shape[1]),
                          'pc1_pc2_train_variance_fraction': (singular[:2]**2 / (singular**2).sum()).tolist(),
                          'label_between_fraction': variance_fraction(x, labels),
                          'batch_between_fraction': variance_fraction(x, batches),
                          'label_between_fraction_train': variance_fraction(train, labels[:ntrain]),
                          'batch_between_fraction_train': variance_fraction(train, batches[:ntrain]),
                          'batch_between_fraction_within_label': {label: variance_fraction(x[labels == label], batches[labels == label]) for label in ['PASS', 'FAIL']},
                          'nearest_train_label_agreement_descriptive': float(np.mean(labels[nearest] == labels[ntrain:])),
                          'note': 'No probe fitting. Different group counts/confounding prohibit direct causal comparison. Neighbors are NOT verified physical scenes.'}
        results['features'] = feature_result
        feature_plot(output, coordinates, rows)
    for p, before_hash in protected.items():
        if digest(p) != before_hash:
            raise ValueError(f'input changed during diagnosis: {p}')
    results['protected_inputs_unchanged'] = True
    results['input_sha256'] = {str(p): h for p, h in protected.items()}
    table(output / 'image_diagnostics.csv', rows)
    table(output / 'threshold_sweep.csv', sweep)
    table(output / 'review_candidates.csv', issues)
    save(output / 'diagnosis.json', results)
    plot_metrics(output, sweep, ranks)
    print(json.dumps({'default': results['default'], 'roc_auc': ranks['roc_auc'], 'average_precision': ranks['average_precision'],
                      'features': results.get('features'), 'protected_inputs_unchanged': True}, indent=2), flush=True)
    return results


def plot_metrics(output, sweep, ranks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot([r['false_fail_rate'] for r in sweep], [r['recall'] for r in sweep])
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlabel='False reject rate (PASS -> FAIL)', ylabel='FAIL recall', title=f"ROC AUC={ranks['roc_auc']:.3f}")
    axes[1].step([r['recall'] for r in ranks['pr_points']], [r['precision'] for r in ranks['pr_points']], where='pre')
    axes[1].axhline(ranks['pr_prevalence_baseline'], linestyle='--', color='gray')
    axes[1].set(xlabel='FAIL recall', ylabel='FAIL precision', title=f"AP={ranks['average_precision']:.3f}")
    axes[2].plot([r['threshold'] for r in sweep], [r['false_pass_rate'] for r in sweep], label='false pass / FAIL support')
    axes[2].plot([r['threshold'] for r in sweep], [r['false_fail_rate'] for r in sweep], label='false reject / PASS support')
    axes[2].axvline(.5, linestyle='--', color='gray'); axes[2].legend()
    axes[2].set(xlabel='Diagnostic threshold (not deployed)', ylabel='Rate')
    fig.suptitle('Reused development Val only; no operating threshold selected')
    fig.tight_layout(); fig.savefig(output / 'roc_pr_thresholds.png', dpi=150); plt.close(fig)


def feature_plot(output, coordinates, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, field in zip(axes, ['label', 'capture_batch']):
        for value in sorted({r[field] for r in rows}):
            for split, marker in [('train', 'o'), ('val', 'x')]:
                mask = np.array([r[field] == value and r['split'] == split for r in rows])
                if mask.any():
                    ax.scatter(coordinates[mask, 0], coordinates[mask, 1], label=f'{value}/{split}', marker=marker, s=18, alpha=.7)
        ax.legend(fontsize=7); ax.set(xlabel='PC1 (fit Train only)', ylabel='PC2', title=field)
    fig.suptitle('Frozen 512-D features: descriptive projection, NOT causal evidence')
    fig.tight_layout(); fig.savefig(output / 'feature_pca.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='members/member06_research/runs/dev_baseline_v1')
    parser.add_argument('--output', required=True)
    parser.add_argument('--features', action='store_true', help='forward frozen Train/Val backbone only; no classifier fitting')
    args = parser.parse_args()
    diagnose(args.run, args.output, args.features)


if __name__ == '__main__':
    main()
