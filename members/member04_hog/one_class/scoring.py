"""Reconstruction scores and validation-only threshold selection."""
import numpy as np


def reconstruction_scores(model, vectors):
    errors = vectors.astype(np.float64) - model.predict(vectors)
    return np.mean(errors ** 2, axis=1)


def threshold_from_validation(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if set(labels) != {"PASS", "FAIL"} or not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires both validation labels and finite scores")
    unique = np.unique(scores)
    candidates = np.concatenate(([unique[0]], unique[:-1] + np.diff(unique) / 2,
                                 [np.nextafter(unique[-1], np.inf)]))
    options = []
    for threshold in candidates:
        failed = scores >= threshold
        positive = labels == "FAIL"
        tp = int(np.sum(failed & positive))
        fp = int(np.sum(failed & ~positive))
        fn = int(np.sum(~failed & positive))
        tn = int(np.sum(~failed & ~positive))
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2
        options.append(((f1, balanced, -fp, float(threshold)), float(threshold)))
    return max(options)[1]


