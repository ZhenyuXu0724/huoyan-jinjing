"""Fixed 11 augmentation groups and deterministic one-view-per-epoch features."""
from __future__ import annotations

import numpy as np

from ..features import extract

GEOMETRY = {
    "A0": [[0, False]],
    "A1": [[0, False], [0, True]],
    "A2": [[0, False], [2, True]],
    "A3": [[0, False], [1, False], [2, False], [3, False]],
    "A4": [[0, False], [0, True], [2, True], [2, False]],
    "A5": [[0, False], [1, False], [2, False], [3, False],
           [0, True], [1, True], [2, True], [3, True]],
}
GAMMAS = (0.8, 1.0, 1.2)

# Ordered exactly as in the experiment table: A0-A5, B1, B2, C0, C_without_geometry,
# C_without_clahe. C_without_geometry equals B2 and C_without_clahe equals A2.
GROUPS = {
    "A0": {"geometry": "A0", "gamma": False, "clahe": False},
    "A1": {"geometry": "A1", "gamma": False, "clahe": False},
    "A2": {"geometry": "A2", "gamma": False, "clahe": False},
    "A3": {"geometry": "A3", "gamma": False, "clahe": False},
    "A4": {"geometry": "A4", "gamma": False, "clahe": False},
    "A5": {"geometry": "A5", "gamma": False, "clahe": False},
    "B1": {"geometry": "A0", "gamma": True, "clahe": False},
    "B2": {"geometry": "A0", "gamma": False, "clahe": True},
    "C0": {"geometry": "A2", "gamma": False, "clahe": True},
    "C_without_geometry": {"geometry": "A0", "gamma": False, "clahe": True},
    "C_without_clahe": {"geometry": "A2", "gamma": False, "clahe": False},
}
EQUIVALENT_PAIRS = (("B2", "C_without_geometry"), ("A2", "C_without_clahe"))


def group_order():
    return tuple(GROUPS)


def views(group):
    if group not in GROUPS:
        raise ValueError(f"unknown group: {group}")
    specification = GROUPS[group]
    gammas = GAMMAS if specification["gamma"] else (1.0,)
    return [(turns, mirror, gamma)
            for turns, mirror in GEOMETRY[specification["geometry"]]
            for gamma in gammas]


def augmentation_count(group):
    specification = GROUPS[group]
    return ((len(GEOMETRY[specification["geometry"]]) > 1)
            + int(specification["gamma"]) + int(specification["clahe"]))


def view_count(group):
    return len(views(group))


def base_settings(group):
    return {"size": 256, "cell": 16, "clahe": GROUPS[group]["clahe"],
            "geometry": GROUPS[group]["geometry"],
            "gamma": GROUPS[group]["gamma"], "epochs": 200}


def base_features(images, group):
    settings = base_settings(group)
    return np.stack([extract(image, settings) for image in images])


def variant_features(images, group, scaler):
    settings = base_settings(group)
    matrices = []
    for turns, mirror, gamma in views(group):
        matrix = np.stack([extract(image, settings, turns, mirror, gamma)
                           for image in images])
        matrices.append(scaler.transform(matrix))
    return matrices


def validation_features(images, group, scaler):
    return scaler.transform(base_features(images, group))


def parameter_count(kind, layers, dimension=8100):
    """Trainable parameters of the sklearn MLP with a known input/output width."""
    output = dimension if kind == "ae" else 2
    widths = [dimension, *layers, output]
    count = 0
    for left, right in zip(widths[:-1], widths[1:], strict=True):
        count += left * right + right
    return count


def selection_key(name, rows, metrics=None):
    """Group selection: mean FN, mean FP, fewer augmentations, fewer views, table order.

    `rows` must be per-seed best/last metrics in seed order.
    """
    mean_fn = float(np.mean([row["fn"] for row in rows]))
    mean_fp = float(np.mean([row["fp"] for row in rows]))
    return (mean_fn, mean_fp, augmentation_count(name), view_count(name),
            group_order().index(name))
