"""Backend-neutral UI bridge; no torch import and no fabricated boxes/counts."""
import math
from typing import Protocol, Sequence

from .config import normalize_counts


class ClassifierBackend(Protocol):
    class_to_idx: dict
    evidence_status: str

    def score_batch(self, images: Sequence) -> list[list[float]]:
        """Return finite uncalibrated softmax vectors, in class index order."""
        ...


def result(status, **extra):
    return {"status": status, "predicted_label": None, "prediction_score": None,
            "score_type": "uncalibrated_softmax", **extra}


class ClassificationAdapter:
    def __init__(self, backend: ClassifierBackend, supported_counts, counts_rule_status):
        self.backend = backend
        self.supported_counts = normalize_counts(supported_counts)
        self.counts_rule_status = counts_rule_status

    def predict(self, image, expected_counts=None):
        return self.predict_batch([image], expected_counts)[0]

    def predict_batch(self, images, expected_counts=None):
        # One requested count vector per batch. Per-image counts can call predict individually.
        images = list(images)
        try:
            counts = self.supported_counts if expected_counts is None else normalize_counts(expected_counts)
        except (ValueError, TypeError) as exc:
            return [result("INPUT_ERROR", error=str(exc)) for _ in images]
        if counts != self.supported_counts:
            return [result("UNSUPPORTED_COUNTS", supported_counts=self.supported_counts) for _ in images]
        # Import image support only at call time to preserve framework-only imports.
        from .image_io import read_image
        outputs, valid, indices = [None] * len(images), [], []
        for i, image in enumerate(images):
            try:
                valid.append(read_image(image))
                indices.append(i)
            except (ValueError, OSError) as exc:
                outputs[i] = result("INPUT_ERROR", error=str(exc))
        if valid:
            scores = self.backend.score_batch(valid)
            if len(scores) != len(valid):
                raise RuntimeError("backend batch length mismatch")
            labels = {v: k for k, v in self.backend.class_to_idx.items()}
            for index, row in zip(indices, scores):
                if len(row) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in row) or abs(sum(row) - 1) > 1e-5:
                    raise RuntimeError("backend returned invalid softmax")
                pred = max(range(2), key=row.__getitem__)
                outputs[index] = result(self.backend.evidence_status if self.backend.evidence_status in {"SMOKE_ONLY", "DEV_ONLY"} else "OK",
                                        predicted_label=labels[pred], prediction_score=row[pred],
                                        class_scores={labels[i]: row[i] for i in range(2)},
                                        evidence_status=self.backend.evidence_status,
                                        expected_counts=list(counts), counts_source="configured_fixed_rule" if expected_counts is None else "caller",
                                        counts_rule_status=self.counts_rule_status)
        return outputs
