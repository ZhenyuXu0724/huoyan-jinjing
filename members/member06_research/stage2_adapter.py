"""Stage 2 unified inference adapter for Member 6 Baseline 2 (Frozen ResNet18 + Linear).

Supports:
1. Member 7 live camera web server (`members/member07_app/server.py` via `load(config, root)` -> `fn(jpeg_bytes)`).
2. Course / Member 7 CLI single-image & batch interface (`predict`, `predict_batch`, `export_batch`).
3. Direct Python integration with file paths, PIL.Image, raw JPEG/PNG bytes, RGB numpy arrays, and OpenCV BGR frames.

Guarantees:
- Model is initialized once per adapter/cache entry; no internet download on checkpoint load.
- Preprocessing is 100% identical to offline training/testing (`224x224` letterbox, `pad_rgb=[0,0,0]`, ImageNet mean/std).
- Explicit RGB/BGR handling; `model.eval()` and `torch.inference_mode()` always active.
- Returns true `PASS`/`FAIL` decision and uncalibrated `FAIL` probability `P(FAIL)`.
- Returns `UNSUPPORTED_COUNTS` when `expected_counts != [2, 2, 2, 2, 2]`.
- Never fabricates bounding boxes, per-class counts, or `non_catalogue_count`.
- Returns `INPUT_ERROR` (or raises explicit error for `server.py` HTTP 400) on invalid/corrupted images.
- Model path is configurable via `config["weights"]`, CLI `--checkpoint`, or `MEMBER06_CHECKPOINT` (no machine-fixed absolute path).
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import io
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

MEMBER_DIR = Path(__file__).resolve().parent
REPO_ROOT = MEMBER_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from members.member06_research.config import normalize_counts  # noqa: E402
from members.member06_research.data import Preprocess  # noqa: E402
from members.member06_research.runtime import file_sha256, load_checkpoint, write_json  # noqa: E402

DEFAULT_FROZEN_WEIGHT_REL = (
    "members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt"
)
DEFAULT_FREEZE_MANIFEST_REL = "members/member06_research/stage2_release/freeze_manifest.json"
MODEL_VERSION = "baseline2_final_v4_seed42_ep7"
SCORE_TYPE = "uncalibrated_fail_softmax_prob"
IMAGE_CSV_COLUMNS = [
    "image_id",
    "model_id",
    "status",
    "predicted_label",
    "prediction_score",
    "score_type",
    "non_catalogue_count",
]


def resolve_checkpoint_path(
    checkpoint: str | Path | None = None,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Resolve checkpoint path portably without relying on any machine-specific absolute path."""
    raw = checkpoint if checkpoint is not None else os.environ.get("MEMBER06_CHECKPOINT")
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        search_roots = []
        if base_dir is not None:
            search_roots.append(Path(base_dir).resolve())
        search_roots.extend([Path.cwd().resolve(), REPO_ROOT, MEMBER_DIR])
        for root in search_roots:
            probe = (root / candidate).resolve()
            if probe.is_file():
                return probe
        return ((Path(base_dir).resolve() if base_dir is not None else REPO_ROOT) / candidate).resolve()
    return (REPO_ROOT / DEFAULT_FROZEN_WEIGHT_REL).resolve()


def decode_rgb_image(source: Any, *, color_order: str = "rgb") -> Image.Image:
    """Decode path, PIL.Image, raw image bytes, or RGB/BGR uint8 ndarray into a clean RGB PIL.Image.

    Does not apply EXIF orientation transpose, matching offline training/evaluation (`image_io.read_image`).
    """
    order = color_order.lower()
    if order not in {"rgb", "bgr"}:
        raise ValueError(f"color_order must be 'rgb' or 'bgr', got {color_order!r}")

    if isinstance(source, Image.Image):
        image = source.convert("RGB").copy()
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"image file not found: {p}")
        with Image.open(p) as opened:
            image = opened.convert("RGB").copy()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
        if not raw:
            raise ValueError("empty image bytes")
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                image = opened.convert("RGB").copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"invalid or corrupted image bytes: {exc}") from exc
    elif isinstance(source, np.ndarray):
        if source.dtype != np.uint8:
            raise ValueError(f"numpy image must have dtype uint8, got {source.dtype}")
        if source.ndim == 2:
            if min(source.shape) < 1:
                raise ValueError("empty grayscale image array")
            image = Image.fromarray(source, mode="L").convert("RGB")
        elif source.ndim == 3:
            h, w, c = source.shape
            if min(h, w) < 1:
                raise ValueError("empty color image array")
            if c == 3:
                rgb = np.ascontiguousarray(source[..., ::-1]) if order == "bgr" else np.ascontiguousarray(source)
                image = Image.fromarray(rgb, mode="RGB")
            elif c == 4:
                rgb = (
                    np.ascontiguousarray(source[..., [2, 1, 0]])
                    if order == "bgr"
                    else np.ascontiguousarray(source[..., :3])
                )
                image = Image.fromarray(rgb, mode="RGB")
            else:
                raise ValueError(f"numpy image must have 1, 3, or 4 channels, got {c}")
        else:
            raise ValueError(f"numpy image must be 2D or 3D, got ndim={source.ndim}")
    else:
        raise ValueError(f"unsupported image input type: {type(source).__name__}")

    if min(image.size) < 1:
        raise ValueError("decoded image has zero width or height")
    return image


def _error_record(status: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "predicted_label": None,
        "prediction_score": None,
        "fail_probability": None,
        "pass_probability": None,
        "score_type": SCORE_TYPE,
        "non_catalogue_count": None,
        **extra,
    }


class Stage2ResNetAdapter:
    """Frozen ResNet18 + Linear(512, 2) Stage 2 production/evaluation adapter."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str = "auto",
        batch_size: int = 16,
        expected_sha256: str | None = None,
        allow_smoke: bool = False,
        allow_dev_only: bool = False,
        base_dir: Path | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.checkpoint_path = resolve_checkpoint_path(checkpoint, base_dir=base_dir)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Frozen Baseline 2 checkpoint not found at {self.checkpoint_path}. "
                f"Place the official weight there or set MEMBER06_CHECKPOINT / config['weights']."
            )
        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)

        # Verify against expected_sha256 or official freeze_manifest.json when loading the default release path
        target_sha = expected_sha256
        if target_sha is None and self.checkpoint_path == (REPO_ROOT / DEFAULT_FROZEN_WEIGHT_REL).resolve():
            fm_path = REPO_ROOT / DEFAULT_FREEZE_MANIFEST_REL
            if fm_path.is_file():
                fm = json.loads(fm_path.read_text(encoding="utf-8"))
                target_sha = fm.get("selected_model", {}).get("checkpoint_sha256")
        if target_sha is not None and self.checkpoint_sha256 != target_sha:
            raise ValueError(
                f"Checkpoint SHA256 mismatch for {self.checkpoint_path}: "
                f"expected {target_sha}, got {self.checkpoint_sha256}"
            )

        self.model, self.config, self.device, self._payload = load_checkpoint(
            self.checkpoint_path,
            device=device,
            allow_smoke=allow_smoke,
            allow_dev_only=allow_dev_only,
        )
        self.model.eval()
        self.transform = Preprocess(self.config)
        self.batch_size = batch_size
        self.class_to_idx = dict(self.config.class_to_idx)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.pass_idx = self.class_to_idx["PASS"]
        self.fail_idx = self.class_to_idx["FAIL"]
        self.supported_counts = list(self.config.supported_counts)
        self.evidence_status = self.config.evidence_status
        self.selected_epoch = self._payload.get("epoch")

    @torch.inference_mode()
    def _score_pil_batch(self, pil_images: Sequence[Image.Image]) -> list[list[float]]:
        self.model.eval()
        probs: list[list[float]] = []
        for start in range(0, len(pil_images), self.batch_size):
            chunk = pil_images[start : start + self.batch_size]
            tensors = torch.stack([self.transform(img) for img in chunk]).to(self.device)
            batch_probs = self.model(tensors).softmax(dim=1).detach().cpu().tolist()
            probs.extend(batch_probs)
        return probs

    def predict(
        self,
        image: Any,
        expected_counts: Any = None,
        *,
        color_order: str = "rgb",
    ) -> dict[str, Any]:
        return self.predict_batch([image], expected_counts=expected_counts, color_order=color_order)[0]

    def predict_batch(
        self,
        images: Sequence[Any],
        expected_counts: Any = None,
        *,
        color_order: str = "rgb",
    ) -> list[dict[str, Any]]:
        items = list(images)
        if not items:
            return []
        try:
            counts = self.supported_counts if expected_counts is None else normalize_counts(expected_counts)
        except (ValueError, TypeError) as exc:
            return [_error_record("INPUT_ERROR", error=str(exc)) for _ in items]

        if counts != self.supported_counts:
            return [
                _error_record("UNSUPPORTED_COUNTS", supported_counts=list(self.supported_counts))
                for _ in items
            ]

        outputs: list[dict[str, Any] | None] = [None] * len(items)
        valid_pils: list[Image.Image] = []
        valid_indices: list[int] = []
        for idx, item in enumerate(items):
            try:
                valid_pils.append(decode_rgb_image(item, color_order=color_order))
                valid_indices.append(idx)
            except (OSError, ValueError, TypeError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                outputs[idx] = _error_record("INPUT_ERROR", error=str(exc))

        if valid_pils:
            scores = self._score_pil_batch(valid_pils)
            if len(scores) != len(valid_pils):
                raise RuntimeError("backend batch score length mismatch")
            ok_status = (
                self.evidence_status if self.evidence_status in {"SMOKE_ONLY", "DEV_ONLY"} else "OK"
            )
            for out_idx, row in zip(valid_indices, scores):
                if (
                    len(row) != 2
                    or any(not math.isfinite(x) or not (0.0 <= x <= 1.0) for x in row)
                    or abs(sum(row) - 1.0) > 1e-4
                ):
                    raise RuntimeError(f"invalid softmax vector: {row}")
                # Frozen argmax decision rule: index 0 = PASS, index 1 = FAIL; tie at 0.5 -> index 0 (PASS)
                pred_idx = max(range(2), key=row.__getitem__)
                pass_prob = float(row[self.pass_idx])
                fail_prob = float(row[self.fail_idx])
                outputs[out_idx] = {
                    "status": ok_status,
                    "predicted_label": self.idx_to_class[pred_idx],
                    "prediction_score": fail_prob,
                    "fail_probability": fail_prob,
                    "pass_probability": pass_prob,
                    "score_type": SCORE_TYPE,
                    "class_scores": {"PASS": pass_prob, "FAIL": fail_prob},
                    "expected_counts": list(counts),
                    "counts_source": "configured_fixed_rule" if expected_counts is None else "caller",
                    "non_catalogue_count": None,
                    "model_version": MODEL_VERSION,
                }
        return [r for r in outputs if r is not None]

    def export_batch(
        self,
        input_dir: str | Path,
        counts_file: str | Path,
        output_dir: str | Path,
        *,
        model_id: str = "baseline2",
    ) -> list[dict[str, Any]]:
        """Export shared-contract `images.csv` + `run_config.json` without touching ground truth."""
        out = Path(output_dir)
        if out.exists():
            raise FileExistsError(f"output directory already exists: {out}")
        in_root = Path(input_dir).resolve()
        if not in_root.is_dir():
            raise FileNotFoundError(f"input_dir does not exist: {in_root}")

        raw_reqs = json.loads(Path(counts_file).read_text(encoding="utf-8-sig"))
        if not isinstance(raw_reqs, list) or not raw_reqs:
            raise ValueError("counts_file must be a non-empty JSON list")
        cases: dict[str, list[int]] = {}
        for item in raw_reqs:
            if not isinstance(item, dict) or set(item) != {"image_id", "expected_counts"}:
                raise ValueError("each counts_file entry must have exact keys {'image_id', 'expected_counts'}")
            img_id = item["image_id"]
            if not isinstance(img_id, str) or not img_id or img_id in {".", ".."} or any(c in img_id for c in "/\\:") or img_id in cases:
                raise ValueError(f"invalid or duplicate image_id: {img_id!r}")
            cases[img_id] = normalize_counts(item["expected_counts"])

        found_paths: dict[str, Path] = {}
        for p in sorted(in_root.rglob("*")):
            if p.is_file() and p.name in cases:
                resolved = p.resolve()
                if not resolved.is_relative_to(in_root) or p.name in found_paths:
                    raise ValueError(f"ambiguous or symlink-escaped input file: {p.name}")
                found_paths[p.name] = resolved

        names = list(cases)
        results_by_name: dict[str, dict[str, Any]] = {}
        times_ms: dict[str, float] = {}
        image_hashes: dict[str, str] = {}

        for start in range(0, len(names), self.batch_size):
            by_req: dict[tuple[int, ...], list[str]] = {}
            for name in names[start : start + self.batch_size]:
                by_req.setdefault(tuple(cases[name]), []).append(name)
            for req_tuple, group_names in by_req.items():
                t0 = perf_counter()
                batch_inputs = [found_paths.get(n, in_root / n) for n in group_names]
                preds = self.predict_batch(batch_inputs, expected_counts=list(req_tuple))
                per_img_ms = (perf_counter() - t0) * 1000.0 / len(group_names)
                for n, pred in zip(group_names, preds):
                    results_by_name[n] = pred
                    times_ms[n] = round(per_img_ms, 4)
                    if n in found_paths:
                        image_hashes[n] = file_sha256(found_paths[n])

        csv_rows: list[dict[str, Any]] = []
        for n in names:
            res = results_by_name[n]
            csv_rows.append(
                {
                    "image_id": n,
                    "model_id": model_id,
                    "status": res["status"],
                    "predicted_label": res["predicted_label"],
                    "prediction_score": res["prediction_score"],
                    "score_type": res["score_type"],
                    "non_catalogue_count": None,
                }
            )

        out.mkdir(parents=True, exist_ok=False)
        with (out / "images.csv").open("x", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=IMAGE_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(csv_rows)

        write_json(
            out / "run_config.json",
            {
                "model_id": model_id,
                "model_version": MODEL_VERSION,
                "model_sha256": self.checkpoint_sha256,
                "selected_epoch": self.selected_epoch,
                "config": self.config.to_dict(),
                "evidence_status": self.evidence_status,
                "supported_expected_counts": self.supported_counts,
                "counts_rule_status": self.config.counts_rule_status,
                "input_expected_counts": {
                    n: {str(i + 1): v for i, v in enumerate(vec)} for n, vec in cases.items()
                },
                "counts_file_sha256": file_sha256(counts_file),
                "image_sha256": image_hashes,
                "score_type": SCORE_TYPE,
                "score_semantics": "uncalibrated softmax probability of FAIL class P(FAIL) in [0, 1]; >0.5 predicts FAIL, <=0.5 predicts PASS",
                "decision_rule": "argmax over [P(PASS), P(FAIL)]; ties at 0.5 choose class index 0 (PASS)",
                "capabilities": {"boxes": False, "actual_counts": False, "non_catalogue_count": False},
                "gt_accessed_during_inference": False,
                "offline_processing_ms": times_ms,
                "timing_scope": "amortized offline batch IO + letterbox preprocess + ResNet18 forward; excludes camera/UI/model load",
            },
        )
        return csv_rows


def load(config: dict[str, Any], root: Path):
    """Member 7 `members/member07_app/server.py` adapter hook.

    `server.py` calls `fn = module.load(config, ROOT)` once and caches `fn`.
    `fn(raw_jpeg_bytes)` returns `{"kind": "image", "decision": ..., "fixed_expected": [2,2,2,2,2], "score": P(FAIL)}`.
    """
    weights_cfg = config.get("weights")
    device = config.get("device", "auto")
    expected_sha = config.get("sha256")
    allow_smoke = bool(config.get("allow_smoke", False))
    allow_dev_only = bool(config.get("allow_dev_only", False))

    adapter = Stage2ResNetAdapter(
        weights_cfg,
        device=device,
        batch_size=1,
        expected_sha256=expected_sha,
        allow_smoke=allow_smoke,
        allow_dev_only=allow_dev_only,
        base_dir=Path(root),
    )

    def run(raw_bytes: bytes) -> dict[str, Any]:
        if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            raise ValueError("摄像头服务输入必须为图像字节流 (bytes)")
        res = adapter.predict(raw_bytes)
        if res["status"] != "OK" and res["status"] not in {"SMOKE_ONLY", "DEV_ONLY"}:
            raise ValueError(res.get("error") or f"模型推理失败: {res['status']}")
        return {
            "kind": "image",
            "decision": res["predicted_label"],
            "fixed_expected": list(adapter.supported_counts),
            "score": round(float(res["fail_probability"]), 6),
        }

    return run


@lru_cache(maxsize=2)
def _cached_default_adapter(checkpoint_str: str, device: str) -> Stage2ResNetAdapter:
    ckpt = checkpoint_str if checkpoint_str else None
    return Stage2ResNetAdapter(ckpt, device=device)


def predict(image_path: Any, expected_counts: Any = None, *, color_order: str = "rgb") -> dict[str, Any]:
    """Module-level single-image entry point (loads model once via LRU cache)."""
    ckpt_env = os.environ.get("MEMBER06_CHECKPOINT", "")
    dev_env = os.environ.get("MEMBER06_DEVICE", "auto")
    try:
        adapter = _cached_default_adapter(ckpt_env, dev_env)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return _error_record("MODEL_UNAVAILABLE", error=str(exc))
    return adapter.predict(image_path, expected_counts=expected_counts, color_order=color_order)


def predict_batch(
    image_paths: Sequence[Any],
    expected_counts: Any = None,
    *,
    color_order: str = "rgb",
) -> list[dict[str, Any]]:
    """Module-level batch entry point (loads model once via LRU cache)."""
    items = list(image_paths)
    if not items:
        return []
    ckpt_env = os.environ.get("MEMBER06_CHECKPOINT", "")
    dev_env = os.environ.get("MEMBER06_DEVICE", "auto")
    try:
        adapter = _cached_default_adapter(ckpt_env, dev_env)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return [_error_record("MODEL_UNAVAILABLE", error=str(exc)) for _ in items]
    return adapter.predict_batch(items, expected_counts=expected_counts, color_order=color_order)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path (defaults to Stage 2 frozen weight)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="Single image path to predict")
    mode.add_argument("--input-dir", help="Directory of images for shared-contract batch export")
    parser.add_argument("--counts-file", help="Required with --input-dir: JSON list of {image_id, expected_counts}")
    parser.add_argument("--expected-counts", help="Optional JSON counts vector/dict for --image, e.g. '[2,2,2,2,2]'")
    parser.add_argument("--output", required=True, help="Output JSON file (for --image) or directory (for --input-dir)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--allow-dev-only", action="store_true")
    args = parser.parse_args()

    try:
        adapter = Stage2ResNetAdapter(
            args.checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            allow_smoke=args.allow_smoke,
            allow_dev_only=args.allow_dev_only,
        )
        if args.image:
            out_path = Path(args.output)
            if out_path.exists():
                raise FileExistsError(f"output already exists: {out_path}")
            exp = json.loads(args.expected_counts) if args.expected_counts else None
            res = adapter.predict(args.image, expected_counts=exp)
            write_json(
                out_path,
                {
                    "model_id": "baseline2",
                    "model_version": MODEL_VERSION,
                    "checkpoint_sha256": adapter.checkpoint_sha256,
                    "image_id": Path(args.image).name,
                    "result": res,
                },
            )
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return 0 if res["status"] in {"OK", "SMOKE_ONLY", "DEV_ONLY"} else 2

        if not args.counts_file:
            parser.error("--counts-file is required when using --input-dir")
        rows = adapter.export_batch(args.input_dir, args.counts_file, args.output)
        print(f"Exported {len(rows)} predictions to {args.output}")
        return 0 if all(r["status"] in {"OK", "SMOKE_ONLY", "DEV_ONLY"} for r in rows) else 2
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"STAGE2_ADAPTER_FAILED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
