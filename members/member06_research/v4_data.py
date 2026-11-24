"""Authoritative manifest-driven intake for members/member03_data/final_training_v4.

Guarantees:
1. Uses manifests/resnet/train.csv and manifests/resnet/val.csv as authoritative inputs (never scans images/ recursively).
2. No hardcoded row counts in loader logic; counts are derived from approved manifests and cross-verified against split_manifest.csv.
3. Enforces strict Test isolation: refuses any path under images/test/ and verifies zero hash/scene overlap with Test metadata.
4. Prepares optional 351-image manifest (249 primary + 102 optional_train_aug) for potential follow-up runs without activating it by default.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

V4_PROVENANCE_SCHEMA = "member06.final_training_v4_intake.v1"
UPSTREAM_V4_COMMIT = "bde8ea7116761bc8ee7fca1d47c336126daa995f"
DEFAULT_V4_ROOT = "members/member03_data/final_training_v4"
DEFAULT_V4_INTAKE_DIR = "members/member06_research/data/final_training_v4_intake"


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_utf8sig(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _resolve_safe_v4_image(manifest_path: Path, rel_image_path: str, v4_root: Path, expected_split: str) -> Path:
    if expected_split not in {"train", "val"}:
        raise ValueError(f"forbidden split {expected_split!r}: Test or unknown partition cannot be loaded for training/selection")
    if not isinstance(rel_image_path, str) or "\\" in rel_image_path:
        raise ValueError(f"invalid manifest image_path: {rel_image_path!r}")
    candidate = (manifest_path.parent / rel_image_path).resolve()
    v4_resolved = v4_root.resolve()
    if not candidate.is_relative_to(v4_resolved / "images"):
        raise ValueError(f"manifest image escapes final_training_v4/images: {rel_image_path!r}")
    if candidate.is_relative_to(v4_resolved / "images" / "test"):
        raise ValueError(f"FORBIDDEN_TEST_ACCESS: {rel_image_path!r} resolves inside images/test")
    if expected_split == "train":
        allowed_roots = (v4_resolved / "images" / "train", v4_resolved / "images" / "train_augmentations")
    else:
        allowed_roots = (v4_resolved / "images" / "val",)
    if not any(candidate.is_relative_to(ar) for ar in allowed_roots):
        raise ValueError(f"path {rel_image_path!r} violates allowed {expected_split} directories")
    if candidate.is_symlink():
        raise ValueError(f"symlink not permitted: {candidate}")
    return candidate


def _anomaly_mechanism(anomaly: str) -> str:
    if anomaly == "none":
        return "PASS_normal"
    if anomaly == "pure_missing":
        return "pure_missing"
    if anomaly == "empty_scene":
        return "empty_scene"
    if anomaly.startswith("non_catalogue_only"):
        return "non_catalogue_only"
    if anomaly == "non_catalogue_plus_missing":
        return "non_catalogue_plus_missing"
    return anomaly or "other"


def prepare_v4_intake(
    v4_root: str | Path = DEFAULT_V4_ROOT,
    output_dir: str | Path = DEFAULT_V4_INTAKE_DIR,
    upstream_commit: str = UPSTREAM_V4_COMMIT,
) -> dict:
    root = Path(v4_root).resolve()
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"intake output directory already exists and is non-empty: {out}")

    req_files = (
        root / "README.md",
        root / "classes.json",
        root / "split_manifest.csv",
        root / "manifests/resnet/train.csv",
        root / "manifests/resnet/val.csv",
        root / "manifests/resnet/test.csv",
    )
    for p in req_files:
        if not p.is_file():
            raise FileNotFoundError(f"missing required final_training_v4 file: {p}")

    classes_doc = json.loads((root / "classes.json").read_text(encoding="utf-8"))
    if classes_doc.get("expected_target_counts") != [2, 2, 2, 2, 2]:
        raise ValueError("unexpected expected_target_counts in classes.json")

    split_rows = _read_csv_utf8sig(root / "split_manifest.csv")
    by_dest: dict[str, dict] = {r["dest_image"]: r for r in split_rows if r.get("dest_image")}
    test_sha_set = {r["image_sha256"] for r in split_rows if r["split"] == "test"}
    test_scene_set = {r["scene_group"] for r in split_rows if r["split"] == "test"}

    train_csv = root / "manifests/resnet/train.csv"
    val_csv = root / "manifests/resnet/val.csv"
    train_manifest_rows = _read_csv_utf8sig(train_csv)
    val_manifest_rows = _read_csv_utf8sig(val_csv)
    test_manifest_rows = _read_csv_utf8sig(root / "manifests/resnet/test.csv")

    def _validate_manifest_subset(m_path: Path, m_rows: list[dict], expected_split: str, allow_aug: bool) -> list[dict]:
        checked = []
        seen_paths = set()
        for r in m_rows:
            if set(r.keys()) != {"image_path", "label", "scene_id", "split", "data_kind"}:
                raise ValueError(f"unexpected ResNet manifest columns in {m_path}: {list(r.keys())}")
            if r["split"] != expected_split or r["data_kind"] != "REAL" or r["label"] not in {"PASS", "FAIL"}:
                raise ValueError(f"invalid manifest row in {m_path}: {r}")
            abs_img = _resolve_safe_v4_image(m_path, r["image_path"], root, expected_split)
            if abs_img in seen_paths:
                raise ValueError(f"duplicate image in {m_path}: {abs_img}")
            seen_paths.add(abs_img)
            dest_rel = abs_img.relative_to(root).as_posix()
            meta = by_dest.get(dest_rel)
            if meta is None:
                raise ValueError(f"manifest image missing from split_manifest.csv: {dest_rel}")
            allowed_usages = {"primary", "optional_train_aug"} if allow_aug else {"primary"}
            if meta["usage"] not in allowed_usages or meta["split"] != expected_split:
                raise ValueError(f"usage/split mismatch for {dest_rel}: {meta['usage']} / {meta['split']}")
            if meta["label"] != r["label"] or meta["scene_group"] != r["scene_id"]:
                raise ValueError(f"label/scene disagreement with split_manifest.csv on {dest_rel}")
            actual_sha = sha256_file(abs_img)
            if abs_img.stat().st_size != int(meta["image_bytes"]) or actual_sha != meta["image_sha256"]:
                raise ValueError(f"image bytes/SHA256 mismatch on {dest_rel}")
            if actual_sha in test_sha_set or r["scene_id"] in test_scene_set:
                raise ValueError(f"Train/Val overlaps Test SHA256 or scene_group: {dest_rel}")
            checked.append({**r, "dest_image": dest_rel, "image_sha256": actual_sha, "meta": meta})
        return checked

    verified_train = _validate_manifest_subset(train_csv, train_manifest_rows, "train", allow_aug=False)
    verified_val = _validate_manifest_subset(val_csv, val_manifest_rows, "val", allow_aug=False)

    if {x["image_sha256"] for x in verified_train} & {x["image_sha256"] for x in verified_val}:
        raise ValueError("Train/Val image_sha256 overlap")
    if {x["scene_id"] for x in verified_train} & {x["scene_id"] for x in verified_val}:
        raise ValueError("Train/Val scene_id overlap")

    # Also verify optional 102 training augmentations on disk and build ready-to-use 5-col manifest (not used in Phase-1)
    opt_aug_rows = [r for r in split_rows if r["usage"] == "optional_train_aug"]
    for ar in opt_aug_rows:
        if ar["split"] != "train":
            raise ValueError(f"optional_train_aug not in train split: {ar['filename']}")
        abs_aug = (root / ar["dest_image"]).resolve()
        if not abs_aug.is_relative_to(root / "images/train_augmentations"):
            raise ValueError(f"unexpected optional_train_aug path: {ar['dest_image']}")
        actual_sha = sha256_file(abs_aug)
        if abs_aug.stat().st_size != int(ar["image_bytes"]) or actual_sha != ar["image_sha256"]:
            raise ValueError(f"optional_train_aug hash mismatch: {ar['dest_image']}")
        if actual_sha in test_sha_set or ar["scene_group"] in test_scene_set:
            raise ValueError(f"optional_train_aug overlaps Test: {ar['dest_image']}")

    out.mkdir(parents=True, exist_ok=False)

    # Write optional combined manifest (primary Train + optional_train_aug) with relative paths to v4 images/
    opt_manifest_path = out / "train_with_optional_aug.csv"
    with opt_manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "label", "scene_id", "split", "data_kind"])
        writer.writeheader()
        for x in verified_train:
            rel_from_out = Path(
                *([".."] * len(out.relative_to(root.parents[2]).parts)),
                root.relative_to(root.parents[2]),
                x["dest_image"],
            ).as_posix()
            writer.writerow({
                "image_path": rel_from_out,
                "label": x["label"],
                "scene_id": x["scene_id"],
                "split": "train",
                "data_kind": "REAL",
            })
        for ar in sorted(opt_aug_rows, key=lambda r: r["filename"]):
            rel_from_out = Path(
                *([".."] * len(out.relative_to(root.parents[2]).parts)),
                root.relative_to(root.parents[2]),
                ar["dest_image"],
            ).as_posix()
            writer.writerow({
                "image_path": rel_from_out,
                "label": ar["label"],
                "scene_id": ar["scene_group"],
                "split": "train",
                "data_kind": "REAL",
            })

    # Verify the generated optional manifest resolves cleanly too
    _validate_manifest_subset(opt_manifest_path, _read_csv_utf8sig(opt_manifest_path), "train", allow_aug=True)

    provenance = {
        "schema_version": V4_PROVENANCE_SCHEMA,
        "dataset_id": "final_training_v4",
        "upstream_commit": upstream_commit,
        "v4_root": str(root),
        "source_file_sha256": {
            "README.md": sha256_file(root / "README.md"),
            "classes.json": sha256_file(root / "classes.json"),
            "split_manifest.csv": sha256_file(root / "split_manifest.csv"),
            "manifests/resnet/train.csv": sha256_file(train_csv),
            "manifests/resnet/val.csv": sha256_file(val_csv),
            "manifests/resnet/test.csv": sha256_file(root / "manifests/resnet/test.csv"),
        },
        "prepared_optional_manifest": {
            "path": str(opt_manifest_path),
            "sha256": sha256_file(opt_manifest_path),
            "total_rows": len(verified_train) + len(opt_aug_rows),
        },
        "verified_summary": {
            "train_primary_total": len(verified_train),
            "train_primary_class_counts": dict(collections.Counter(x["label"] for x in verified_train)),
            "val_primary_total": len(verified_val),
            "val_primary_class_counts": dict(collections.Counter(x["label"] for x in verified_val)),
            "optional_train_aug_total": len(opt_aug_rows),
            "optional_train_aug_class_counts": dict(collections.Counter(r["label"] for r in opt_aug_rows)),
            "test_manifest_metadata_only_total": len(test_manifest_rows),
            "test_manifest_metadata_only_class_counts": dict(collections.Counter(r["label"] for r in test_manifest_rows)),
        },
        "supported_counts": [2, 2, 2, 2, 2],
        "test_content_read": False,
    }
    prov_path = out / "provenance.json"
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return provenance


def load_verified_v4_manifests(
    train_manifest_path: str | Path,
    val_manifest_path: str | Path,
    provenance_path: str | Path,
    *,
    allow_optional_aug: bool = False,
) -> tuple[dict, list[dict], list[dict]]:
    """Load authoritative Train & Val manifests, verify source hashes, and attach slice metadata from split_manifest.csv."""
    prov_p = Path(provenance_path).resolve()
    if not prov_p.is_file():
        raise FileNotFoundError(f"missing v4 provenance file: {prov_p}")
    prov = json.loads(prov_p.read_text(encoding="utf-8"))
    if prov.get("schema_version") != V4_PROVENANCE_SCHEMA or prov.get("test_content_read") is not False:
        raise ValueError("invalid or unsafe v4 provenance")

    v4_root = Path(prov["v4_root"]).resolve()
    for rel_f, exp_sha in prov["source_file_sha256"].items():
        if sha256_file(v4_root / rel_f) != exp_sha:
            raise ValueError(f"final_training_v4 file modified since intake: {rel_f}")

    split_rows = _read_csv_utf8sig(v4_root / "split_manifest.csv")
    by_dest = {r["dest_image"]: r for r in split_rows if r.get("dest_image")}
    test_sha_set = {r["image_sha256"] for r in split_rows if r["split"] == "test"}

    def _load_one(m_path_in: str | Path, expected_split: str, allow_aug: bool) -> list[dict]:
        mp = Path(m_path_in).resolve()
        raw_rows = _read_csv_utf8sig(mp)
        if not raw_rows:
            raise ValueError(f"empty manifest: {mp}")
        out_rows = []
        seen = set()
        for idx, r in enumerate(raw_rows, 1):
            if set(r.keys()) != {"image_path", "label", "scene_id", "split", "data_kind"}:
                raise ValueError(f"manifest columns must be image_path,label,scene_id,split,data_kind in {mp}")
            if r["split"] != expected_split or r["label"] not in {"PASS", "FAIL"}:
                raise ValueError(f"invalid split/label in {mp}: {r}")
            abs_p = _resolve_safe_v4_image(mp, r["image_path"], v4_root, expected_split)
            if abs_p in seen:
                raise ValueError(f"duplicate image path in {mp}: {abs_p}")
            seen.add(abs_p)
            dest_rel = abs_p.relative_to(v4_root).as_posix()
            meta = by_dest.get(dest_rel)
            if meta is None:
                raise ValueError(f"unregistered image in {mp}: {dest_rel}")
            allowed_u = {"primary", "optional_train_aug"} if allow_aug else {"primary"}
            if meta["usage"] not in allowed_u:
                raise ValueError(f"disallowed usage={meta['usage']!r} for {dest_rel} (allow_optional_aug={allow_aug})")
            actual_sha = sha256_file(abs_p)
            if actual_sha != meta["image_sha256"] or actual_sha in test_sha_set:
                raise ValueError(f"image hash mismatch or Test collision on {dest_rel}")
            nc_count = sum(int(meta[k]) for k in "SHNWB")
            t_counts = [int(c) for c in meta["target_counts"].split("|")]
            missing_units = sum(max(0, 2 - c) for c in t_counts)
            out_rows.append({
                "sample_id": f"V4_{expected_split.upper()}_{idx:04d}_{meta['filename']}",
                "image_filename": meta["filename"],
                "v2_image_relpath": dest_rel,
                "dest_image": dest_rel,
                "path": abs_p,
                "label": r["label"],
                "scene_id": r["scene_id"],
                "split": expected_split,
                "data_kind": r["data_kind"],
                "usage": meta["usage"],
                "data_tier": meta["tier"],
                "source_batch": meta["batch"],
                "condition": meta["condition"],
                "parent_image_id": meta["parent"],
                "non_catalogue_count": nc_count,
                "missing_target_units": missing_units,
                "distractor_breakdown": {k: int(meta[k]) for k in "SHNWB"},
                "anomaly_category": meta["anomaly"],
                "anomaly_mechanism": _anomaly_mechanism(meta["anomaly"]),
                "image_sha256": actual_sha,
            })
        if {x["label"] for x in out_rows} != {"PASS", "FAIL"}:
            raise ValueError(f"partition {expected_split} must contain both PASS and FAIL")
        return out_rows

    train_rows = _load_one(train_manifest_path, "train", allow_optional_aug)
    val_rows = _load_one(val_manifest_path, "val", False)

    if {x["image_sha256"] for x in train_rows} & {x["image_sha256"] for x in val_rows}:
        raise ValueError("Train/Val image_sha256 overlap")
    if {x["scene_id"] for x in train_rows} & {x["scene_id"] for x in val_rows}:
        raise ValueError("Train/Val scene_id overlap")

    return prov, train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--v4-root", default=DEFAULT_V4_ROOT)
    p_prep.add_argument("--output-dir", default=DEFAULT_V4_INTAKE_DIR)
    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--train-manifest", default=f"{DEFAULT_V4_ROOT}/manifests/resnet/train.csv")
    p_ver.add_argument("--val-manifest", default=f"{DEFAULT_V4_ROOT}/manifests/resnet/val.csv")
    p_ver.add_argument("--provenance", default=f"{DEFAULT_V4_INTAKE_DIR}/provenance.json")
    args = parser.parse_args()
    if args.cmd == "prepare":
        res = prepare_v4_intake(args.v4_root, args.output_dir)
        print(json.dumps({"status": "V4_PREPARED_OK", "summary": res["verified_summary"]}, ensure_ascii=False, indent=2))
    else:
        prov, tr, vl = load_verified_v4_manifests(args.train_manifest, args.val_manifest, args.provenance)
        print(json.dumps({
            "status": "V4_VERIFIED_OK",
            "upstream_commit": prov["upstream_commit"],
            "train_rows": len(tr),
            "val_rows": len(vl),
            "test_content_read": prov["test_content_read"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
