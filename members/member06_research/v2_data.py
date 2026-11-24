"""Dataset v2 intake, Train/Val manifest preparation, and deterministic R2 parent schedule.

Strict guarantees:
1. Verifies finalized 438/92/79 Dataset v2 metadata without opening any Test image or Test JSON.
2. Enforces path containment within dataset_v2/train and dataset_v2/val only.
3. Produces deterministic manifests for R1 (294 raw), R2 (294 parents with 1 or 4 variants),
   R3 (438 full), primary Val (74 raw), and post-training robustness Val (18 augmented).
4. Uses SHA-256 (never Python's randomized hash()) for R2 initial variant offsets.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

PREPARED_SCHEMA = "member06.dataset_v2_phase1.v1"
DEFAULT_DATASET_V2_ROOT = "members/member03_data/dataset_v2"
DEFAULT_OUTPUT_DIR = "members/member06_research/data/dataset_v2_phase1"

APPROVED_QUOTAS = {
    ("train", "PASS", "original"): 149,
    ("train", "FAIL", "original"): 145,
    ("train", "PASS", "augmented"): 6,
    ("train", "FAIL", "augmented"): 138,
    ("val", "PASS", "original"): 37,
    ("val", "FAIL", "original"): 37,
    ("val", "PASS", "augmented"): 0,
    ("val", "FAIL", "augmented"): 18,
    ("test", "PASS", "original"): 15,
    ("test", "FAIL", "original"): 16,
    ("test", "PASS", "augmented"): 0,
    ("test", "FAIL", "augmented"): 48,
}

CANONICAL_AUG_ORDER = ("original", "A01_lowlight", "A02_warm", "A03_softnoise")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_dev_path(dataset_root: Path, rel_posix: str, expected_split: str) -> Path:
    if expected_split not in {"train", "val"}:
        raise ValueError(f"forbidden split {expected_split!r}: only train and val are accessible")
    if not isinstance(rel_posix, str) or "\\" in rel_posix:
        raise ValueError(f"invalid relative path {rel_posix!r}")
    rel = PurePosixPath(rel_posix)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError(f"path escapes root: {rel_posix!r}")
    if rel.parts[0] != expected_split:
        raise ValueError(f"path {rel_posix!r} does not belong to {expected_split!r} (Test/other access forbidden)")
    root_resolved = dataset_root.resolve()
    target = (root_resolved / rel_posix).resolve()
    if not target.is_relative_to(root_resolved / expected_split):
        raise ValueError(f"symlink or alias escapes {expected_split}: {rel_posix!r}")
    if target != root_resolved / rel_posix or target.is_symlink():
        raise ValueError(f"symlink not permitted in development inputs: {rel_posix!r}")
    return target


def r2_initial_offset(seed: int, parent_id: str, num_variants: int = 4) -> int:
    """Deterministic cross-platform initial offset using SHA-256 (never Python hash())."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative int")
    if not isinstance(parent_id, str) or not parent_id.strip():
        raise ValueError("parent_id must be a nonempty string")
    if num_variants <= 1:
        return 0
    payload = f"R2_PARENT_EQUAL_AUG_CE|seed={seed}|parent_id={parent_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % num_variants


def r2_select_variant(parent_entry: dict, seed: int, epoch: int) -> dict:
    """Select the exact single variant exposed for `parent_entry` at 1-based `epoch`."""
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a 1-based positive integer")
    variants = parent_entry["variants"]
    if len(variants) == 1:
        return {
            "parent_index": parent_entry["parent_index"],
            "parent_id": parent_entry["parent_id"],
            "initial_offset": 0,
            "selected_slot": 0,
            **variants[0],
        }
    if len(variants) != 4:
        raise ValueError(f"expected 1 or 4 variants for {parent_entry['parent_id']}, got {len(variants)}")
    offset = r2_initial_offset(seed, parent_entry["parent_id"], 4)
    slot = (offset + (epoch - 1)) % 4
    return {
        "parent_index": parent_entry["parent_index"],
        "parent_id": parent_entry["parent_id"],
        "initial_offset": offset,
        "selected_slot": slot,
        **variants[slot],
    }


def _derive_mechanism(anomaly_category: str) -> str:
    if anomaly_category == "none":
        return "PASS_normal"
    if anomaly_category == "pure_missing":
        return "pure_missing"
    if anomaly_category == "empty_scene":
        return "empty_scene"
    if anomaly_category.startswith("non_catalogue_only_"):
        return "non_catalogue_only"
    if anomaly_category == "non_catalogue_plus_missing":
        return "non_catalogue_plus_missing"
    return "other_anomaly"


def _row_metadata(row: dict, abs_img: Path) -> dict:
    c1 = int(row["count_1_M4x25"])
    c2 = int(row["count_2_M5x10"])
    c3 = int(row["count_3_M4x10"])
    c4 = int(row["count_4_M4_hex_nut"])
    c5 = int(row["count_5_flat_washer"])
    c0 = int(row["count_0_non_catalogue"])
    missing_units = sum(max(0, 2 - c) for c in (c1, c2, c3, c4, c5))
    return {
        "sample_id": row["sample_id"],
        "image_filename": row["image_filename"],
        "split": row["split"],
        "partition_status": row["partition_status"],
        "label": row["label"],
        "data_tier": row["data_tier"],
        "source_batch": row["source_batch"],
        "source_subcondition": row["source_subcondition"],
        "v2_image_relpath": row["v2_image_relpath"],
        "image_path": str(abs_img),
        "scene_group_id": row["scene_group_id"],
        "parent_image_id": row["parent_image_id"] or row["image_filename"],
        "augmentation_type": row["augmentation_type"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "non_catalogue_count": c0,
        "missing_target_units": missing_units,
        "has_missing": missing_units > 0,
        "has_non_catalogue": c0 > 0,
        "anomaly_category": row["anomaly_category"],
        "anomaly_mechanism": _derive_mechanism(row["anomaly_category"]),
        "image_sha256": row["image_sha256"],
        "json_sha256": row["json_sha256"],
    }


def prepare_dataset_v2(
    dataset_root: str | Path = DEFAULT_DATASET_V2_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(dataset_root).resolve()
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory already exists and is non-empty: {out}")

    manifest_csv = root / "dataset_manifest.csv"
    stats_json = root / "dataset_statistics.json"
    aug_csv = root / "augmentation_mapping.csv"
    sha_txt = root / "SHA256SUMS.txt"
    for p in (manifest_csv, stats_json, aug_csv, sha_txt):
        if not p.is_file():
            raise FileNotFoundError(f"missing Dataset v2 metadata file: {p}")

    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    with manifest_csv.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with aug_csv.open(encoding="utf-8", newline="") as fh:
        aug_rows = list(csv.DictReader(fh))
    if len(rows) != 609:
        raise ValueError(f"expected 609 rows in dataset_manifest.csv, got {len(rows)}")

    # 1. Verify exact 438/92/79 quotas from metadata ONLY (never touch test files!)
    quota_counts = collections.Counter((r["split"], r["label"], r["data_tier"]) for r in rows)
    for key, exp_n in APPROVED_QUOTAS.items():
        if quota_counts.get(key, 0) != exp_n:
            raise ValueError(f"Dataset v2 quota mismatch on {key}: expected {exp_n}, got {quota_counts.get(key, 0)}")

    if (
        stats["splits"]["train"]["total_images"] != 438
        or stats["splits"]["val"]["total_images"] != 92
        or stats["splits"]["test"]["total_images"] != 79
        or stats["splits"]["test"]["original_PASS"] != 15
        or stats["splits"]["test"]["original_FAIL"] != 16
    ):
        raise ValueError("dataset_statistics.json disagrees with approved 438/92/79 (Test 15P/16F) freeze")

    # Parse SHA256SUMS.txt lookup for cross-checking Train/Val entries
    sha_sums = {}
    for line in sha_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            digest, rel = line.split("  ", 1)
            sha_sums[rel] = digest

    # 2. Validate ONLY train (438) and val (92) files on disk; NEVER open test files!
    dev_records: list[dict] = []
    test_sha_blacklist = {r["image_sha256"] for r in rows if r["split"] == "test"}
    test_groups_blacklist = {r["scene_group_id"] for r in rows if r["split"] == "test"}

    for r in rows:
        sp = r["split"]
        if sp == "test":
            if not r["v2_image_relpath"].startswith("test/"):
                raise ValueError(f"unexpected test path in manifest: {r['v2_image_relpath']}")
            continue
        if sp not in {"train", "val"}:
            raise ValueError(f"unexpected split: {sp}")
        img_path = safe_dev_path(root, r["v2_image_relpath"], sp)
        actual_sha = sha256_file(img_path)
        if actual_sha != r["image_sha256"] or sha_sums.get(r["v2_image_relpath"]) != actual_sha:
            raise ValueError(f"SHA256 mismatch on {r['v2_image_relpath']}")
        if actual_sha in test_sha_blacklist:
            raise ValueError(f"Train/Val image overlaps Test SHA256 metadata: {r['v2_image_relpath']}")
        if r["scene_group_id"] in test_groups_blacklist:
            raise ValueError(f"Train/Val scene_group_id overlaps Test metadata: {r['scene_group_id']}")
        dev_records.append(_row_metadata(r, img_path))

    train_all = [r for r in dev_records if r["split"] == "train"]
    val_all = [r for r in dev_records if r["split"] == "val"]
    if len(train_all) != 438 or len(val_all) != 92:
        raise ValueError(f"unexpected dev counts: train={len(train_all)}, val={len(val_all)}")

    # Verify Train/Val disjointness on image_sha256 and scene_group_id
    if {r["image_sha256"] for r in train_all} & {r["image_sha256"] for r in val_all}:
        raise ValueError("Train/Val image_sha256 overlap")
    if {r["scene_group_id"] for r in train_all} & {r["scene_group_id"] for r in val_all}:
        raise ValueError("Train/Val scene_group_id overlap")

    #Deterministic ordering by sample_id (which is DSV2_0001.. in canonical split/label/tier/filename order)
    train_raw_294 = sorted(
        (r for r in train_all if r["data_tier"] == "original"),
        key=lambda r: (r["image_filename"], r["sample_id"]),
    )
    train_aug_144 = sorted(
        (r for r in train_all if r["data_tier"] == "augmented"),
        key=lambda r: (r["image_filename"], r["sample_id"]),
    )
    train_full_438 = sorted(train_all, key=lambda r: (r["data_tier"] != "original", r["image_filename"], r["sample_id"]))
    val_raw_74 = sorted(
        (r for r in val_all if r["data_tier"] == "original"),
        key=lambda r: (r["image_filename"], r["sample_id"]),
    )
    val_aug_18 = sorted(
        (r for r in val_all if r["data_tier"] == "augmented"),
        key=lambda r: (r["image_filename"], r["sample_id"]),
    )

    if len(train_raw_294) != 294 or len(train_aug_144) != 144 or len(train_full_438) != 438:
        raise ValueError("unexpected Train tier counts")
    if len(val_raw_74) != 74 or len(val_aug_18) != 18:
        raise ValueError("unexpected Val tier counts")

    # Build R2 parent-equal table in the EXACT SAME order as train_raw_294
    aug_by_parent: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for ar in train_aug_144:
        aug_by_parent[ar["parent_image_id"]][ar["augmentation_type"]] = ar

    r2_parents: list[dict] = []
    augmented_parent_count = 0
    for idx, parent in enumerate(train_raw_294):
        pid = parent["image_filename"]
        child_map = aug_by_parent.get(pid, {})
        base_variant = {
            "variant_slot": 0,
            "variant_name": "original",
            "sample_id": parent["sample_id"],
            "image_filename": parent["image_filename"],
            "v2_image_relpath": parent["v2_image_relpath"],
            "image_sha256": parent["image_sha256"],
        }
        if not child_map:
            variants = [base_variant]
        else:
            if set(child_map.keys()) != {"A01_lowlight", "A02_warm", "A03_softnoise"}:
                raise ValueError(f"incomplete augmented variants for {pid}: {list(child_map.keys())}")
            variants = [base_variant]
            for slot_idx, aug_key in enumerate(CANONICAL_AUG_ORDER[1:], start=1):
                c = child_map[aug_key]
                if c["label"] != parent["label"] or c["scene_group_id"] != parent["scene_group_id"]:
                    raise ValueError(f"parent/child label or group mismatch on {pid} -> {c['image_filename']}")
                variants.append({
                    "variant_slot": slot_idx,
                    "variant_name": aug_key,
                    "sample_id": c["sample_id"],
                    "image_filename": c["image_filename"],
                    "v2_image_relpath": c["v2_image_relpath"],
                    "image_sha256": c["image_sha256"],
                })
            augmented_parent_count += 1
        r2_parents.append({
            "parent_index": idx,
            "parent_id": pid,
            "sample_id": parent["sample_id"],
            "label": parent["label"],
            "scene_group_id": parent["scene_group_id"],
            "source_batch": parent["source_batch"],
            "anomaly_category": parent["anomaly_category"],
            "anomaly_mechanism": parent["anomaly_mechanism"],
            "non_catalogue_count": parent["non_catalogue_count"],
            "missing_target_units": parent["missing_target_units"],
            "has_augmented_variants": len(variants) == 4,
            "variants": variants,
        })

    if augmented_parent_count != 48:
        raise ValueError(f"expected 48 augmented parents in Train, got {augmented_parent_count}")

    # Verify Val augmented 18 parents all exist in val_raw_74
    val_parent_names = {r["image_filename"] for r in val_raw_74}
    for va in val_aug_18:
        if va["parent_image_id"] not in val_parent_names:
            raise ValueError(f"Val augmented image {va['image_filename']} missing parent in val_raw_74")

    out.mkdir(parents=True, exist_ok=False)

    csv_fields = [
        "sample_id", "image_filename", "v2_image_relpath", "label", "scene_id",
        "split", "data_kind", "data_tier", "source_batch", "source_subcondition",
        "parent_image_id", "augmentation_type", "non_catalogue_count",
        "missing_target_units", "anomaly_category", "anomaly_mechanism", "image_sha256",
    ]

    def write_manifest_csv(filename: str, subset: list[dict]) -> Path:
        p = out / filename
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields)
            writer.writeheader()
            for r in subset:
                writer.writerow({
                    "sample_id": r["sample_id"],
                    "image_filename": r["image_filename"],
                    "v2_image_relpath": r["v2_image_relpath"],
                    "label": r["label"],
                    "scene_id": r["scene_group_id"],
                    "split": r["split"],
                    "data_kind": "REAL",
                    "data_tier": r["data_tier"],
                    "source_batch": r["source_batch"],
                    "source_subcondition": r["source_subcondition"],
                    "parent_image_id": r["parent_image_id"],
                    "augmentation_type": r["augmentation_type"],
                    "non_catalogue_count": r["non_catalogue_count"],
                    "missing_target_units": r["missing_target_units"],
                    "anomaly_category": r["anomaly_category"],
                    "anomaly_mechanism": r["anomaly_mechanism"],
                    "image_sha256": r["image_sha256"],
                })
        return p

    p_train_raw = write_manifest_csv("train_raw_294.csv", train_raw_294)
    p_train_full = write_manifest_csv("train_full_438.csv", train_full_438)
    p_val_raw = write_manifest_csv("val_raw_74.csv", val_raw_74)
    p_val_aug = write_manifest_csv("val_aug_18.csv", val_aug_18)

    p_r2_parents = out / "train_r2_parents_294.json"
    p_r2_parents.write_text(
        json.dumps(
            {
                "schema_version": "member06.r2_parents.v1",
                "aligned_with": "train_raw_294.csv",
                "total_parents": len(r2_parents),
                "parents_without_augmentation": len(r2_parents) - augmented_parent_count,
                "parents_with_4_variants": augmented_parent_count,
                "variant_slot_order": list(CANONICAL_AUG_ORDER),
                "offset_hash_rule": "int.from_bytes(sha256('R2_PARENT_EQUAL_AUG_CE|seed={seed}|parent_id={parent_id}')[:8], 'big') % 4",
                "parents": r2_parents,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    provenance = {
        "schema_version": PREPARED_SCHEMA,
        "dataset_v2_root": str(root),
        "dataset_v2_metadata_sha256": {
            "dataset_manifest.csv": sha256_file(manifest_csv),
            "dataset_statistics.json": sha256_file(stats_json),
            "augmentation_mapping.csv": sha256_file(aug_csv),
            "SHA256SUMS.txt": sha256_file(sha_txt),
        },
        "verified_quotas_438_92_79": {
            "train_original_PASS": 149,
            "train_original_FAIL": 145,
            "train_original_total": 294,
            "train_augmented_PASS": 6,
            "train_augmented_FAIL": 138,
            "train_augmented_total": 144,
            "train_total": 438,
            "val_original_PASS": 37,
            "val_original_FAIL": 37,
            "val_original_total": 74,
            "val_augmented_FAIL": 18,
            "val_total": 92,
            "test_metadata_only_original_PASS": 15,
            "test_metadata_only_original_FAIL": 16,
            "test_metadata_only_augmented_FAIL": 48,
            "test_metadata_only_total": 79,
        },
        "prepared_files": {
            "train_raw_294.csv": sha256_file(p_train_raw),
            "train_full_438.csv": sha256_file(p_train_full),
            "train_r2_parents_294.json": sha256_file(p_r2_parents),
            "val_raw_74.csv": sha256_file(p_val_raw),
            "val_aug_18.csv": sha256_file(p_val_aug),
        },
        "test_content_read": False,
        "legacy_dataset_v1_mixed": False,
        "supported_counts": [2, 2, 2, 2, 2],
        "notes": [
            "Primary validation uses ONLY val_raw_74.csv (37 PASS + 37 FAIL real images) with unweighted CE.",
            "val_aug_18.csv (18 FAIL augmented derivatives) is strictly restricted to post-training robustness diagnostics.",
            "Test partition (31 real + 48 augmented = 79) is PROVISIONAL_HOLDOUT and completely untouched.",
        ],
    }
    prov_path = out / "provenance.json"
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return provenance


def verify_prepared_dataset_v2(prepared_dir: str | Path = DEFAULT_OUTPUT_DIR) -> tuple[dict, dict[str, list[dict]], dict]:
    """Verify prepared manifests, Dataset v2 source hashes, and zero Test access."""
    pdir = Path(prepared_dir).resolve()
    prov_path = pdir / "provenance.json"
    if not prov_path.is_file():
        raise FileNotFoundError(f"missing Dataset v2 prepared provenance: {prov_path}")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    if prov.get("schema_version") != PREPARED_SCHEMA or prov.get("test_content_read") is not False:
        raise ValueError("invalid or unsafe Dataset v2 provenance")

    root = Path(prov["dataset_v2_root"]).resolve()
    for meta_name, expected_sha in prov["dataset_v2_metadata_sha256"].items():
        if sha256_file(root / meta_name) != expected_sha:
            raise ValueError(f"Dataset v2 metadata changed after preparation: {meta_name}")

    for fname, expected_sha in prov["prepared_files"].items():
        if sha256_file(pdir / fname) != expected_sha:
            raise ValueError(f"prepared file modified: {fname}")

    manifests: dict[str, list[dict]] = {}
    for key, fname, exp_split, exp_len in (
        ("train_raw_294", "train_raw_294.csv", "train", 294),
        ("train_full_438", "train_full_438.csv", "train", 438),
        ("val_raw_74", "val_raw_74.csv", "val", 74),
        ("val_aug_18", "val_aug_18.csv", "val", 18),
    ):
        with (pdir / fname).open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != exp_len:
            raise ValueError(f"{fname} length mismatch: expected {exp_len}, got {len(rows)}")
        for r in rows:
            if r["split"] != exp_split:
                raise ValueError(f"unexpected split in {fname}: {r['split']}")
            abs_p = safe_dev_path(root, r["v2_image_relpath"], exp_split)
            if sha256_file(abs_p) != r["image_sha256"]:
                raise ValueError(f"image file hash changed: {r['v2_image_relpath']}")
            r["path"] = abs_p
        manifests[key] = rows

    r2_doc = json.loads((pdir / "train_r2_parents_294.json").read_text(encoding="utf-8"))
    if len(r2_doc["parents"]) != 294:
        raise ValueError("train_r2_parents_294.json must contain 294 parents")
    if [p["parent_id"] for p in r2_doc["parents"]] != [r["image_filename"] for r in manifests["train_raw_294"]]:
        raise ValueError("R1/R2 parent alignment mismatch")
    for p in r2_doc["parents"]:
        for v in p["variants"]:
            vp = safe_dev_path(root, v["v2_image_relpath"], "train")
            if sha256_file(vp) != v["image_sha256"]:
                raise ValueError(f"R2 variant hash mismatch: {v['v2_image_relpath']}")
            v["path"] = vp

    return prov, manifests, r2_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_prep = sub.add_parser("prepare", help="prepare verified Train/Val manifests for Dataset v2 Phase 1")
    p_prep.add_argument("--dataset-root", default=DEFAULT_DATASET_V2_ROOT)
    p_prep.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_check = sub.add_parser("verify", help="verify prepared manifests without touching Test")
    p_check.add_argument("--prepared-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.cmd == "prepare":
        res = prepare_dataset_v2(args.dataset_root, args.output_dir)
        print(json.dumps({"status": "PREPARED_OK", "quotas": res["verified_quotas_438_92_79"]}, ensure_ascii=False, indent=2))
    else:
        prov, manifests, r2_doc = verify_prepared_dataset_v2(args.prepared_dir)
        print(json.dumps({
            "status": "VERIFIED_OK",
            "counts": {k: len(v) for k, v in manifests.items()},
            "r2_augmented_parents": r2_doc["parents_with_4_variants"],
            "test_content_read": prov["test_content_read"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
