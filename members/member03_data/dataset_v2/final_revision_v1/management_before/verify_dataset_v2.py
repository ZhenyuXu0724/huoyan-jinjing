#!/usr/bin/env python3
"""Independent verifier for members/member03_data/dataset_v2.

Checks:
1. Directory structure (train/val/test x PASS/FAIL x original/augmented) exists.
2. Exact file counts per leaf directory match approved Dataset v2 quotas (609 images + 609 JSONs = 1,218 files).
3. Every copied image and JSON matches its original source file byte-for-byte (SHA256 + size + not hardlinked).
4. SHA256SUMS.txt matches all 1,218 files on disk.
5. Parent-child augmentation isolation: 0 augmented images differ in split or label from their parent original.
6. Labelme JSON integrity: every file's PASS/FAIL directory matches the [2,2,2,2,2] + non_catalogue==0 rule.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
from pathlib import Path

DATASET_V2_DIR = Path(__file__).resolve().parent
MEMBER03_ROOT = DATASET_V2_DIR.parent

EXPECTED_QUOTAS = {
    ("train", "PASS", "original"): 151,
    ("train", "FAIL", "original"): 145,
    ("train", "PASS", "augmented"): 6,
    ("train", "FAIL", "augmented"): 138,
    ("val", "PASS", "original"): 37,
    ("val", "FAIL", "original"): 37,
    ("val", "PASS", "augmented"): 0,
    ("val", "FAIL", "augmented"): 18,
    ("test", "PASS", "original"): 13,
    ("test", "FAIL", "original"): 16,
    ("test", "PASS", "augmented"): 0,
    ("test", "FAIL", "augmented"): 48,
}

LABEL_TO_ID = {
    "non_catalogue": 0,
    "M4x25": 1,
    "M5x10": 2,
    "M4x10": 3,
    "M4_hex_nut": 4,
    "flat_washer": 5,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_label_from_json(json_path: Path) -> str:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    counts = collections.Counter(s["label"].strip() for s in raw.get("shapes", []))
    for k in counts:
        if k not in LABEL_TO_ID:
            raise AssertionError(f"unknown label {k} in {json_path}")
    ok = (
        counts.get("non_catalogue", 0) == 0
        and all(counts.get(lbl, 0) == 2 for lbl in ("M4x25", "M5x10", "M4x10", "M4_hex_nut", "flat_washer"))
    )
    return "PASS" if ok else "FAIL"


def main() -> None:
    # Independent enumeration validates bijections, Git snapshot, geometry and metadata.
    # Explicit exceptions remain active even when Python assertions are disabled.
    from independent_acceptance import run
    run()
    # 1. Check leaf directories & counts
    leaf_counts = {}
    total_imgs = 0
    total_jsons = 0
    for (split, label, tier), exp_n in EXPECTED_QUOTAS.items():
        leaf = DATASET_V2_DIR / split / label / tier
        if not leaf.is_dir():
            raise AssertionError(f"missing directory: {leaf}")
        imgs = sorted(p for p in leaf.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        jsons = sorted(p for p in leaf.iterdir() if p.suffix.lower() == ".json")
        others = [p for p in leaf.iterdir() if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".json"}]
        if others:
            raise AssertionError(f"unexpected files in {leaf}: {others}")
        if len(imgs) != exp_n or len(jsons) != exp_n:
            raise AssertionError(f"count mismatch in {leaf}: expected {exp_n} pairs, got {len(imgs)} imgs / {len(jsons)} jsons")
        if [p.stem for p in imgs] != [p.stem for p in jsons]:
            raise AssertionError(f"image/json stem mismatch in {leaf}")
        for jp in jsons:
            derived = derive_label_from_json(jp)
            if derived != label:
                raise AssertionError(f"label rule mismatch for {jp}: placed in {label}, derived {derived}")
        leaf_counts[f"{split}/{label}/{tier}"] = len(imgs)
        total_imgs += len(imgs)
        total_jsons += len(jsons)

    assert total_imgs == 609 and total_jsons == 609, (total_imgs, total_jsons)

    # 2. Verify dataset_manifest.csv against source files & v2 copies (byte-for-byte + inode separation)
    manifest_rows = list(csv.DictReader((DATASET_V2_DIR / "dataset_manifest.csv").open(encoding="utf-8")))
    assert len(manifest_rows) == 609, len(manifest_rows)

    byte_verified = 0
    inodes_disjoint = 0
    for row in manifest_rows:
        src_img = MEMBER03_ROOT / row["original_relpath"]
        src_json = MEMBER03_ROOT / row["original_json_relpath"]
        dst_img = DATASET_V2_DIR / row["v2_image_relpath"]
        dst_json = DATASET_V2_DIR / row["v2_json_relpath"]
        for s, d, expected_sha in (
            (src_img, dst_img, row["image_sha256"]),
            (src_json, dst_json, row["json_sha256"]),
        ):
            assert s.is_file() and d.is_file(), (s, d)
            assert not d.is_symlink(), f"symlink not allowed: {d}"
            assert not os.path.samefile(s, d), f"hardlink not allowed: {s} == {d}"
            inodes_disjoint += 1
            s_sha = sha256_file(s)
            d_sha = sha256_file(d)
            assert s.stat().st_size == d.stat().st_size, f"size mismatch: {s} vs {d}"
            assert s_sha == d_sha == expected_sha, f"sha mismatch: {s} vs {d}"
            byte_verified += 1

    # 3. Verify SHA256SUMS.txt
    sha_lines = [
        line.strip() for line in (DATASET_V2_DIR / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(sha_lines) == 1218, len(sha_lines)
    for line in sha_lines:
        recorded_sha, rel = line.split("  ", 1)
        actual_sha = sha256_file(DATASET_V2_DIR / rel)
        assert recorded_sha == actual_sha, f"SHA256SUMS mismatch on {rel}"

    # 4. Verify parent-child augmentation mapping (210 augmented -> 70 parents, 0 cross-split)
    aug_rows = list(csv.DictReader((DATASET_V2_DIR / "augmentation_mapping.csv").open(encoding="utf-8")))
    assert len(aug_rows) == 210, len(aug_rows)
    orig_split = {r["image_filename"]: (r["split"], r["label"], r["scene_group_id"]) for r in manifest_rows if r["data_tier"] == "original"}
    for ar in aug_rows:
        ps, pl, pg = orig_split[ar["parent_filename"]]
        assert ar["split"] == ps == ar["parent_split"], f"augmentation split leak: {ar}"
        assert ar["label"] == pl == ar["parent_label"], f"augmentation label mismatch: {ar}"
        assert ar["scene_group_id"] == pg, f"augmentation group mismatch: {ar}"
        assert ar["same_split_verified"] == "True" and ar["same_label_verified"] == "True"

    print(json.dumps({
        "verification": "ENGINEERING_CHECKS_PASSED_NOT_TEST_QUALIFIED",
        "total_images": total_imgs,
        "total_jsons": total_jsons,
        "byte_identical_files_verified": byte_verified,
        "physical_copy_non_hardlink_verified": inodes_disjoint,
        "sha256sums_entries_verified": len(sha_lines),
        "augmented_parent_isolation_verified": len(aug_rows),
        "leaf_directory_image_counts": leaf_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
