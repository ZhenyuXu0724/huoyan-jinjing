"""Pre/post-freeze dataset audit and machine-readable v4 experiment protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps

from .common import (
    AUDIT, CELL, CONFIG, DATA, DATA_CODE, DATASET_BYTES, DATASET_TREE, EPOCHS,
    EXPECTED_COUNTS, FP_BUDGET, MODEL_CODE, OUT, PROTOCOL, PROTOCOL_CODE, SEEDS, SIZE,
    SOURCE_LOCK, SOURCE_COMMIT, DELIVERY_NAME, METHOD_SUFFIX, PREFIX, ROOT, code_hashes,
    dependencies, ensure, eol_variant, git_blob_sha1, now, read, read_csv, record_stage,
    sha256, sha256_bytes, update_state, write,
)
from .groups import GROUPS, augmentation_count, parameter_count, view_count, views


def check(result, group, name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    result["checks"].append({"group": group, "id": name, "status": status, "detail": detail})
    if not condition:
        result["failures"].append(f"{group}/{name}: {detail}")
    return condition


def digest(payload):
    return sha256_bytes(payload)


def decoded_pixel_hash(image):
    return hashlib.sha256(f"{image.size}:".encode() + image.tobytes()).hexdigest()


def run_verify_dataset(scope):
    command = [sys.executable, "verify_dataset.py"] + (["--all"] if scope == "full" else [])
    started = perf_counter()
    completed = subprocess.run(command, cwd=DATA, capture_output=True, text=True,
                               timeout=7200)
    seconds = perf_counter() - started
    body = ("COMMAND: " + " ".join(command) + "\n"
            f"EXIT: {completed.returncode}\n" f"SECONDS: {seconds:.2f}\n"
            "STDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr + "\n")
    ensure(AUDIT)
    (AUDIT / f"verify_dataset_{scope}.txt").write_text(body, encoding="utf-8")
    return {"command": " ".join(command), "exit": completed.returncode,
            "seconds": seconds, "stdout_sha256": digest(completed.stdout.encode()),
            "stdout_tail": completed.stdout.strip().splitlines()[-1:]}


def classify_inventory(entries):
    by_kind = Counter()
    by_split = Counter()
    for entry in entries:
        by_kind[entry["kind"]] += 1
        path = entry["path"]
        if path.startswith("images/"):
            parts = path.split("/")
            by_split[parts[1] if len(parts) > 2 else "-"] += 1
        elif path.startswith("labels/"):
            parts = path.split("/")
            by_split[parts[1] if len(parts) > 2 else "-"] += 1
    return dict(by_kind), dict(by_split)


def inventory_check(result, scope):
    lock = read(SOURCE_LOCK)
    file_list = read(AUDIT / f"file_list_{scope}.json")
    entries = {entry["path"]: entry for entry in file_list["entries"]}
    on_disk = {}
    for path in DATA.rglob("*"):
        if path.is_file():
            relative = path.relative_to(DATA).as_posix()
            on_disk[relative] = path
    missing = sorted(set(entries) - set(on_disk))
    extra = sorted(set(on_disk) - set(entries))
    check(result, "inventory", "no_missing_files", not missing, str(missing[:5]))
    check(result, "inventory", "no_extra_files", not extra, str(extra[:5]))
    physical_bytes = 0
    git_mismatches, media_mismatches, metadata_mismatches = [], [], []
    for name, entry in entries.items():
        path = DATA / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        physical_bytes += len(raw)
        candidate = eol_variant(raw, entry["eol"])
        if git_blob_sha1(candidate) != entry["sha"] or len(candidate) != entry["size"]:
            git_mismatches.append(name)
        if entry["kind"] == "metadata":
            normalised = raw.replace(b"\r\n", b"\n")
            if sha256_bytes(normalised) != lock["metadata_sha256"][name]:
                metadata_mismatches.append(name)
    check(result, "inventory", "git_blob_and_size", not git_mismatches,
          str(git_mismatches[:5]))
    check(result, "inventory", "metadata_sha256", not metadata_mismatches,
          str(metadata_mismatches[:5]))
    by_kind, by_split = classify_inventory(entries.values())
    result["summary"]["files"] = len(entries)
    result["summary"]["physical_bytes"] = physical_bytes
    result["summary"]["tree_bytes"] = file_list["tree_bytes"]
    result["summary"]["by_kind"] = by_kind
    result["summary"]["by_split_files"] = by_split
    return entries


def load_manifest_rows():
    return read_csv(DATA / "split_manifest.csv")


def label_map():
    classes = read(DATA / "classes.json")["classes"]
    return {row["labelme_label"]: int(row["id"]) for row in classes}, set(
        int(row["id"]) for row in classes)


def parse_labelme(row, ids):
    path = DATA / row["dest_json"]
    annotation = json.loads(path.read_text(encoding="utf-8-sig"))
    width, height = int(annotation["imageWidth"]), int(annotation["imageHeight"])
    counts = Counter()
    boxes = []
    issues = []
    for shape in annotation["shapes"]:
        if shape.get("shape_type") != "rectangle":
            issues.append("non-rectangle")
            continue
        label = shape.get("label")
        if label not in ids:
            issues.append(f"unknown-label:{label}")
            continue
        points = shape.get("points") or []
        if len(points) != 2:
            issues.append("point-count")
            continue
        (x1, y1), (x2, y2) = points
        values = [x1, y1, x2, y2]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            issues.append("non-finite")
            continue
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            issues.append("out-of-bounds")
            continue
        counts[label] += 1
        boxes.append((label, (left + right) / 2 / width,
                      (top + bottom) / 2 / height, (right - left) / width,
                      (bottom - top) / height))
    return annotation, width, height, counts, boxes, issues


def check_yolo(row, boxes, ids):
    path = DATA / row["yolo_label"]
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != len(boxes):
        return False, f"line count {len(lines)} != {len(boxes)}"
    for number, (line, box) in enumerate(zip(lines, boxes, strict=True), 1):
        fields = line.split()
        if len(fields) != 5:
            return False, f"line {number} has {len(fields)} fields"
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            return False, f"line {number} is not numeric"
        if class_id != ids[box[0]] or class_id not in range(6):
            return False, f"line {number} class mismatch"
        if any(abs(value - expected) > 1e-9 for value, expected in
               zip(values, box[1:], strict=True)):
            return False, f"line {number} box mismatch"
    return True, ""


def check_active_images(result, scope, rows, ids):
    if scope == "development":
        active = [row for row in rows if row["usage"] == "primary"
                  and row["split"] in ("train", "val")]
    else:
        active = [row for row in rows if row["usage"] in ("primary", "optional_train_aug")]
    split_stats = defaultdict(lambda: {"images": 0, "PASS": 0, "FAIL": 0, "boxes": 0,
                                       "original_boxes": 0,
                                       "scenes": set(), "source_scenes": set(),
                                       "anomalies": Counter()})
    pixel_sets = defaultdict(set)
    dimension_issues, decode_issues, box_issues, yolo_issues = [], [], [], []
    empty_labels = 0
    exif_orientations = Counter()
    for index, row in enumerate(active, 1):
        split = row["split"]
        stats = split_stats[split]
        stats["images"] += 1
        stats["PASS" if row["label"] == "PASS" else "FAIL"] += 1
        stats["scenes"].add(row["scene_group"])
        stats["source_scenes"].add(row["source_scene_group"])
        stats["anomalies"][row["anomaly"]] += 1
        image_path = DATA / row["dest_image"]
        try:
            with Image.open(image_path) as source:
                orientation = int(source.getexif().get(274, 1) or 1)
                exif_orientations[orientation] += 1
                raw_size = source.size
                image = ImageOps.exif_transpose(source).convert("L")
                transposed_size = image.size
                pixel_hash = decoded_pixel_hash(image)
        except (OSError, ValueError) as error:
            decode_issues.append(f"{image_path.name}: {error}")
            continue
        pixel_sets[split].add(pixel_hash)
        annotation, width, height, counts, boxes, issues = parse_labelme(row, ids)
        if issues:
            box_issues.append(f"{row['filename']}: {','.join(sorted(set(issues)))}")
        if not (width, height) == raw_size and not (width, height) == transposed_size:
            dimension_issues.append(
                f"{row['filename']}: labelme={width}x{height} raw={raw_size} "
                f"transposed={transposed_size}")
        raw_image = image_path.read_bytes()
        if (len(raw_image) != int(row["image_bytes"])
                or digest(raw_image) != row["image_sha256"]
                or git_blob_sha1(raw_image) != row["image_blob"]):
            box_issues.append(f"{row['filename']}: image manifest mismatch")
        raw_json = (DATA / row["dest_json"]).read_bytes()
        if (len(raw_json) != int(row["json_bytes"])
                or digest(raw_json) != row["json_sha256"]
                or git_blob_sha1(raw_json) != row["json_blob"]):
            box_issues.append(f"{row['filename']}: json manifest mismatch")
        by_label = Counter({0: counts.get("non_catalogue", 0)})
        for label, count in counts.items():
            class_id = ids.get(label)
            if class_id is None or class_id == 0:
                if label != "non_catalogue":
                    box_issues.append(f"{row['filename']}: unknown label {label}")
                continue
            by_label[class_id] += count
        target = [int(value) for value in row["target_counts"].split("|")]
        distractor_expected = sum(int(row[key]) for key in "NBSWH")
        if by_label.get(0, 0) != distractor_expected:
            box_issues.append(f"{row['filename']}: non_catalogue count mismatch")
        if [by_label.get(class_id, 0) for class_id in range(1, 6)] != target:
            box_issues.append(f"{row['filename']}: target count mismatch")
        verdict = "PASS" if by_label.get(0, 0) == 0 and target == [2] * 5 else "FAIL"
        if verdict != row["label"]:
            box_issues.append(f"{row['filename']}: recomputed verdict {verdict}")
        folder = Path(row["dest_image"]).parts[2]
        if folder != row["label"]:
            box_issues.append(f"{row['filename']}: folder label {folder}")
        yolo_ok, yolo_detail = check_yolo(row, boxes, ids)
        if not yolo_ok:
            yolo_issues.append(f"{row['filename']}: {yolo_detail}")
        text_bytes = (DATA / row["yolo_label"]).stat().st_size
        if not boxes:
            empty_labels += 1
            if row["label"] != "FAIL" or text_bytes != 0:
                box_issues.append(f"{row['filename']}: illegal empty label")
        elif text_bytes == 0:
            box_issues.append(f"{row['filename']}: missing YOLO content")
        stats["boxes"] += len(boxes)
        if row["usage"] == "primary":
            stats["original_boxes"] += len(boxes)
        if index % 40 == 0:
            print(f"Audited {index}/{len(active)} images ({scope})", flush=True)
    check(result, "images", "decode_all", not decode_issues, str(decode_issues[:5]))
    check(result, "images", "dimensions_match_labelme", not dimension_issues,
          str(dimension_issues[:5]))
    check(result, "images", "manifest_image_json_bytes", not box_issues,
          str(box_issues[:5]))
    check(result, "annotations", "labelme_and_yolo", not yolo_issues, str(yolo_issues[:5]))
    check(result, "annotations", "legal_empty_labels", not [
        item for item in box_issues if "empty label" in item], str(empty_labels))
    result["summary"]["exif_orientations"] = dict(exif_orientations)
    result["summary"]["empty_label_images"] = empty_labels
    return split_stats, pixel_sets


def aggregate_originals(rows, split_stats):
    originals = [row for row in rows if row["tier"] == "original"]
    by_split = defaultdict(lambda: {"images": 0, "PASS": 0, "FAIL": 0, "boxes": None,
                                    "scenes": set(), "source_scenes": set(),
                                    "anomalies": Counter()})
    for row in originals:
        split = row["split"]
        entry = by_split[split]
        entry["images"] += 1
        entry["PASS" if row["label"] == "PASS" else "FAIL"] += 1
        entry["scenes"].add(row["scene_group"])
        entry["source_scenes"].add(row["source_scene_group"])
        entry["anomalies"][row["anomaly"]] += 1
    for split, stats in split_stats.items():
        by_split[split]["boxes"] = stats["original_boxes"]
    return {
        split: {
            "images": stats["images"],
            "labels": {"PASS": stats["PASS"], "FAIL": stats["FAIL"]},
            "boxes": stats.get("boxes"),
            "scenes": len(stats["scenes"]),
            "source_scenes": len(stats["source_scenes"]),
            "anomalies": dict(sorted(stats["anomalies"].items())),
        }
        for split, stats in by_split.items()
    }


def cross_split_checks(result, rows, pixel_sets, scope):
    formal = [row for row in rows if row["split"] != "unused"]
    scene_sets, source_sets, image_sets = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in formal:
        scene_sets[row["split"]].add(row["scene_group"])
        source_sets[row["split"]].add(row["source_scene_group"])
        image_sets[row["split"]].add(row["image_sha256"])
    def overlap(left, right, key):
        return sorted(set(left) & set(right))
    split_names = ["train", "val", "test"]
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1:]:
            check(result, "isolation", f"scene_group_{left}_{right}",
                  not overlap(scene_sets[left], scene_sets[right], ""), "")
            check(result, "isolation", f"source_scene_group_{left}_{right}",
                  not overlap(source_sets[left], source_sets[right], ""), "")
            check(result, "isolation", f"image_sha256_{left}_{right}",
                  not overlap(image_sets[left], image_sets[right], ""), "")
    decoded = [split for split in split_names if pixel_sets.get(split)]
    for left_index, left in enumerate(decoded):
        for right in decoded[left_index + 1:]:
            check(result, "isolation", f"pixel_sha256_{left}_{right}",
                  not overlap(pixel_sets[left], pixel_sets[right], ""), "")


def augmentation_checks(result, rows, scope):
    originals = {row["filename"]: row for row in rows if row["tier"] == "original"}
    augmentations = [row for row in rows if row["tier"] == "augmented"
                     and row["usage"] == "optional_train_aug"]
    parents = Counter(row["parent"] for row in augmentations)
    problems = []
    for row in augmentations:
        parent = originals.get(row["parent"])
        if parent is None:
            problems.append(f"{row['filename']}: missing parent")
            continue
        for key in ("split", "scene_group", "label", "batch", "N", "B", "S", "W", "H"):
            if row[key] != parent[key]:
                problems.append(f"{row['filename']}: parent field {key}")
        if parent["label"] != "FAIL" or parent["split"] != "train":
            problems.append(f"{row['filename']}: parent is not a train FAIL")
        if "train_augmentations" not in row["dest_image"]:
            problems.append(f"{row['filename']}: augmentation path")
    check(result, "augmentation", "parent_fields", not problems, str(problems[:5]))
    check(result, "augmentation", "counts", len(augmentations) == 102
          and len(parents) == 34 and set(parents.values()) == {3},
          f"{len(augmentations)} rows, {len(parents)} parents")
    result["summary"]["augmentations"] = {
        "rows": len(augmentations), "parents": len(parents),
        "per_parent": sorted(set(parents.values()))}


def original_detail_checks(result, rows):
    path = DATA / "原图明细.csv"
    raw = path.read_bytes()
    reader = csv.reader(raw.decode("utf-8-sig").splitlines())
    header = next(reader)
    records = [record for record in reader if record]
    originals = {row["filename"]: row for row in rows if row["tier"] == "original"}
    problems = []
    if len(header) != 9:
        problems.append(f"unexpected header width {len(header)}")
    for record in records:
        if len(record) < 9:
            problems.append(f"short row {record[:1]}")
            continue
        filename, split, label, batch, scene, _verdict, target, distractor, dest = record[:9]
        row = originals.get(filename)
        if row is None:
            problems.append(f"{filename}: not an original")
            continue
        if (split, label, batch, scene) != (row["split"], row["label"],
                                            row["batch"], row["scene_group"]):
            problems.append(f"{filename}: split/label/batch/scene mismatch")
        if target != row["target_counts"]:
            problems.append(f"{filename}: target counts mismatch")
        distractor_sum = sum(int(row[key]) for key in "NBSWH")
        if (distractor_sum == 0) != (distractor == ""):
            problems.append(f"{filename}: distractor description mismatch")
        expected_dest = row["dest_image"] if row["usage"] == "primary" else ""
        if dest != expected_dest:
            problems.append(f"{filename}: relative path mismatch")
    check(result, "metadata", "original_detail_rows", len(records) == 399,
          str(len(records)))
    check(result, "metadata", "original_detail_crosscheck", not problems,
          str(problems[:5]))


def manifest_checks(result, scope):
    manifests = {}
    for name in ("train", "train_pass", "val", "test"):
        path = DATA / f"manifests/hog/{name}.csv"
        rows = read_csv(path)
        manifests[name] = {"rows": len(rows), "sha256": sha256(path)}
    result["summary"]["manifests"] = manifests
    expected = {"train": 249, "train_pass": 166, "val": 34, "test": 30}
    check(result, "metadata", "hog_manifest_counts",
          {name: manifests[name]["rows"] for name in expected} == expected,
          str({name: manifests[name]["rows"] for name in expected}))
    train = read_csv(DATA / "manifests/hog/train.csv")
    pass_only = read_csv(DATA / "manifests/hog/train_pass.csv")
    check(result, "metadata", "train_pass_is_pass_subset",
          pass_only == [row for row in train if row["label"] == "PASS"], "")
    for name in ("train", "val", "test"):
        if scope == "development" and name == "test":
            continue
        rows = read_csv(DATA / f"manifests/hog/{name}.csv")
        for row in rows:
            relative = row["image_path"]
            resolved = (DATA / "manifests/hog" / relative).resolve()
            if not resolved.is_file():
                check(result, "metadata", f"manifest_{name}_paths", False,
                      f"missing {relative}")
                break
        else:
            check(result, "metadata", f"manifest_{name}_paths", True, "")
    return manifests


def reference_checks(result, aggregate, rows):
    reference = read(ROOT / "members/member04_hog/plans/v4_rebuild_20261124/dataset_audit.json")
    for split in ("train", "val", "test", "unused"):
        expected = reference["splits"][split]
        actual = aggregate.get(split)
        if actual is None:
            continue
        checks = [
            ("images", actual["images"], expected["images"]),
            ("labels", actual["labels"], expected["labels"]),
            ("scenes", actual["scenes"], expected["scenes"]),
            ("source_scenes", actual["source_scenes"], expected["source_scenes"]),
            ("anomalies", actual["anomalies"], expected["anomalies"]),
        ]
        if split != "unused" and actual.get("boxes") is not None:
            checks.append(("boxes", actual["boxes"], expected["boxes"]))
        for key, got, want in checks:
            if key == "labels" and isinstance(got, dict):
                got = {name: value for name, value in got.items() if value}
                want = {name: value for name, value in want.items() if value}
            check(result, "reference", f"{split}_{key}", got == want,
                  f"{got} != {want}" if got != want else "")
    batch = Counter((row["batch"], row["split"]) for row in rows
                    if row["tier"] == "original")
    expected_batch = reference["batch_split_counts"]
    check(result, "reference", "batch_split_counts",
          {f"{key[0]}/{key[1]}": value for key, value in batch.items()} == expected_batch,
          "")
    counts = Counter(row["batch"] for row in rows if row["tier"] == "original")
    check(result, "reference", "batch_counts", dict(counts) == reference["batch_counts"], "")


def full_scope_checks(result, rows, scope):
    if scope == "development":
        test_media = list(DATA.glob("images/test/**/*"))
        augment_media = list(DATA.glob("images/train_augmentations/**/*"))
        check(result, "test_gate", "no_test_media_in_development",
              not test_media and not augment_media,
              f"{len(test_media)} test, {len(augment_media)} augment files")
        result["summary"]["test_media_decoded"] = 0
        result["summary"]["test_media_present"] = False
        result["summary"]["test_metadata_read"] = True
        return
    freeze_path = OUT / "freeze.json"
    freeze = read(freeze_path)
    access = read(AUDIT / "test_access.json") if (AUDIT / "test_access.json").exists() else None
    check(result, "test_gate", "freeze_present", True, freeze.get("selected_default", ""))
    check(result, "test_gate", "access_record_present", access is not None, "")
    if access is not None:
        check(result, "test_gate", "access_matches_freeze",
              access.get("freeze_sha256") == sha256(freeze_path), "")
        result["summary"]["test_access"] = access
    triples = 0
    for folder in ("images/train", "images/val", "images/test",
                   "images/train_augmentations"):
        for image in (DATA / folder).rglob("*"):
            if image.is_file() and image.suffix.lower() in (".jpg", ".jpeg", ".png"):
                triples += 1
                if not image.with_suffix(".json").is_file():
                    check(result, "full", f"triple_json_{image.name}", False, "")
                relative = image.relative_to(DATA).as_posix()
                label = relative.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
                if not (DATA / label).is_file():
                    check(result, "full", f"triple_txt_{image.name}", False, "")
    check(result, "full", "physical_triples", triples == 415, str(triples))
    result["summary"]["physical_triples"] = triples
    result["summary"]["test_media_decoded"] = sum(
        1 for row in rows if row["split"] == "test"
        and row["usage"] in ("primary", "optional_train_aug"))
    result["summary"]["test_media_present"] = True


def build_protocol(result, scope):
    lock = read(SOURCE_LOCK)
    manifests = result["summary"]["manifests"]
    tasks = []
    for method, layers, alpha in (("ae", (64, 16, 64), 0.0001),
                                  ("supervised", (64, 16), 0.0001)):
        for group in GROUPS:
            for seed in SEEDS:
                tasks.append({
                    "id": f"{method}/{group}/seed{seed}", "method": method,
                    "group": group, "seed": seed, "epochs": EPOCHS,
                    "layers": list(layers), "alpha": alpha,
                    "geometry": GROUPS[group]["geometry"],
                    "gamma": GROUPS[group]["gamma"], "clahe": GROUPS[group]["clahe"],
                    "views": view_count(group),
                })
    refinement = []
    for config, layers, alpha in (("R0", (64, 16), 0.0001),
                                  ("R1", (32, 8), 0.0001),
                                  ("R2", (64, 16), 0.01)):
        for seed in SEEDS:
            refinement.append({
                "id": f"refinement/{config}/seed{seed}", "config": config,
                "base": "selected supervised group", "seed": seed, "epochs": EPOCHS,
                "layers": list(layers), "alpha": alpha,
            })
    protocol = {
        "schema_version": 1,
        "name": DELIVERY_NAME,
        "created_at": now(),
        "source": {"repository": lock["repository"], "commit": lock["source_commit"],
                   "dataset_path": lock["dataset_path"],
                   "dataset_tree": lock["dataset_tree"],
                   "tree_listing_sha256": lock["tree_listing"]["sha256"],
                   "source_lock_sha256": sha256(SOURCE_LOCK)},
        "audit": {"path": "outputs/member04_hog/final_v4/audit/"
                          f"dataset_audit_{scope}.json",
                  "sha256": sha256(AUDIT / f"dataset_audit_{scope}.json")},
        "data": {
            "manifests_sha256": {f"manifests/hog/{name}.csv": value["sha256"]
                                 for name, value in manifests.items()},
            "counts": {split: values for split, values in result["summary"]["by_split"].items()},
            "expected_counts": EXPECTED_COUNTS,
        },
        "preprocessing": {"size": SIZE, "cell": CELL, "orientations": 9,
                          "pixels_per_cell": [CELL, CELL], "cells_per_block": [2, 2],
                          "block_norm": "L2-Hys", "dimension": 8100,
                          "clahe_clip_limit": 0.01, "gammas": [0.8, 1.0, 1.2]},
        "training": {"epochs": EPOCHS, "learning_rate": 0.001, "batch_size": 32,
                     "alpha": 0.0001, "seeds": list(SEEDS), "single_thread": True,
                     "shuffle": False, "early_stopping": False,
                     "sample_weights": {"PASS": 0.75, "FAIL": 1.5},
                     "checkpoint": "main runs use epoch 200 only"},
        "groups": {name: {**spec, "views": view_count(name),
                          "augmentations": augmentation_count(name)}
                   for name, spec in GROUPS.items()},
        "tasks": tasks,
        "refinement": {"configs": {"R0": {"layers": [64, 16], "alpha": 0.0001},
                                   "R1": {"layers": [32, 8], "alpha": 0.0001},
                                   "R2": {"layers": [64, 16], "alpha": 0.01}},
                       "tasks": refinement},
        "selection_rules": {
            "threshold": "unique-score cuts; require FP<=4/20; fewest FN, then FP, "
                         "then larger representative threshold; all-PASS and all-FAIL included",
            "validation_budget": {"false_pass_max": FP_BUDGET, "validation_pass": 20},
            "group": "mean FN, mean FP, fewer augmentation types, fewer views, table order; "
                     "single-factor group wins exact duplicates",
            "checkpoint": "refinement: fewest FN, then FP, then earliest epoch",
            "refinement_group": "mean FN, mean FP, fewer parameters, smaller alpha, table order",
            "default_model": "mean FN, mean FP, fewer trainable parameters, fewer training "
                             "views, then supervised main > supervised refinement > AE",
            "representative_seed": 42,
            "equivalent_pairs": [["B2", "C_without_geometry"],
                                 ["A2", "C_without_clahe"]],
        },
        "versions": {method: suffix for method, suffix in METHOD_SUFFIX.items()},
        "code_sha256": code_hashes(PROTOCOL_CODE),
        "dependencies": dependencies(),
    }
    write(PROTOCOL, protocol)
    write(OUT / "protocol.json", protocol)
    return protocol


def run(scope):
    ensure(AUDIT)
    result = {"schema_version": 1, "scope": scope, "generated_at": now(),
              "status": "PASS", "source": {}, "data_root": "data/member04_hog/final_training_v4",
              "checks": [], "failures": [], "warnings": [], "limitations": [],
              "summary": {}}
    lock = read(SOURCE_LOCK)
    result["source"] = {"commit": lock["source_commit"], "tree": lock["dataset_tree"],
                        "source_lock_sha256": sha256(SOURCE_LOCK),
                        "tree_cache_sha256": sha256(AUDIT / "source_tree.json")}
    inventory = inventory_check(result, scope)
    verify = run_verify_dataset(scope)
    result["verify_dataset"] = verify
    check(result, "provider", "verify_dataset_exit", verify["exit"] == 0,
          str(verify["stdout_tail"]))
    rows = load_manifest_rows()
    ids, class_ids = label_map()
    check(result, "metadata", "class_ids", class_ids == set(range(6)), str(class_ids))
    split_stats, pixel_sets = check_active_images(result, scope, rows, ids)
    aggregate = aggregate_originals(rows, split_stats)
    result["summary"]["by_split"] = aggregate
    cross_split_checks(result, rows, pixel_sets, scope)
    augmentation_checks(result, rows, scope)
    original_detail_checks(result, rows)
    manifest_checks(result, scope)
    reference_checks(result, aggregate, rows)
    full_scope_checks(result, rows, scope)
    result["eol"] = lock["eol"]
    result["limitations"] = [
        "Scene-group fields are disjoint; physical scene independence is not independently proven.",
        "Test images are decoded only in the full-scope audit after freeze.",
        "Git stores LF for text files checked out as CRLF on Windows; the provider split_manifest "
        "SHA256 values lock the CRLF working bytes, and both are verified here.",
    ]
    result["summary"]["files"] = result["summary"].get("files", 0)
    result["summary"]["test_media_decoded"] = result["summary"].get("test_media_decoded", 0)
    result["status"] = "PASS" if not result["failures"] else "FAIL"
    target = AUDIT / f"dataset_audit_{scope}.json"
    write(target, result)
    if result["failures"]:
        record_stage(f"audit_{scope}", "failed",
                     f"{len(result['failures'])} checks failed")
        raise RuntimeError(f"audit {scope} failed: {result['failures'][:5]}")
    if scope == "development":
        protocol = build_protocol(result, scope)
        record_stage("audit_development", "completed",
                     f"development audit PASS, protocol locked ({len(protocol['tasks'])} tasks)")
        update_state(next_command="python -m members.member04_hog.final_v4.train "
                                  "--method ae --protocol members/member04_hog/configs/final_v4/protocol.json")
    else:
        record_stage("audit_full", "completed", "full audit PASS after freeze")
        update_state(next_command="python -m members.member04_hog.final_v4.evaluate --frozen")
    print(f"Audit {scope}: PASS; files={result['summary']['files']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("development", "full"), default="development")
    args = parser.parse_args()
    try:
        run(args.scope)
    except Exception as error:
        record_stage(f"audit_{args.scope}", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
