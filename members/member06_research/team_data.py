"""Member-local adapter for member03.gt.v1. Never open test images or annotations.

Audit does not assign labels. Prepare requires explicit dataset and per-image review
records; those records describe a local handoff format, not a new team protocol.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from .config import normalize_counts
from .image_io import read_image

DEFAULT_ROOT = "members/member03_data/full_dataset_upload/datasets/parts_v1"
DEFAULT_CATALOGUE = "shared/catalogue.json"
MANIFEST_COLUMNS = {"split", "category", "filename", "source", "destination", "sha256"}
GT_SCHEMA = "member03.gt.v1"
PREPARED_SCHEMA = "member06.prepared-data.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def development_path(root, relative, split):
    """Validate the partition BEFORE opening a path, including through symlinks."""
    if split not in {"train", "val"}:
        raise ValueError("only train/val content access is allowed")
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("expected a POSIX relative path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or rel.parts[0] != split or ".." in rel.parts:
        raise ValueError("image path escapes declared development partition")
    root = Path(root).resolve()
    partition = (root / split).resolve()
    if partition != root / split:
        raise ValueError("partition symlinks are not accepted")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(partition):
        raise ValueError("path resolves outside its development partition")
    return candidate


def _gt_path(root, relative, split):
    path = development_path(Path(root) / "统一GT", relative, split).with_suffix(".json")
    if not path.resolve().is_relative_to((Path(root).resolve() / "统一GT" / split)):
        raise ValueError("GT path resolves outside its development partition")
    return path


def object_counts(gt):
    if not isinstance(gt.get("objects"), list):
        raise ValueError("GT objects must be a list, including for an empty scene")
    counts = {str(i): 0 for i in range(6)}
    for obj in gt["objects"]:
        class_id = obj.get("class_id")
        if type(class_id) is not int or class_id not in range(6):
            raise ValueError("unknown class_id must be reviewed; never remap it to a catalogue class")
        counts[str(class_id)] += 1
    return counts


def unresolved_flags(value):
    """Do not turn ignored, unjudgeable or unknown-specification objects into FAIL."""
    names = {"ignore", "ignored", "unjudgeable", "unknown_specification", "unknown_spec", "review_required"}
    if isinstance(value, dict):
        return any((key.lower() in names and item is not False and item is not None)
                   or (key.lower() == "judgeable" and item is False)
                   or unresolved_flags(item) for key, item in value.items())
    if isinstance(value, list):
        return any(unresolved_flags(item) for item in value)
    return False


def classify_reviewed_gt(gt, expected_counts, review):
    """Pure label rule; never a prediction and never a substitute for human review."""
    expected = normalize_counts(expected_counts)
    if review.get("review_status") != "APPROVED" or any(
        review.get(key) is not True for key in ("judgeable", "annotation_complete", "physical_specifications_verified")
    ) or unresolved_flags(gt):
        return {"status": "REVIEW_REQUIRED", "label": None}
    counts = object_counts(gt)
    if "actual_counts" in gt and gt["actual_counts"] != counts:
        raise ValueError("stored actual_counts disagree with object annotations")
    passed = counts["0"] == 0 and [counts[str(i)] for i in range(1, 6)] == expected
    return {"status": "OK", "label": "PASS" if passed else "FAIL", "actual_counts": counts}


def audit_dataset(root=DEFAULT_ROOT, catalogue_path=DEFAULT_CATALOGUE):
    root = Path(root).resolve()
    manifest = root / "split_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(MANIFEST_COLUMNS) or set(reader.fieldnames or []) != MANIFEST_COLUMNS:
            raise ValueError("unexpected team split_manifest columns")
        rows = list(reader)
    if not rows:
        raise ValueError("team split manifest is empty")
    if any(row["split"] not in {"train", "val", "test", "stage2_test", "stage3_test"} for row in rows):
        raise ValueError("unrecognized split; refusing content access")
    catalogue, mapping = read_json(catalogue_path), read_json(root / "统一GT/class_mapping.json")
    classes = {item["id"]: item for item in catalogue["classes"]}
    mapped = {item["id"]: item for item in mapping["classes"]}
    if len(classes) != 6 or set(classes) != set(range(6)) or set(mapped) != set(classes):
        raise ValueError("class IDs must match public catalogue 0..5")
    if any(mapped[i]["name"] != classes[i]["name"] or
           mapped[i]["catalogue_specification"] != classes[i]["specification"] for i in classes):
        raise ValueError("member03 class mapping conflicts with public catalogue")
    label_to_id = {item["labelme_label"]: item["id"] for item in mapped.values()}
    expected = catalogue.get("reference_expected_counts")
    report = {"schema_version": "member06.team-audit.v1", "dataset_root": str(root),
              "source_manifest_sha256": sha256_file(manifest), "catalogue_sha256": sha256_file(catalogue_path),
              "class_mapping_sha256": sha256_file(root / "统一GT/class_mapping.json"),
              "split_counts": dict(Counter(row["split"] for row in rows)),
              "reference_expected_counts": expected, "reference_counts_status": catalogue.get("reference_counts_status"),
              "unverified_catalogue_ids": [i for i, item in mapped.items() if item.get("physical_specification_verified") is False],
              "test_content_read": False, "formal_labels_generated": 0,
              "errors": [], "images": [], "per_split": {}}
    raw_hashes, rgb_hashes = defaultdict(list), defaultdict(list)
    for row in rows:
        # Hash strings from the partition manifest are metadata, not test file reads.
        raw_hashes[row["sha256"]].append({"split": row["split"], "path": row["destination"]})
    seen_paths = set()
    for split in ("train", "val"):
        totals = Counter()
        vectors = Counter()
        subset = [row for row in rows if row["split"] == split]
        for row in subset:
            relative = row["destination"]
            try:
                path = development_path(root, relative, split)
                if relative in seen_paths:
                    raise ValueError("duplicate destination in split manifest")
                seen_paths.add(relative)
                if row["source"] != relative or row["filename"] != path.name:
                    raise ValueError("source/destination/filename do not match current package layout")
                if sha256_file(path) != row["sha256"]:
                    raise ValueError("image hash mismatch; partial download or changed source")
                image = read_image(path)
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                rgb_hashes[digest].append({"split": split, "path": relative})
                raw_path = path.with_suffix(".json").resolve()
                if not raw_path.is_relative_to(root / split):
                    raise ValueError("annotation symlink escapes development partition")
                raw, gt_path = read_json(raw_path), _gt_path(root, relative, split)
                gt = read_json(gt_path)
                if gt.get("schema_version") != GT_SCHEMA or gt.get("bbox_convention") != "original_pixel_xyxy_continuous_half_open":
                    raise ValueError("unsupported GT schema/bbox convention")
                if gt.get("split") != split or gt.get("image_relative_path") != relative or gt.get("image_id") != path.name:
                    raise ValueError("GT partition/image identity mismatch")
                if (gt.get("image_width"), gt.get("image_height")) != image.size or (raw.get("imageWidth"), raw.get("imageHeight")) != image.size:
                    raise ValueError("annotation/image dimensions disagree")
                if gt.get("source_annotation_sha256") != sha256_file(raw_path):
                    raise ValueError("GT source annotation hash mismatch")
                counts = object_counts(gt)
                if gt.get("actual_counts") != counts:
                    raise ValueError("GT actual_counts mismatch")
                shapes = raw.get("shapes")
                if not isinstance(shapes, list) or len(shapes) != len(gt["objects"]):
                    raise ValueError("raw/GT annotation count mismatch")
                for obj, shape in zip(gt["objects"], shapes):
                    if obj.get("class_name") != classes[obj["class_id"]]["name"] or label_to_id.get(shape.get("label")) != obj["class_id"]:
                        raise ValueError("raw/GT class mapping mismatch")
                    bbox = obj.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in bbox):
                        raise ValueError("bbox must be four finite real numbers")
                    x1, y1, x2, y2 = bbox
                    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                        raise ValueError("invalid/out-of-bounds bbox")
                    points = shape.get("points")
                    if shape.get("shape_type") != "rectangle" or not isinstance(points, list) or len(points) != 2:
                        raise ValueError("unsupported raw annotation shape")
                    raw_box = [min(p[0] for p in points), min(p[1] for p in points),
                               max(p[0] for p in points), max(p[1] for p in points)]
                    if any(abs(a - b) > 1e-5 for a, b in zip(raw_box, bbox)):
                        raise ValueError("raw rectangle/GT bbox mismatch")
                pending = any(s.get("flags", {}).get("auto_label_review") or
                              "human review required" in s.get("description", "").lower() for s in shapes)
                info = {"image_path": relative, "split": split, "capture_condition": row["category"],
                        "image_sha256": row["sha256"], "rgb_sha256": digest,
                        "annotation_sha256": sha256_file(raw_path), "gt_sha256": sha256_file(gt_path),
                        "actual_counts": counts, "raw_auto_review_pending": bool(pending),
                        "unresolved_ignore_or_judgment": unresolved_flags(gt) or unresolved_flags(raw),
                        "has_scene_id": bool(gt.get("scene_id")), "has_expected_counts": "expected_counts" in gt,
                        "has_image_label": any(key in gt for key in ("label", "image_label", "pass_fail"))}
                report["images"].append(info)
                totals.update({f"class_{i}": counts[str(i)] for i in range(6)})
                totals["images_checked"] += 1
                totals["raw_auto_review_pending_images"] += int(bool(pending))
                totals["missing_scene_id"] += int(not info["has_scene_id"])
                totals["missing_expected_counts"] += int(not info["has_expected_counts"])
                totals["missing_image_label"] += int(not info["has_image_label"])
                vectors[",".join(str(counts[str(i)]) for i in range(6))] += 1
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                report["errors"].append({"image_path": relative, "error": str(exc)})
        # Inventory of development partitions only; no traversal inside test directories.
        actual_images = {p.relative_to(root).as_posix() for p in (root / split).rglob("*.png")}
        listed_images = {row["destination"] for row in subset}
        if actual_images != listed_images:
            report["errors"].append({"split": split, "error": "PNG inventory differs from manifest",
                                     "unlisted": sorted(actual_images - listed_images), "missing": sorted(listed_images - actual_images)})
        report["per_split"][split] = {**totals, "annotation_count_vectors_id_0_to_5": dict(vectors)}
    report["duplicate_raw_hash_metadata"] = [group for group in raw_hashes.values() if len(group) > 1]
    report["duplicate_development_rgb"] = [group for group in rgb_hashes.values() if len(group) > 1]
    report["readiness"] = "D"
    report["status"] = "NOT_READY"
    report["readiness_reason"] = "Audit alone does not establish approved counts, scene independence, human review, specifications or binary labels."
    return report


def approval_template(report):
    return {"schema_version": "member06.data-approval.v1", "status": "PENDING", "approved_by": None,
            "decision_reference": None, "dataset_version": None,
            "source_manifest_sha256": report["source_manifest_sha256"], "catalogue_sha256": report["catalogue_sha256"],
            "fixed_expected_counts": None, "scene_split_verified": False,
            "catalogue_specs_verified": False, "dataset_usage_approved": False,
            "note": "Member-local handoff template. Requires actual team review; do not mark approved merely to run training."}


def review_template(report):
    return {"schema_version": "member06.image-reviews.v1", "images": [
        {**{key: image[key] for key in ("image_path", "image_sha256", "annotation_sha256", "gt_sha256")},
         "scene_id": None, "review_status": "PENDING", "judgeable": None, "annotation_complete": None,
         "physical_specifications_verified": None, "reviewer": None, "review_reference": None}
        for image in report["images"]]}


def require_approval(approval):
    if approval.get("schema_version") != "member06.data-approval.v1" or approval.get("status") != "APPROVED":
        raise ValueError("WAITING_UPSTREAM: fixed counts and dataset approval are PENDING")
    for key in ("approved_by", "decision_reference", "dataset_version"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            raise ValueError(f"WAITING_UPSTREAM: missing {key}")
    for key in ("scene_split_verified", "catalogue_specs_verified", "dataset_usage_approved"):
        if approval.get(key) is not True:
            raise ValueError(f"WAITING_UPSTREAM: {key} is not verified")
    return normalize_counts(approval.get("fixed_expected_counts"))


def prepare_dataset(root, catalogue_path, approval_path, reviews_path, output):
    """Do not write any classification manifest until all required checks pass."""
    approval = read_json(approval_path)
    expected = require_approval(approval)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists; choose a new prepared data version")
    report = audit_dataset(root, catalogue_path)
    if report["errors"] or report["duplicate_raw_hash_metadata"] or report["duplicate_development_rgb"]:
        raise ValueError("dataset integrity/duplicate checks failed; inspect audit report")
    for key in ("source_manifest_sha256", "catalogue_sha256"):
        if approval.get(key) != report[key]:
            raise ValueError(f"approval is stale: {key} differs")
    reviews = read_json(reviews_path)
    if reviews.get("schema_version") != "member06.image-reviews.v1" or not isinstance(reviews.get("images"), list):
        raise ValueError("unsupported review ledger")
    by_path = {row["image_path"]: row for row in reviews["images"]}
    if len(by_path) != len(reviews["images"]) or set(by_path) != {r["image_path"] for r in report["images"]}:
        raise ValueError("review ledger must cover every development image exactly once, without test rows")
    manifests, lineage, excluded, scenes = {"train": [], "val": []}, [], [], {}
    root = Path(root).resolve()
    for image in report["images"]:
        relative, split = image["image_path"], image["split"]
        review = by_path[relative]
        for key in ("image_sha256", "annotation_sha256", "gt_sha256"):
            if review.get(key) != image[key]:
                raise ValueError(f"stale per-image review for {relative}: {key}")
        if review.get("review_status") != "APPROVED" or any(
            not isinstance(review.get(k), str) or not review[k].strip() for k in ("scene_id", "reviewer", "review_reference")
        ):
            raise ValueError(f"WAITING_UPSTREAM: unresolved scene/human review for {relative}")
        scene_id = review["scene_id"]
        if scene_id in scenes and scenes[scene_id] != split:
            raise ValueError("reviewed scene_id crosses train/val; never repartition automatically")
        scenes[scene_id] = split
        gt = read_json(_gt_path(root, relative, split))
        if image["unresolved_ignore_or_judgment"]:
            label = {"status": "REVIEW_REQUIRED", "label": None}
        else:
            label = classify_reviewed_gt(gt, expected, review)
        if label["status"] != "OK":
            excluded.append({"image_path": relative, "reason": "REVIEW_REQUIRED; no label assigned"})
            continue
        absolute = str(development_path(root, relative, split))
        manifests[split].append({"image_path": absolute, "label": label["label"], "scene_id": scene_id,
                                 "split": split, "data_kind": "REAL"})
        lineage.append({**image, "scene_id": scene_id, "label": label["label"]})
    for split in manifests:
        if {row["label"] for row in manifests[split]} != {"PASS", "FAIL"}:
            raise ValueError(f"WAITING_UPSTREAM: {split} needs both trusted PASS and FAIL; do not fabricate or relabel examples")
    # All semantic checks precede writes. Interrupted exports lack a final provenance file
    # and therefore cannot pass the training preflight.
    output.mkdir(parents=True, exist_ok=False)
    approval_copy, review_copy = output / "approval.json", output / "reviews.json"
    write_new_json(approval_copy, approval)
    write_new_json(review_copy, reviews)
    for split, rows in manifests.items():
        with (output / f"{split}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "label", "scene_id", "split", "data_kind"])
            writer.writeheader()
            writer.writerows(rows)
    provenance = {"schema_version": PREPARED_SCHEMA, "dataset_root": str(root),
                  "catalogue_path": str(Path(catalogue_path).resolve()), "expected_counts": expected,
                  "source_manifest_sha256": report["source_manifest_sha256"],
                  "class_mapping_sha256": report["class_mapping_sha256"],
                  "approval_sha256": sha256_file(approval_copy), "reviews_sha256": sha256_file(review_copy),
                  "manifest_sha256": {s: sha256_file(output / f"{s}.csv") for s in manifests},
                  "label_counts": {s: dict(Counter(r["label"] for r in values)) for s, values in manifests.items()},
                  "scene_counts": {s: len({r["scene_id"] for r in values}) for s, values in manifests.items()},
                  "lineage": lineage, "excluded": excluded, "test_content_read": False}
    write_new_json(output / "provenance.json", provenance)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "prepare"])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--catalogue", default=DEFAULT_CATALOGUE)
    parser.add_argument("--output", required=True, help="audit JSON path, or a new prepared-data directory")
    parser.add_argument("--approval-template")
    parser.add_argument("--review-template")
    parser.add_argument("--approval")
    parser.add_argument("--reviews")
    args = parser.parse_args(argv)
    try:
        if args.mode == "audit":
            destinations = [p for p in (args.output, args.approval_template, args.review_template) if p]
            if len({Path(p).resolve() for p in destinations}) != len(destinations) or any(Path(p).exists() for p in destinations):
                raise ValueError("output/template already exists or paths are repeated")
            report = audit_dataset(args.root, args.catalogue)
            write_new_json(args.output, report)
            if args.approval_template:
                write_new_json(args.approval_template, approval_template(report))
            if args.review_template:
                write_new_json(args.review_template, review_template(report))
            print(json.dumps({k: report[k] for k in ("status", "readiness", "split_counts", "per_split", "test_content_read")}, ensure_ascii=False, indent=2))
            print(f"Audit complete; no training labels generated. Report: {args.output}")
            return 2 if report["errors"] else 0
        if not args.approval or not args.reviews:
            raise ValueError("prepare requires --approval and --reviews from actual upstream review")
        report = prepare_dataset(args.root, args.catalogue, args.approval, args.reviews, args.output)
        print(json.dumps({"status": "PREPARED_TRAIN_VAL_ONLY", "label_counts": report["label_counts"]}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"DATA_PREPARATION_FAILED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
