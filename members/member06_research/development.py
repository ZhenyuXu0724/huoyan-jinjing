"""Versioned multi-package development data. No test content or implicit repartition."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from .config import Config, normalize_counts
from .image_io import read_image
from .team_data import (audit_dataset, approval_template, classify_reviewed_gt, read_json,
                        require_approval, sha256_file, unresolved_flags, write_new_json)

SCHEMA = "member06.prepared-data.v2"
DEFAULT_SOURCES = "members/member06_research/configs/data_sources.json"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_path(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("expected POSIX relative source path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError("source path escapes root")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("source symlink escapes root")
    # Disallow aliases to test or another partition even inside the root.
    if path != root / relative:
        raise ValueError("source symlink/alias is not accepted")
    return path


def audit(sources_path):
    sources = read_json(sources_path)
    if sources.get("schema_version") != "member06.sources.v1":
        raise ValueError("unsupported sources configuration")
    legacy = audit_dataset(sources["legacy_root"], sources["catalogue"])
    root = Path(sources["legacy_root"]).resolve()
    report = {"schema_version": "member06.development-audit.v2", "sources": sources,
              "sources_path": str(Path(sources_path).resolve()), "sources_sha256": sha256_file(sources_path),
              "catalogue_sha256": legacy["catalogue_sha256"], "source_files": {},
              "images": [], "errors": list(legacy["errors"]), "test_content_read": False,
              "formal_labels_generated": 0, "status": "REVIEW_REQUIRED"}
    for path in (root / "split_manifest.csv", root / "统一GT/class_mapping.json", Path(sources["catalogue"]).resolve()):
        report["source_files"][str(path)] = sha256_file(path)
    # Compare only the hashes already published in split metadata; never open test content.
    with (root / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        test_hashes = {r["sha256"] for r in csv.DictReader(stream) if r["split"] not in {"train", "val"}}
    mapping = read_json(root / "统一GT/class_mapping.json")["classes"]
    labels = {c["labelme_label"]: c["id"] for c in mapping}
    for row in legacy["images"]:
        report["images"].append({**row, "image_path": str(root / row["image_path"]),
                                  "source_kind": "legacy", "source_package": "parts_v1"})
    for package in sources.get("supplements", []):
        package_root = Path(package["root"]).resolve()
        manifest_path = Path(package["manifest"]).resolve()
        index = read_json(manifest_path)
        report["source_files"][str(manifest_path)] = sha256_file(manifest_path)
        if index.get("class_ids") != {str(c["id"]): c["labelme_label"] for c in mapping}:
            raise ValueError("supplement class mapping differs from legacy/public catalogue")
        partitions = package["partitions"]
        if not partitions or any(v not in {"train", "val", "test"} for v in partitions.values()):
            raise ValueError("explicit upstream train/val/test partition mapping required")
        for name, role in partitions.items():
            if ("验证" in name or name.lower() in {"val", "valid", "validation"}) and role != "val":
                raise ValueError("validation-only source cannot be reassigned")
            if ("测试" in name or "封存" in name or "test" in name.lower()) and role != "test":
                raise ValueError("test source cannot be reassigned to development")
        if set(index["splits"]) != set(partitions):
            raise ValueError("source index partitions differ from configuration")
        prefix = index["source_folder"]
        entries = {}
        for entry in index["files"]:
            rel = PurePosixPath(entry["path"])
            # Validate all roles from metadata before reading any content.
            if len(rel.parts) != 3 or rel.parts[0] != prefix or rel.parts[1] not in partitions:
                raise ValueError("unexpected supplement path/partition")
            if entry["path"] in entries:
                raise ValueError("duplicate source index entry")
            entries[entry["path"]] = entry
            if partitions[rel.parts[1]] == "test" and rel.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                test_hashes.add(entry["sha256"])
        for partition, split in partitions.items():
            if split == "test":
                continue
            allowed = safe_path(package_root, f"{prefix}/{partition}")
            listed = {p for p in entries if PurePosixPath(p).parts[1] == partition}
            actual = {p.relative_to(package_root).as_posix() for p in allowed.iterdir()
                      if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".json"}}
            if listed != actual:
                report["errors"].append({"source": str(allowed), "error": "source inventory mismatch"})
            image_entries = [p for p in listed if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg"}]
            if len(image_entries) != index["splits"][partition]["images"]:
                report["errors"].append({"source": str(allowed), "error": "image count mismatch"})
            for relative in sorted(image_entries):
                try:
                    path = safe_path(package_root, relative)
                    annotation = safe_path(package_root, str(PurePosixPath(relative).with_suffix(".json")))
                    for source in (path, annotation):
                        recorded = entries[source.relative_to(package_root).as_posix()]
                        if sha256_file(source) != recorded["sha256"] or source.stat().st_size != recorded["bytes"]:
                            raise ValueError("source bytes/hash mismatch")
                    image, raw = read_image(path), read_json(annotation)
                    if (raw["imageWidth"], raw["imageHeight"]) != image.size:
                        raise ValueError("annotation/image dimension mismatch")
                    objects = []
                    for shape in raw["shapes"]:
                        if shape.get("label") not in labels or shape.get("shape_type") != "rectangle":
                            raise ValueError("unknown label or unsupported shape; review upstream")
                        pts = shape["points"]
                        if len(pts) != 2 or any(len(p) != 2 for p in pts):
                            raise ValueError("rectangle needs two xy points")
                        if any(type(x) not in (int, float) or not math.isfinite(x) for p in pts for x in p):
                            raise ValueError("coordinates must be finite")
                        x1, x2 = sorted(p[0] for p in pts)
                        y1, y2 = sorted(p[1] for p in pts)
                        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                            raise ValueError("invalid/out-of-bounds rectangle")
                        objects.append({"class_id": labels[shape["label"]]})
                    counts = {str(i): sum(o["class_id"] == i for o in objects) for i in range(6)}
                    report["images"].append({"image_path": str(path), "split": split,
                        "source_kind": "labelme", "source_package": str(manifest_path),
                        "image_sha256": sha256_file(path), "annotation_sha256": sha256_file(annotation),
                        "gt_sha256": None, "rgb_sha256": hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest(),
                        "actual_counts": counts, "unresolved_ignore_or_judgment": unresolved_flags(raw),
                        "raw_auto_review_pending": any(s.get("flags", {}).get("auto_label_review") or
                            "human review required" in s.get("description", "").lower() for s in raw["shapes"])})
                except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                    report["errors"].append({"image_path": relative, "error": str(exc)})
    for index_path in sources.get("flat_packages", []):
        from .corpus_audit import audit_flat
        images, errors = audit_flat(index_path, labels)
        report["source_files"][str(Path(index_path).resolve())] = sha256_file(index_path)
        report["images"].extend(images)
        report["errors"].extend(errors)
    for key in ("image_path", "image_sha256", "rgb_sha256"):
        values = [r[key] for r in report["images"]]
        if len(set(values)) != len(values):
            report["errors"].append({"error": f"duplicate development {key}"})
    if any(row["image_sha256"] in test_hashes for row in report["images"]):
        report["errors"].append({"error": "development image overlaps test SHA256 metadata"})
    report["test_hash_metadata_checked"] = True
    expected = normalize_counts(read_json(sources["catalogue"])["reference_expected_counts"])
    report["reference_expected_counts"] = expected
    report["per_split"] = {}
    for split in ("train", "val"):
        rows = [r for r in report["images"] if r["split"] == split]
        matches = sum(r["actual_counts"]["0"] == 0 and [r["actual_counts"][str(i)] for i in range(1, 6)] == expected for r in rows)
        report["per_split"][split] = {"images": len(rows), "reference_rule_matches_unreviewed": matches,
                                      "reference_rule_mismatches_unreviewed": len(rows) - matches}
    report["source_manifest_sha256"] = fingerprint({k: report[k] for k in
        ("sources_sha256", "source_files", "images")})
    return report


def templates(report):
    approval = approval_template(report)
    reviews = {"schema_version": "member06.image-reviews.v1", "images": [
        {**{k: r[k] for k in ("image_path", "image_sha256", "annotation_sha256", "gt_sha256")},
         "scene_id": None, "review_status": "PENDING", "judgeable": None,
         "annotation_complete": None, "physical_specifications_verified": None,
         "reviewer": None, "review_reference": None} for r in report["images"]]}
    return approval, reviews


def reviewed_rows(report, approval, reviews):
    expected = require_approval(approval)
    if report["errors"]:
        raise ValueError("source integrity audit failed")
    for key in ("source_manifest_sha256", "catalogue_sha256"):
        if approval.get(key) != report[key]:
            raise ValueError(f"stale approval: {key}")
    if reviews.get("schema_version") != "member06.image-reviews.v1":
        raise ValueError("unsupported reviews")
    by_path = {r["image_path"]: r for r in reviews["images"]}
    if len(by_path) != len(reviews["images"]) or set(by_path) != {r["image_path"] for r in report["images"]}:
        raise ValueError("reviews must cover all development images exactly once")
    output, scenes = [], {}
    for image in report["images"]:
        review = by_path[image["image_path"]]
        if any(review.get(k) != image[k] for k in ("image_sha256", "annotation_sha256", "gt_sha256")):
            raise ValueError("stale image review")
        if any(not isinstance(review.get(k), str) or not review[k].strip() for k in ("scene_id", "reviewer", "review_reference")):
            raise ValueError("WAITING_UPSTREAM: trusted scene_id and reviewer/reference required")
        if review["scene_id"].upper().startswith("UNVERIFIED"):
            raise ValueError("WAITING_UPSTREAM: filename-based UNVERIFIED scene IDs are not reviewed scene identities")
        scene, split = review["scene_id"], image["split"]
        if scene in scenes and scenes[scene] != split:
            raise ValueError("scene crosses train/val; preserve upstream partition and request correction")
        scenes[scene] = split
        gt = {"objects": [{"class_id": i} for i in range(6) for _ in range(image["actual_counts"][str(i)])],
              "review_required": image["unresolved_ignore_or_judgment"]}
        result = classify_reviewed_gt(gt, expected, review)
        if result["status"] != "OK":
            raise ValueError("WAITING_UPSTREAM: unresolved image review; no guessed label")
        output.append({"image_path": image["image_path"], "label": result["label"],
                       "scene_id": scene, "split": split, "data_kind": "REAL"})
    for split in ("train", "val"):
        if {r["label"] for r in output if r["split"] == split} != {"PASS", "FAIL"}:
            raise ValueError(f"WAITING_UPSTREAM: {split} needs trusted PASS AND FAIL; validation negatives cannot train")
    return output


def prepare(sources, approval_path, reviews_path, output):
    approval, reviews = read_json(approval_path), read_json(reviews_path)
    require_approval(approval)
    report = audit(sources)
    rows = reviewed_rows(report, approval, reviews)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "approval.json", approval)
    write_new_json(output / "reviews.json", reviews)
    for split in ("train", "val"):
        with (output / f"{split}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(r for r in rows if r["split"] == split)
    record = {"schema_version": SCHEMA, "sources_path": str(Path(sources).resolve()),
              "source_manifest_sha256": report["source_manifest_sha256"],
              "expected_counts": require_approval(approval), "test_content_read": False,
              "files": {name: sha256_file(output / name) for name in ("train.csv", "val.csv", "approval.json", "reviews.json")},
              "lineage": rows, "audit": report}
    write_new_json(output / "provenance.json", record)
    config = Config(train_manifest=str(output / "train.csv"), val_manifest=str(output / "val.csv"),
                    data_provenance=str(output / "provenance.json"), counts_rule_status="CONFIRMED",
                    supported_counts=record["expected_counts"])
    write_new_json(output / "config.json", config.to_dict())
    return record


def verify(config):
    parent = Path(config.data_provenance).resolve().parent
    record = read_json(config.data_provenance)
    if record["schema_version"] != SCHEMA or record.get("test_content_read") is not False or config.counts_rule_status != "CONFIRMED":
        raise ValueError("unapproved development provenance")
    names = {"train.csv", "val.csv", "approval.json", "reviews.json"}
    if set(record["files"]) != names:
        raise ValueError("incomplete prepared-data file manifest")
    for name in names:
        if sha256_file(parent / name) != record["files"][name]:
            raise ValueError(f"prepared file changed: {name}")
    for split in ("train", "val"):
        if Path(getattr(config, f"{split}_manifest")).resolve() != parent / f"{split}.csv":
            raise ValueError("config differs from approved manifest path")
    approval = read_json(parent / "approval.json")
    if require_approval(approval) != config.supported_counts or record["expected_counts"] != config.supported_counts:
        raise ValueError("fixed counts differ from approved rule")
    report = audit(record["sources_path"])
    if report["source_manifest_sha256"] != record["source_manifest_sha256"] or record.get("audit") != report:
        raise ValueError("source data/config changed; create a reviewed new version")
    expected = reviewed_rows(report, approval, read_json(parent / "reviews.json"))
    if record["lineage"] != expected:
        raise ValueError("provenance lineage changed")
    for split in ("train", "val"):
        with (parent / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            if list(csv.DictReader(stream)) != [r for r in expected if r["split"] == split]:
                raise ValueError("manifest labels/scene/split differ from reviewed source")
    return record


DEV_SCHEMA = "member06.prepared-dev.v1"
DEV_LIMITATIONS = [
    "DEV_ONLY: annotation-derived labels; not final physical specification/annotation approval.",
    "Physical scene independence is unverified unless a real scene ledger is supplied; image IDs are not scenes.",
    "PASS/FAIL are confounded with capture batches; high Val scores do not prove count/specification reasoning.",
    "Historical HOG Val use is retained; old published Test is excluded from training/Val selection.",
    "Reference expected counts are provisional; no formal freeze or independent-generalization claim.",
]


def dev_rows(report, expected, scene_ledger=None):
    from .corpus_audit import describe
    if report["errors"]:
        raise ValueError("DEV_ONLY blocked: integrity or known source/partition conflict")
    identities = {}
    if scene_ledger is not None:
        if scene_ledger.get("schema_version") != "member06.scene-ledger.v1":
            raise ValueError("unsupported physical scene ledger")
        entries = scene_ledger["images"]
        identities = {r["image_path"]: r["scene_id"] for r in entries}
        if len(identities) != len(entries) or set(identities) != {r["image_path"] for r in report["images"]}:
            raise ValueError("scene ledger must cover every development image exactly once")
    rows, scenes = [], {}
    for image in report["images"]:
        if image["split"] not in {"train", "val"}:
            raise ValueError("DEV_ONLY never imports test roles")
        label = describe(image, expected)
        if label["annotation_label"] is None:
            raise ValueError("DEV_ONLY blocked: unresolved ignore/unknown/judgment annotation")
        scene = identities.get(image["image_path"], "UNVERIFIED:"+image["image_sha256"])
        if not isinstance(scene,str) or not scene.strip():
            raise ValueError("empty scene identity")
        if scene in scenes and scenes[scene] != image["split"]:
            raise ValueError("known scene crosses train/val")
        scenes[scene] = image["split"]
        rows.append({"image_path":image["image_path"], "label":label["annotation_label"],
                     "scene_id":scene, "split":image["split"], "data_kind":"REAL"})
    for split in ("train","val"):
        if {r["label"] for r in rows if r["split"]==split} != {"PASS","FAIL"}:
            raise ValueError(f"DEV_ONLY {split} needs both annotation-derived classes")
    return rows


def prepare_dev(sources, output, expected_counts=None, scene_ledger_path=None):
    report = audit(sources)
    expected = normalize_counts(expected_counts if expected_counts is not None else report["reference_expected_counts"])
    ledger = read_json(scene_ledger_path) if scene_ledger_path else None
    rows = dev_rows(report, expected, ledger)
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    for split in ("train","val"):
        with (output/f"{split}.csv").open("x",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows(r for r in rows if r["split"]==split)
    record={"schema_version":DEV_SCHEMA, "evidence_status":"DEV_ONLY", "sources_path":str(Path(sources).resolve()),
            "source_manifest_sha256":report["source_manifest_sha256"], "expected_counts":expected,
            "lineage":rows, "audit":report, "test_content_read":False, "scene_ledger":ledger,
            "scene_ledger_sha256":sha256_file(scene_ledger_path) if scene_ledger_path else None,
            "limitations":list(DEV_LIMITATIONS), "formal_approval_claimed":False,
            "training_requires_explicit_allow_dev_only":True,
            "files":{f"{split}.csv":sha256_file(output/f"{split}.csv") for split in ("train","val")}}
    write_new_json(output/"provenance.json",record)
    config=Config(training_tier="DEV_ONLY",train_manifest=str(output/"train.csv"),val_manifest=str(output/"val.csv"),
                  data_provenance=str(output/"provenance.json"),supported_counts=expected)
    write_new_json(output/"config.json",config.to_dict())
    return record


def verify_dev(config):
    if config.training_tier != "DEV_ONLY" or not config.data_provenance or config.counts_rule_status != "PROVISIONAL_PENDING_PILOT":
        raise ValueError("DEV_ONLY requires explicit tier, provisional counts and development provenance")
    parent=Path(config.data_provenance).resolve().parent;record=read_json(config.data_provenance)
    if record.get("schema_version")!=DEV_SCHEMA or record.get("evidence_status")!="DEV_ONLY" or record.get("test_content_read") is not False:
        raise ValueError("invalid DEV_ONLY provenance")
    if record.get("formal_approval_claimed") is not False or record.get("limitations") != DEV_LIMITATIONS:
        raise ValueError("DEV_ONLY limitations/status changed")
    if config.supported_counts!=record["expected_counts"]:
        raise ValueError("DEV_ONLY label rule differs from config")
    if set(record["files"])!={"train.csv","val.csv"}:raise ValueError("incomplete DEV_ONLY files")
    report=audit(record["sources_path"])
    if report!=record["audit"] or report["source_manifest_sha256"]!=record["source_manifest_sha256"]:
        raise ValueError("DEV_ONLY source changed; prepare new version")
    rows=dev_rows(report,config.supported_counts,record["scene_ledger"])
    if rows!=record["lineage"]:raise ValueError("DEV_ONLY lineage changed")
    for split in ("train","val"):
        path=parent/f"{split}.csv"
        if Path(getattr(config,f"{split}_manifest")).resolve()!=path or sha256_file(path)!=record["files"][path.name]:
            raise ValueError("DEV_ONLY manifest changed")
        with path.open(encoding="utf-8",newline="") as stream:
            if list(csv.DictReader(stream))!=[r for r in rows if r["split"]==split]:raise ValueError("DEV_ONLY labels/roles changed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "prepare", "prepare-dev"])
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--output", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--reviews")
    parser.add_argument("--scene-ledger")
    parser.add_argument("--expected-counts", help="JSON reference vector for DEV_ONLY")
    args = parser.parse_args()
    try:
        if args.mode == "audit":
            report = audit(args.sources)
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=False)
            write_new_json(out / "audit.json", report)
            approval, reviews = templates(report)
            write_new_json(out / "approval.json", approval)
            write_new_json(out / "reviews.json", reviews)
            print(json.dumps({"status": report["status"], "per_split": report["per_split"], "errors": report["errors"]}, ensure_ascii=False, indent=2))
            return 2 if report["errors"] else 0
        if args.mode == "prepare-dev":
            record=prepare_dev(args.sources,args.output,json.loads(args.expected_counts) if args.expected_counts else None,args.scene_ledger)
            print(json.dumps({"status":"DEV_ONLY", "images":len(record["lineage"]),"limitations":record["limitations"],"training_executed":False},ensure_ascii=False,indent=2))
            return 0
        if not args.approval or not args.reviews:
            raise ValueError("prepare requires actual approved ledger and reviews")
        record = prepare(args.sources, args.approval, args.reviews, args.output)
        print(json.dumps({"status": "PREPARED_TRAIN_VAL_ONLY", "images": len(record["lineage"])}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"DEVELOPMENT_FAILED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
