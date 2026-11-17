"""Audit the complete supplied dataset after a valid model freeze."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from ..data import identity, load_manifest, reject_overlap, sha256
from ..evaluate import validate_freeze, write_json
from .prepare import LABELS, NEW, OLD, PREFIX, object_hash


def audit(data, frozen, output):
    validate_freeze(frozen)
    index = json.loads((Path(__file__).resolve().parents[1] /
                        "configs/member03_sources.json").read_text(encoding="utf-8"))
    verified = 0
    for source, digest in index["files"].items():
        if source.startswith(OLD) and any(
                section in source.removeprefix(OLD).split("/")
                for section in ("train", "val", "test")):
            local = data / "old" / source.removeprefix(OLD)
        elif source.startswith(PREFIX + NEW):
            local = data / "new" / source.removeprefix(PREFIX + NEW)
        elif source == PREFIX + "data_20261114_manifest.json":
            local = data / "source_manifest_20261114.json"
        else:
            local = data / "source_documents/old" / source.removeprefix(PREFIX + "full_dataset_upload/")
        if object_hash(local.read_bytes()) != digest:
            raise ValueError(f"Source changed: {local}")
        verified += 1
    records = []
    for filename in ("provenance.json", "test_provenance.json"):
        records.extend(json.loads((data / filename).read_text(encoding="utf-8"))["sources"])
    splits = {name: load_manifest(data / f"{name}.csv") for name in ("train", "valid", "test")}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        reject_overlap(identity(splits[left]), identity(splits[right]))
    groups = defaultdict(list)
    for record in records:
        image = data / record["image_path"]
        raw = image.with_suffix(".json")
        annotation = json.loads(raw.read_text(encoding="utf-8-sig"))
        with Image.open(image) as opened:
            width, height = opened.size
        if (annotation["imageWidth"], annotation["imageHeight"]) != (width, height):
            raise ValueError(f"Image dimensions disagree: {raw}")
        counts = Counter()
        for obj in annotation["shapes"]:
            counts[LABELS[obj["label"]]] += 1
            if obj["shape_type"] != "rectangle" or len(obj["points"]) != 2:
                raise ValueError(f"Unexpected box type: {raw}")
            if not all(0 <= x <= width and 0 <= y <= height for x, y in obj["points"]):
                raise ValueError(f"Box outside image: {raw}")
        if any(counts[str(i)] != record["counts"][str(i)] for i in range(6)):
            raise ValueError(f"Raw label and derived counts disagree: {raw}")
        groups[record["split"]].append(record)
    support = {}
    for split, rows in groups.items():
        conditions = defaultdict(Counter)
        for row in rows:
            conditions[row["source_session"]][row["label"]] += 1
        support[split] = {
            "images": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
            "objects": sum(sum(r["counts"].values()) for r in rows),
            "class_counts": {str(i): sum(r["counts"][str(i)] for r in rows) for i in range(6)},
            "conditions": {key: dict(value) for key, value in conditions.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {
        "commit": index["commit"], "verified_source_files": verified,
        "freeze_sha256": sha256(frozen), "support": support,
        "images": len(records), "objects": sum(s["objects"] for s in support.values()),
        "cross_partition_exact_duplicates": 0, "raw_annotation_count_and_bounds": "PASS",
        "scene_independence": "UNVERIFIED", "manual_annotation_review": "NOT_CONFIRMED",
        "physical_specification_and_licence_review": "NOT_CONFIRMED"})
    print(json.dumps(support, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.data, args.freeze, args.out)


if __name__ == "__main__":
    main()
