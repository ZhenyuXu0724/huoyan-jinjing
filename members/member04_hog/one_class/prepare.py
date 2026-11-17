"""Acquire pinned development sources, or provided test sources after a valid freeze."""
import argparse
import concurrent.futures
import csv
import hashlib
import http.client
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..data import sha256
from ..evaluate import validate_freeze, write_json

COMMIT = "127eeded2d9e7ceebb802b26f6ab9f3ccac9ef26"
PREFIX = "members/member03_data/"
OLD = PREFIX + "full_dataset_upload/datasets/parts_v1/"
NEW = "11.14训练补强＋干扰验证/"
LABELS = {"non_catalogue": "0", "M4x25": "1", "M5x10": "2", "M4x10": "3",
          "M4_hex_nut": "4", "flat_washer": "5"}


def fetch_bytes(path):
    url = ("https://raw.githubusercontent.com/ZhenyuXu0724/huoyan-jinjing/"
           + COMMIT + "/" + urllib.parse.quote(path, safe="/"))
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=25) as response:
                return response.read()
        except (OSError, ValueError, http.client.HTTPException):
            time.sleep(1)
    return subprocess.check_output(
        ["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "1",
         "--connect-timeout", "10", "--max-time", "60", url])


def object_hash(content):
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def prepare(destination, phase="development"):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    index = json.loads((Path(__file__).resolve().parents[1] /
                        "configs/member03_sources.json").read_text(encoding="utf-8"))
    if index["commit"] != COMMIT:
        raise ValueError("Source index version mismatch")
    tree = index["files"]
    manifest_bytes = fetch_bytes(PREFIX + "data_20261114_manifest.json")
    if object_hash(manifest_bytes) != tree[PREFIX + "data_20261114_manifest.json"]:
        raise ValueError("Upstream manifest checksum mismatch")
    source_manifest = json.loads(manifest_bytes)
    tasks = []
    allowed = ("train/", "val/") if phase == "development" else ("test/",)
    allowed_gt = tuple("统一GT/" + split for split in allowed)
    for path in tree:
        relative = path.removeprefix(OLD)
        if path.startswith(OLD) and (
                relative.startswith(allowed) and path.endswith((".png", ".json"))
                or relative.startswith(allowed_gt) and path.endswith(".json")):
            tasks.append((path, destination / "old" / relative))
        if (phase == "test" and path.startswith(PREFIX + "full_dataset_upload/")
                and not any(segment in path.split("/") for segment in ("train", "val", "test"))):
            relative_document = path.removeprefix(PREFIX + "full_dataset_upload/")
            tasks.append((path, destination / "source_documents/old" / relative_document))
    for item in source_manifest["files"]:
        relative = item["path"]
        if phase == "development" and relative.startswith((NEW + "补强数据(训练）/", NEW + "验证集_不要训练/")):
            tasks.append((PREFIX + relative, destination / "new" / Path(relative).relative_to(NEW)))

    def download(task):
        path, target = task
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and object_hash(target.read_bytes()) == tree[path]:
            return
        payload = fetch_bytes(path)
        if object_hash(payload) != tree[path]:
            raise ValueError(f"Source checksum mismatch: {path}")
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(target)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        for index, _ in enumerate(executor.map(download, tasks), 1):
            if index % 20 == 0 or index == len(tasks):
                print(f"Verified source files {index}/{len(tasks)}", flush=True)
    if phase == "development":
        (destination / "source_manifest_20261114.json").write_bytes(manifest_bytes)

    rows = {"train": [], "valid": []} if phase == "development" else {"test": []}
    provenance = []
    for split in (("train", "val") if phase == "development" else ("test",)):
        for path in sorted((destination / "old/统一GT" / split).rglob("*.json")):
            annotation = json.loads(path.read_text(encoding="utf-8-sig"))
            relative_image = Path(annotation["image_relative_path"])
            if (annotation["split"] != split or relative_image.parts[0] != split
                    or relative_image.drive or relative_image.root or ".." in relative_image.parts):
                raise ValueError("Old annotation points outside its supplied partition")
            counts = Counter(str(obj["class_id"]) for obj in annotation["objects"])
            if set(counts) - {str(i) for i in range(6)}:
                raise ValueError("Unknown class ID in unified annotation")
            if any(obj.get("ignore") for obj in annotation["objects"]):
                raise ValueError("Ignored annotation requires review")
            if {str(i): counts[str(i)] for i in range(6)} != annotation["actual_counts"]:
                raise ValueError("GT counts do not agree with object annotations")
            image = destination / "old" / relative_image
            add_row(rows, provenance, "valid" if split == "val" else split, image,
                    counts, "old_" + str(annotation["capture_condition"]), destination)
    for folder, split in ((("补强数据(训练）", "train"), ("验证集_不要训练", "valid"))
                          if phase == "development" else ()):
        for path in sorted((destination / "new" / folder).glob("*.json")):
            annotation = json.loads(path.read_text(encoding="utf-8-sig"))
            counts = Counter(LABELS[obj["label"]] for obj in annotation["shapes"])
            image = path.with_suffix(".jpg")
            add_row(rows, provenance, split, image, counts, "new_" + folder, destination)
    for split, samples in rows.items():
        samples.sort(key=lambda row: row["image_path"])
        with (destination / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "scene_id", "label"])
            writer.writeheader()
            writer.writerows(samples)
    write_json(destination / ("provenance.json" if phase == "development" else "test_provenance.json"), {
        "commit": COMMIT, "expected_counts": {str(i): 2 for i in range(1, 6)},
        "count_rule_status": "PROPOSED", "annotation_review": "NOT_CONFIRMED",
        "scene_independence": "UNVERIFIED",
        "scene_id_policy": "UNVERIFIED image IDs, not verified physical scene groups",
        "usage": "provided_train_valid" if phase == "development" else "provided_public_test",
        "support": {split: dict(Counter(row["label"] for row in samples))
                    for split, samples in rows.items()}, "sources": provenance})
    print(json.dumps({split: dict(Counter(row["label"] for row in samples))
                      for split, samples in rows.items()}), flush=True)


def add_row(rows, provenance, split, image, counts, scene, destination):
    actual = {str(i): counts[str(i)] for i in range(6)}
    label = "PASS" if actual["0"] == 0 and all(actual[str(i)] == 2 for i in range(1, 6)) else "FAIL"
    if split == "train" and label != "PASS":
        raise ValueError("One-class weight fitting accepts only PASS training samples")
    relative = os.path.relpath(image, destination)
    rows[split].append({"image_path": Path(relative).as_posix(),
                        "scene_id": "UNVERIFIED:" + image.stem, "label": label})
    provenance.append({"image_path": relative, "split": split, "counts": actual,
                       "label": label, "source_session": scene})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "test"), default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.phase == "test" and (args.freeze is None or not args.freeze.is_file()):
        parser.error("Test acquisition requires a completed freeze record")
    if args.phase == "test":
        validate_freeze(args.freeze)
        access = args.out / "test_source_access.json"
        args.out.mkdir(parents=True, exist_ok=True)
        if access.exists():
            if json.loads(access.read_text(encoding="utf-8"))["freeze_sha256"] != sha256(args.freeze):
                parser.error("Test sources were already accessed under another freeze")
        else:
            write_json(access, {"first_access": datetime.now(timezone.utc).isoformat(),
                                "freeze_sha256": sha256(args.freeze)})
    prepare(args.out, args.phase)


if __name__ == "__main__":
    main()
