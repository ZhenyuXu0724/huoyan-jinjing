"""Verify v4 data, then optionally write member05's machine-local dataset YAML."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(all_files=False, root=ROOT):
    rows = read_csv(root / "split_manifest.csv")
    originals = {r["filename"]: r for r in rows if r["tier"] == "original"}
    check(len(originals) == 399 and len({r["filename"] for r in rows}) == len(rows) == 609, "incomplete inventory")
    expected = {"train": (166, 83), "val": (20, 14), "test": (15, 15), "unused": (0, 86)}
    types = defaultdict(set)
    scenes, owners = {}, {}
    for split, counts in expected.items():
        found = Counter(r["label"] for r in originals.values() if r["split"] == split)
        check((found["PASS"], found["FAIL"]) == counts, f"class totals changed in {split}")
    names = json.loads((root / "classes.json").read_text(encoding="utf-8"))["classes"]
    ids = {r["labelme_label"]: r["id"] for r in names}
    usage_counts = Counter()
    active = []
    for row in rows:
        parent = originals[row["parent"]]
        split, name = row["split"], row["filename"]
        for key in ("split", "scene_group", "label", "batch", "N", "B", "S", "W", "H"):
            check(row[key] == parent[key], f"parent mismatch for {name}: {key}")
        expected_usage = ("unused" if split == "unused" else "primary" if row["tier"] == "original"
                          else "optional_train_aug" if split == "train" else "excluded_eval_aug")
        check(row["usage"] == expected_usage, f"wrong role for {name}")
        usage_counts[expected_usage] += 1
        if split != "unused":
            for key in ("scene_group", "source_scene_group"):
                check(scenes.setdefault((key, row[key]), split) == split, f"shared scene: {name}")
            check(owners.setdefault(row["image_sha256"], split) == split, f"shared image: {name}")
            types[split].update(k for k in "NBSWH" if int(row[k]))
        if expected_usage not in ("primary", "optional_train_aug"):
            check(not any(row[k] for k in ("dest_image","dest_json","yolo_label")), f"unexpected output for {name}")
            continue
        folder = split if row["tier"] == "original" else "train_augmentations"
        relative = Path(folder, row["label"], name)
        check(row["dest_image"] == (Path("images")/relative).as_posix(), f"wrong image path for {name}")
        check(row["dest_json"] == (Path("images")/relative.with_suffix(".json")).as_posix(), f"wrong annotation path for {name}")
        check(row["yolo_label"] == (Path("labels")/relative.with_suffix(".txt")).as_posix(), f"wrong YOLO path for {name}")
        if all_files or (row["tier"] == "original" and split in ("train", "val")):
            active.append(row)
    check(usage_counts == {"primary":313,"optional_train_aug":102,"excluded_eval_aug":63,"unused":131}, "role totals")
    check(types["test"] == set("SHNW") and types["test"] - (types["train"] | types["val"]) == set("SH"), "held-out types")
    for split in ("train", "val", "test"):
        base = [r for r in originals.values() if r["split"] == split]
        for model in ("hog", "resnet"):
            path = root / "manifests" / model / f"{split}.csv"
            expected_rows = {(r["dest_image"],r["label"],r["scene_group"]) for r in base}
            model_rows = read_csv(path)
            if model == "resnet":
                check(all(r["split"] == split and r["data_kind"] == "REAL" for r in model_rows), f"invalid ResNet role: {split}")
            expected_refs = {("../../"+p,l,g) for p,l,g in expected_rows}
            check(len(model_rows) == len(base) and {(r["image_path"],r["label"],r["scene_id"]) for r in model_rows} == expected_refs, f"{model} manifest: {split}")
    pass_only = read_csv(root/"manifests/hog/train_pass.csv")
    train = read_csv(root/"manifests/hog/train.csv")
    check(pass_only == [r for r in train if r["label"] == "PASS"], "HOG single-class manifest mismatch")
    for row in active:
        for kind in ("image","json"):
            file = root / row["dest_"+kind]
            check(file.stat().st_size == int(row[kind+"_bytes"]) and file_digest(file) == row[kind+"_sha256"], f"content changed: {file}")
        ann = json.loads((root / row["dest_json"]).read_text(encoding="utf-8-sig"))
        w,h = ann["imageWidth"],ann["imageHeight"]
        counts, expected_boxes = Counter(), []
        for shape in ann["shapes"]:
            check(shape["shape_type"] == "rectangle", row["filename"])
            (x1,y1),(x2,y2) = shape["points"]
            check(0 <= min(x1,x2) < max(x1,x2) <= w and 0 <= min(y1,y2) < max(y1,y2) <= h, row["filename"])
            cid = ids[shape["label"]]
            counts[cid] += 1
            expected_boxes.append([cid, (x1+x2)/(2*w), (y1+y2)/(2*h), abs(x2-x1)/w, abs(y2-y1)/h])
        labels = (root/row["yolo_label"]).read_text(encoding="utf-8").splitlines()
        check(len(labels) == len(expected_boxes), f"box count: {row['filename']}")
        for label, box in zip(labels, expected_boxes):
            fields = label.split()
            check(len(fields)==5 and int(fields[0])==box[0] and all(math.isclose(float(v), b, abs_tol=1e-9, rel_tol=0) for v,b in zip(fields[1:],box[1:])), f"box mismatch: {row['filename']}")
        check(counts[0] == sum(int(row[k]) for k in "NBSWH") and [counts[i] for i in range(1,6)] == list(map(int,row["target_counts"].split("|"))), f"part counts: {row['filename']}")
        passes = counts[0] == 0 and all(counts[i] == 2 for i in range(1,6))
        check(passes == (row["label"]=="PASS"), f"image verdict: {row['filename']}")
    print(f"Verified the v4 split and {len(active)} physical image/LabelMe/YOLO triples.")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="also read test and optional training augmentation files")
    parser.add_argument("--yolo-yaml", type=Path, help="write a new YAML with this checkout's absolute dataset root")
    args = parser.parse_args()
    verify(args.all)
    if args.yolo_yaml:
        with args.yolo_yaml.open("x", encoding="utf-8") as out:
            out.write("path: " + json.dumps(ROOT.as_posix(), ensure_ascii=False) + "\n" + (ROOT/"data.yaml").read_text(encoding="utf-8"))
