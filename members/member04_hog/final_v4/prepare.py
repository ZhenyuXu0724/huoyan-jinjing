"""Restore the frozen final_training_v4 tree into member04's own ignored data copy.

Development scope restores only the 283 train/val originals and 13 metadata files.
Full scope additionally restores test, optional training augmentations and the
dedicated test per-image table, and requires a completed freeze.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import (
    AUDIT, CONFIG, DATA, DATASET_PATH, DATASET_TREE, EXPECTED_COUNTS, FULL_ONLY_FILES,
    INTRO_COMMIT, METADATA_FILES, OUT, REPOSITORY, ROOT, SOURCE_COMMIT, SOURCE_LOCK,
    SUPPLIER, ensure, eol_variant, git, git_blob_sha1, git_eol_map, git_text, git_tree,
    now, read, read_csv, record_stage, sha256, sha256_bytes, update_state, write,
)

DEV_FILES = 862
FULL_FILES = 1260
DATASET_BYTES = 1789463157
METADATA = frozenset(METADATA_FILES)


def audit_reference():
    return read(ROOT / "members/member04_hog/plans/v4_rebuild_20261124/dataset_audit.json")


def canonical_tree_hash(entries):
    payload = b"".join(
        (f"{entry['path']}\0{entry['sha']}\0{entry['size']}\n").encode("utf-8")
        for entry in sorted(entries, key=lambda item: item["path"]))
    return sha256_bytes(payload)


def dataset_eol(prefix=DATASET_PATH):
    mapping = git_eol_map(prefix)
    result = {}
    for path, flags in mapping.items():
        if path.startswith(prefix + "/"):
            result[path[len(prefix) + 1:]] = flags
    return result


def select_scope(entries, scope):
    if scope == "full":
        return list(entries)
    selected = []
    for entry in entries:
        path = entry["path"]
        inside_media = (path.startswith("images/train/") or path.startswith("images/val/")
                        or path.startswith("labels/train/") or path.startswith("labels/val/"))
        if inside_media or path in METADATA:
            selected.append(entry)
    return selected


def build_lock(commit):
    if git_text("rev-parse", f"{commit}^{{commit}}") != commit:
        raise RuntimeError(f"source commit is not available locally: {commit}")
    tree = git_text("rev-parse", f"{commit}:{DATASET_PATH}")
    if tree != DATASET_TREE:
        raise RuntimeError(f"dataset tree changed: {tree} != {DATASET_TREE}")
    entries = git_tree(commit, DATASET_PATH)
    if len(entries) != FULL_FILES:
        raise RuntimeError(f"unexpected dataset file count: {len(entries)}")
    total = sum(entry["size"] for entry in entries)
    if total != DATASET_BYTES:
        raise RuntimeError(f"unexpected dataset bytes: {total}")
    eol = dataset_eol()
    counts = {"lf_to_crlf": 0, "crlf": 0, "binary": 0, "other": 0}
    for flags in eol.values():
        if "i/lf" in flags and "w/crlf" in flags:
            counts["lf_to_crlf"] += 1
        elif "i/crlf" in flags:
            counts["crlf"] += 1
        elif "i/-text" in flags:
            counts["binary"] += 1
        else:
            counts["other"] += 1
    reference = audit_reference()
    locked_metadata = {}
    self_locked = []
    for name in METADATA_FILES:
        raw = (SUPPLIER / name).read_bytes()
        normalised = raw.replace(b"\r\n", b"\n")
        if name in reference["metadata_files"]:
            expected = reference["metadata_files"][name]["sha256"]
            if sha256_bytes(normalised) != expected:
                raise RuntimeError(f"metadata content no longer matches the locked audit: {name}")
            locked_metadata[name] = expected
        else:
            locked_metadata[name] = sha256_bytes(normalised)
            self_locked.append(name)
    dev = select_scope(entries, "development")
    if len(dev) != DEV_FILES:
        raise RuntimeError(f"unexpected development file count: {len(dev)}")
    lock = {
        "schema_version": 1,
        "created_at": now(),
        "repository": REPOSITORY,
        "source_commit": commit,
        "dataset_path": DATASET_PATH,
        "dataset_tree": tree,
        "introduction_commit": INTRO_COMMIT,
        "tree_listing": {"entries": len(entries), "bytes": total,
                         "sha256": canonical_tree_hash(entries)},
        "scopes": {
            "development": {"files": len(dev), "bytes": sum(item["size"] for item in dev)},
            "full": {"files": len(entries), "bytes": total},
        },
        "metadata_files": list(METADATA_FILES),
        "metadata_sha256": locked_metadata,
        "metadata_sha256_self_locked": self_locked,
        "eol": counts,
        "notes": [
            "Git stores LF for files checked out as CRLF on Windows; provider split_manifest "
            "SHA256 values lock the CRLF working bytes for images/JSON. Both the Git blob "
            "(after CRLF->LF normalisation) and the provider SHA256 are verified.",
            "Development scope intentionally excludes test media, optional training "
            "augmentations and the dedicated test per-image table.",
        ],
    }
    write(SOURCE_LOCK, lock)
    return lock


def load_lock(commit):
    if SOURCE_LOCK.exists():
        lock = read(SOURCE_LOCK)
        if lock["source_commit"] != commit:
            raise RuntimeError(f"source lock pins {lock['source_commit']}, not {commit}")
        if lock["dataset_tree"] != DATASET_TREE:
            raise RuntimeError("source lock has an unexpected dataset tree")
        return lock
    return build_lock(commit)


def tree_cache(entries, eol, lock):
    payload = {
        "source_commit": lock["source_commit"],
        "dataset_tree": lock["dataset_tree"],
        "entries": [{**entry, "eol": eol.get(entry["path"], [])} for entry in entries],
        "eol_counts": lock["eol"],
        "canonical_sha256": canonical_tree_hash(entries),
    }
    ensure(AUDIT)
    write(AUDIT / "source_tree.json", payload)
    return payload


def split_rows():
    rows = read_csv(SUPPLIER / "split_manifest.csv")
    images, jsons = {}, {}
    for row in rows:
        if row.get("dest_image"):
            images[row["dest_image"]] = row
        if row.get("dest_json"):
            jsons[row["dest_json"]] = row
    return images, jsons


def classify(path):
    if path in METADATA:
        return "metadata"
    if path in FULL_ONLY_FILES:
        return "full_only"
    if path.endswith(".json"):
        return "json"
    if path.startswith("labels/") and path.endswith(".txt"):
        return "yolo"
    if path.startswith("images/"):
        return "image"
    return "other"


def copy_scope(scope, lock, check_only=False):
    entries = git_tree(lock["source_commit"], DATASET_PATH)
    selected = select_scope(entries, scope)
    eol = dataset_eol()
    images, jsons = split_rows()
    missing = []
    restored = skipped = 0
    physical_bytes = 0
    changed = []
    details = []
    for index, entry in enumerate(selected, 1):
        path = entry["path"]
        source = SUPPLIER / path
        if not source.is_file():
            missing.append(path)
            continue
        raw = source.read_bytes()
        flags = eol.get(path)
        if flags is None:
            raise RuntimeError(f"missing Git eol information for {path}")
        candidate = eol_variant(raw, flags)
        if (git_blob_sha1(candidate) != entry["sha"] or len(candidate) != entry["size"]):
            if (git_blob_sha1(raw) == entry["sha"] and len(raw) == entry["size"]):
                candidate = raw
            else:
                raise RuntimeError(f"Git blob verification failed for {path}")
        raw_digest = sha256_bytes(raw)
        normalised_digest = sha256_bytes(raw.replace(b"\r\n", b"\n"))
        kind = classify(path)
        details.append({"path": path, "sha": entry["sha"], "size": entry["size"],
                        "eol": flags, "kind": kind})
        if kind == "image" and path in images:
            row = images[path]
            if len(raw) != int(row["image_bytes"]) or raw_digest != row["image_sha256"]:
                raise RuntimeError(f"provider image SHA/size mismatch: {path}")
        elif kind == "json" and path in jsons:
            row = jsons[path]
            if len(raw) != int(row["json_bytes"]) or raw_digest != row["json_sha256"]:
                raise RuntimeError(f"provider JSON SHA/size mismatch: {path}")
        elif kind == "metadata":
            if normalised_digest != lock["metadata_sha256"][path]:
                raise RuntimeError(f"metadata SHA256 mismatch: {path}")
        target = DATA / path
        physical_bytes += len(raw)
        if target.exists():
            if sha256(target) == raw_digest:
                skipped += 1
                continue
            changed.append(path)
            continue
        if check_only:
            missing.append(path)
            continue
        ensure(target.parent)
        temporary = target.with_name(target.name + ".part")
        temporary.write_bytes(raw)
        if sha256(temporary) != raw_digest:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"copy changed bytes: {path}")
        temporary.replace(target)
        restored += 1
        if index % 40 == 0:
            print(f"Prepared {index}/{len(selected)} files ({scope})", flush=True)
    if changed:
        raise RuntimeError(f"existing data files differ from the frozen source: {changed[:5]}")
    if missing:
        raise RuntimeError(f"missing source files in {scope} scope: {missing[:5]}")
    return {"scope": scope, "files": len(selected), "restored": restored,
            "skipped": skipped, "bytes": physical_bytes, "entries": details}


def full_scope_gate():
    freeze_path = OUT / "freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("full scope requires outputs/member04_hog/final_v4/freeze.json")
    freeze = read(freeze_path)
    if freeze.get("test_accessed"):
        raise RuntimeError("freeze already records test access; do not restore twice")
    freeze_sha = sha256(freeze_path)
    access_path = AUDIT / "test_access.json"
    access = read(access_path) if access_path.exists() else {"events": []}
    if not access.get("events"):
        access = {
            "schema_version": 1,
            "freeze_sha256": freeze_sha,
            "events": [{
                "type": "restore_full_scope",
                "at": now(),
                "freeze_sha256": freeze_sha,
                "scope": "full",
                "decoded_test_media": False,
                "note": "test/augmentation bytes restored after freeze; decode happens in audit --scope full",
            }],
        }
        write(access_path, access)
    return freeze, freeze_sha


def run(scope, commit, check_only=False):
    lock = load_lock(commit)
    if scope == "full":
        full_scope_gate()
    tree = git_tree(lock["source_commit"], DATASET_PATH)
    cache = tree_cache(tree, dataset_eol(), lock)
    result = copy_scope(scope, lock, check_only=check_only)
    entries = result.pop("entries")
    write(AUDIT / f"file_list_{scope}.json", {
        "scope": scope,
        "source_commit": lock["source_commit"],
        "dataset_tree": lock["dataset_tree"],
        "files": len(entries),
        "tree_bytes": sum(entry["size"] for entry in entries),
        "entries": entries,
    })
    result.update({"source_commit": lock["source_commit"],
                   "dataset_tree": lock["dataset_tree"],
                   "tree_cache_sha256": sha256(AUDIT / "source_tree.json"),
                   "file_list_sha256": sha256(AUDIT / f"file_list_{scope}.json"),
                   "tree_cache_canonical_sha256": cache["canonical_sha256"],
                   "checked_at": now(), "check_only": check_only})
    write(AUDIT / f"source_restore_{scope}.json", result)
    if scope == "full":
        update_state(test_accessed=True, test_restored_at=now(), run_scope="full")
    record_stage(f"prepare_{scope}", "completed",
                 f"restored {result['files']} files for {scope} scope",
                 files=result["files"], restored=result["restored"],
                 skipped=result["skipped"], source_commit=lock["source_commit"])
    next_command = ("python -m members.member04_hog.final_v4.audit --scope development"
                    if scope == "development"
                    else "python -m members.member04_hog.final_v4.audit --scope full")
    update_state(next_command=next_command)
    print(f"Prepared {scope}: {result['files']} files, restored {result['restored']}, "
          f"skipped {result['skipped']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("development", "full"), default="development")
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args.scope, args.source_commit, check_only=args.check_only)
    except Exception as error:
        record_stage(f"prepare_{args.scope}", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
