"""Build the portable v4 delivery ZIP with checksums and no raw images."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .common import (
    DATA, EVALUATION, EXPERIMENTS, METHOD_SUFFIX, MODULE, OUT, PROTOCOL, REFINEMENT,
    ROOT, WEIGHTS, ensure, read, record_stage, sha256, update_state, write,
)

DELIVERY_NAME = "Baseline1_HOGMLP_v4_FinalDataset_Rebuild"
CHECKSUMS = "DELIVERY_SHA256.json"
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".err", ".png", ".jpg", ".jpeg", ".zip", ".tmp"}
EXCLUDE_DIRS = {"__pycache__", "pairs", "failed_attempts", "plans", ".venv"}


def module_files():
    paths = []
    for path in (MODULE).rglob("*"):
        if not path.is_file() or set(path.parts) & EXCLUDE_DIRS:
            continue
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        paths.append(path)
    return paths


def output_files():
    paths = []
    for path in OUT.rglob("*"):
        if not path.is_file() or set(path.parts) & EXCLUDE_DIRS:
            continue
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        paths.append(path)
    paths = [path for path in paths if not path.name.startswith("train_")]
    return paths


def weight_files():
    return [path for path in WEIGHTS.rglob("*.joblib") if path.is_file()]


def data_files():
    root = DATA
    selected = [root / name for name in ("README.md", "classes.json", "data.yaml",
                                         "verify_dataset.py", "split_manifest.csv",
                                         "原图明细.csv")]
    selected += [path for path in (root / "manifests").rglob("*.csv")]
    return [path for path in selected if path.is_file()]


def inclusion():
    mapping = {}
    for path in module_files():
        mapping[path.relative_to(ROOT).as_posix()] = path
    for path in (ROOT / "shared/contracts.py", ROOT / "shared/catalogue.json"):
        mapping[path.relative_to(ROOT).as_posix()] = path
    for path in weight_files():
        mapping[path.relative_to(ROOT).as_posix()] = path
    for path in output_files():
        mapping[path.relative_to(ROOT).as_posix()] = path
    for path in data_files():
        mapping[path.relative_to(ROOT).as_posix()] = path
    return mapping


def build(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    verify_protocols()
    mapping = inclusion()
    readme = (
        "# Baseline1_HOGMLP_v4_FinalDataset_Rebuild\n\n"
        "先读 `members/member04_hog/README.md` 与 `progress.md`。\n\n"
        "本包含 v4 新代码、配置、75 次运行记录、84 份检查点、三个代表模型与验证证据；"
        "不含原始图片、虚拟环境、旧版模型或旧交付包。\n\n"
        "解压后在根目录执行（Python 3.11）：\n\n"
        "```powershell\n"
        "py -3.11 -m venv .venv\n"
        ".\\.venv\\Scripts\\python.exe -m pip install -r members/member04_hog/requirements.txt\n"
        ".\\.venv\\Scripts\\python.exe -m members.member04_hog.delivery --verify .\n"
        ".\\.venv\\Scripts\\python.exe -m unittest discover -s members/member04_hog/tests -v\n"
        "```\n\n"
        "原始数据需按固定 commit 重新恢复；ZIP 保留打包时快照，最新交接状态以 Git 为准。\n"
    ).encode("utf-8")
    payloads = {"DELIVERY_README.md": readme}
    checksums = {}
    ensure(destination.parent)
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("wb") as stream, zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(payloads):
            archive.writestr(name, payloads[name])
            checksums[name] = hashlib.sha256(payloads[name]).hexdigest()
        for number, name in enumerate(sorted(mapping), 1):
            payload = mapping[name].read_bytes()
            archive.writestr(name, payload)
            checksums[name] = hashlib.sha256(payload).hexdigest()
            if number % 100 == 0:
                print(f"Packaged {number}/{len(mapping)} files", flush=True)
        archive.writestr(CHECKSUMS, (json.dumps(checksums, ensure_ascii=False, indent=2)
                                     + "\n").encode("utf-8"))
    temporary.replace(destination)
    digest = sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        digest + "  " + destination.name + "\n", encoding="utf-8")
    from ..delivery import verify

    verified = verify(destination)
    result = {"path": str(destination), "bytes": destination.stat().st_size,
              "sha256": digest, "files": len(checksums), "verified_entries": verified}
    record_stage("package", "completed", f"{len(checksums)} files, {destination.name}")
    update_state(next_command="python -m members.member04_hog.delivery --verify "
                              f"../delivery/{destination.name}")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def verify_protocols():
    from .common import verify_protocol

    verify_protocol()
    freeze = read(OUT / "freeze.json")
    state = read(OUT / "state.json")
    audit = read(OUT / "audit/dataset_audit_full.json")
    if audit.get("status") != "PASS" or not state.get("test_accessed"):
        raise RuntimeError("packaging requires the post-freeze full evaluation")
    if freeze.get("test_accessed"):
        raise RuntimeError("freeze must predate test access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        build(args.out)
    except Exception as error:
        record_stage("package", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
