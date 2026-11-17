"""Build or verify a portable code, model and evidence package (without raw images)."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKSUMS = "DELIVERY_SHA256.json"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def build(destination):
    from .evaluate import validate_freeze

    frozen = ROOT / "outputs/member04_hog/assessment/freeze.json"
    record, _ = validate_freeze(frozen)
    results = ROOT / "outputs/member04_hog/experiments"
    if not (results / "provided_test_summary.json").is_file():
        raise ValueError("Complete the frozen provided-test report before packaging")
    if not (frozen.parent / "data_audit.json").is_file():
        raise ValueError("Complete the supplied-data audit before packaging")
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    if digest((ROOT / "weights/member04_hog/baseline1.joblib").read_bytes()) != selection["model_sha256"]:
        raise ValueError("Default model does not match the validation selection")
    paths = []
    for directory in (ROOT / "members/member04_hog", results,
                      ROOT / "outputs/member04_hog/assessment",
                      ROOT / "outputs/member04_hog/handoff"):
        paths.extend(p for p in directory.rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts
                     and p.suffix.lower() not in (".pyc", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    paths.extend(ROOT / p for p in ("shared/contracts.py", "shared/catalogue.json",
                                  "weights/member04_hog/baseline1.joblib"))
    paths.extend((frozen.parent / entry["path"]).resolve() for entry in record["models"])
    data = ROOT / "data/member04_hog/dataset"
    paths.extend(p for p in data.iterdir() if p.is_file() and p.suffix in (".csv", ".json"))
    payloads = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(set(paths))}
    payloads["DELIVERY_README.md"] = (
        "# 成员 4 基线交付包\n\n"
        "先读 members/member04_hog/README.md 和 progress.md。\n\n"
        "本包含代码、默认模型、冻结对照模型及完整运行证据，不包含原始图片和虚拟环境。"
        "图片按固定源索引重新下载；原始完整数据保留在交付机器的 data/member04_hog/dataset/。\n\n"
        "在解压根目录执行（Python 3.11）：\n\n"
        "```powershell\n"
        "py -3.11 -m members.member04_hog.delivery --verify .\n"
        "py -3.11 -m venv .venv\n"
        ".\\.venv\\Scripts\\python.exe -m pip install -r members/member04_hog/requirements.txt\n"
        ".\\.venv\\Scripts\\python.exe -m unittest discover -s members/member04_hog/tests -v\n"
        "```\n\n"
        "校验仅需要标准库。新项目主仓库的公共入口需由组长整合，"
        "本包自带成员 4 的单图和批量入口。提供分区评测不等同于独立封存测试。\n"
    ).encode("utf-8")
    payloads[CHECKSUMS] = (json.dumps({k: digest(v) for k, v in sorted(payloads.items())},
                                   ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in sorted(payloads.items()):
                archive.writestr(name, payload)
    verify(destination)
    checksum = digest(destination.read_bytes())
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        checksum + "  " + destination.name + "\n", encoding="utf-8")
    print(f"Packaged {len(payloads)} files; {destination.stat().st_size} bytes")


def verify(target):
    if target.is_dir():
        expected = json.loads((target / CHECKSUMS).read_text(encoding="utf-8"))
        read = lambda name: (target / name).read_bytes()
        for name, checksum in expected.items():
            if digest(read(name)) != checksum:
                raise ValueError(f"Delivery checksum mismatch: {name}")
    else:
        with zipfile.ZipFile(target) as archive:
            expected = json.loads(archive.read(CHECKSUMS))
            if set(archive.namelist()) != set(expected) | {CHECKSUMS}:
                raise ValueError("Unexpected or missing ZIP entries")
            for name, checksum in expected.items():
                if digest(archive.read(name)) != checksum:
                    raise ValueError(f"Delivery checksum mismatch: {name}")
    print(f"Verified {len(expected)} delivery files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        verify(args.verify)
    else:
        build(args.out)


if __name__ == "__main__":
    main()
