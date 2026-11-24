"""Build or verify a portable v4 delivery package without raw images."""
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
    from .final_v4.package import build as build_final_v4

    return build_final_v4(Path(destination))


def verify(target):
    target = Path(target)
    if target.is_dir():
        expected = json.loads((target / CHECKSUMS).read_text(encoding="utf-8"))
        names = {path.relative_to(target).as_posix() for path in target.rglob("*")
                 if path.is_file() and path.name != CHECKSUMS
                 and "__pycache__" not in path.parts
                 and path.suffix.lower() not in (".pyc", ".pyo")}
        missing = set(expected) - names
        if missing:
            raise ValueError(f"Missing delivery files: {sorted(missing)[:5]}")
        for name, checksum in expected.items():
            if digest((target / name).read_bytes()) != checksum:
                raise ValueError(f"Delivery checksum mismatch: {name}")
        ignored = names - set(expected)
        if ignored:
            print(f"Ignored {len(ignored)} runtime-generated files during directory verify")
    else:
        with zipfile.ZipFile(target) as archive:
            expected = json.loads(archive.read(CHECKSUMS))
            if set(archive.namelist()) != set(expected) | {CHECKSUMS}:
                raise ValueError("Unexpected or missing ZIP entries")
            for name, checksum in expected.items():
                if digest(archive.read(name)) != checksum:
                    raise ValueError(f"Delivery checksum mismatch: {name}")
    print(f"Verified {len(expected)} delivery files")
    return len(expected)


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
