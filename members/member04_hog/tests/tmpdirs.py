"""Workspace-local temporary directories (the DSH sandbox denies the system temp)."""
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

TMP_ROOT = Path(__file__).resolve().parents[3] / "outputs/member04_hog/final_v4/tmp"


@contextmanager
def workspace_temp():
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"tmp_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
