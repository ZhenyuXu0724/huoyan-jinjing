"""Fixed, shared definitions for the reproducible autoencoder comparisons."""
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SEEDS = (42, 43, 44)
GEOMETRY = json.loads((CONFIG_DIR / "groups/geometry.json").read_text(encoding="utf-8"))
PHOTOMETRIC = json.loads((CONFIG_DIR / "groups/photometric.json").read_text(encoding="utf-8"))
