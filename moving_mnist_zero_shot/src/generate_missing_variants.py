import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from motion_controller import run_variant

# Generate missing fast variants (s=2.0, s=3.0)
missing_variants = [
    {"name": "fast_s2.0", "mode": "fast", "multiplier": 2.0},
    {"name": "fast_s3.0", "mode": "fast", "multiplier": 3.0},
]

# Load grouped patches
det_root = Path(cfg.OUTPUT_ROOT) / "patch_detach"
manifest_path = det_root / "manifest.json"
with open(manifest_path, "r") as f:
    manifest = json.load(f)

grouped = {}
for entry in manifest["items"]:  # Changed from "entries" to "items"
    seq_idx = entry["seq"]  # Changed from "seq_idx" to "seq"
    t = entry["t"]
    if seq_idx not in grouped:
        grouped[seq_idx] = {}
    grouped[seq_idx][t] = entry

# Generate missing variants
for variant in missing_variants:
    print(f"Generating {variant['name']}...")
    run_variant(variant["mode"], variant, grouped, cfg.MOTION_OUTPUT_LENGTH)
    print(f"  Done: moving_mnist_zero_shot/data_variants/{variant['name']}/")

print("All missing variants generated.")
