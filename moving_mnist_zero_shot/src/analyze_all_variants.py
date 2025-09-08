import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from analyze_variants import analyze_variant_dir

# Analyze all variants including new ones
variants_root = Path(cfg.VARIANTS_ROOT)
all_variants = sorted([d for d in variants_root.iterdir() if d.is_dir()])

results = {}
for variant_dir in all_variants:
    variant_name = variant_dir.name
    print(f"Analyzing {variant_name}...")
    
    summary = analyze_variant_dir(variant_dir)
    results[variant_name] = summary
    
    if summary["avg_speed_error"] is not None:
        print(f"  Speed error: {summary['avg_speed_error']:.2e}")
    if summary["avg_direction_error"] is not None:
        print(f"  Direction error: {summary['avg_direction_error']:.2e}")

# Save updated results
out_path = Path(cfg.OUTPUT_ROOT) / "eval" / "all_variants_analysis.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {out_path}")

# Summary table
print("\n=== Speed Analysis ===")
for name, data in results.items():
    if "fast" in name or "baseline" in name or "center_speed" in name:
        err = data.get("avg_speed_error")
        if err is not None:
            print(f"{name:20s}: {err:.2e}")

print("\n=== Direction Analysis ===")
for name, data in results.items():
    if "center_direction" in name:
        err = data.get("avg_direction_error")
        if err is not None:
            print(f"{name:20s}: {err:.2e}")
