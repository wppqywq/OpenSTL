import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
from pathlib import Path as _Path
CURRENT_DIR = _Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import config as cfg


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_centers(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def velocity(c0: Tuple[float, float], c1: Tuple[float, float]) -> Tuple[float, float]:
    return float(c1[0] - c0[0]), float(c1[1] - c0[1])


def norm(vx: float, vy: float) -> float:
    return float(math.hypot(vx, vy))


def unit(vx: float, vy: float) -> Tuple[float, float]:
    n = norm(vx, vy)
    if n == 0.0:
        return 1.0, 0.0
    return vx / n, vy / n


def angle_between(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    ux, uy = unit(u[0], u[1])
    vx, vy = unit(v[0], v[1])
    dot = max(-1.0, min(1.0, ux * vx + uy * vy))
    return float(math.degrees(math.acos(dot)))


def expected_speed_center_speed(c: Tuple[float, float], v0: float, H: int, W: int, alpha: float) -> float:
    cx0, cy0 = W / 2.0, H / 2.0
    dist = math.hypot(c[0] - cx0, c[1] - cy0)
    rmax = math.hypot(cx0, cy0)
    return float(v0 * (1.0 + alpha * (dist / max(rmax, 1e-6))))


def expected_direction_center(c: Tuple[float, float], H: int, W: int) -> Tuple[float, float]:
    cx0, cy0 = W / 2.0, H / 2.0
    dx = cx0 - c[0]
    dy = cy0 - c[1]
    return unit(dx, dy)


def analyze_variant_dir(variant_dir: Path) -> Dict:
    seq_files = sorted(variant_dir.glob("seq*_centers.json"))
    results = {
        "variant": variant_dir.name,
        "num_sequences": 0,
        "avg_speed_error": None,
        "avg_direction_error": None,
    }
    speed_errors = []
    dir_errors = []

    for path in seq_files:
        data = load_centers(path)
        mode = data.get("mode")
        H = int(data.get("H"))
        W = int(data.get("W"))
        centers = data.get("centers", [])  # shape: T x num_digits x 2
        vmeta = data.get("variant", {})
        s_mult = float(vmeta.get("s", 1.0))
        alpha = float(vmeta.get("alpha", 0.0))

        v0 = float(cfg.MOTION_BASE_SPEED)
        for t in range(len(centers) - 1):
            c_t = centers[t]
            c_tp1 = centers[t + 1]
            for d in range(len(c_t)):
                v = velocity(tuple(c_t[d]), tuple(c_tp1[d]))
                vmag = norm(v[0], v[1])
                if mode == "baseline":
                    exp_speed = v0 * s_mult
                    speed_errors.append(abs(vmag - exp_speed))
                elif mode == "center_speed":
                    exp_speed = expected_speed_center_speed(tuple(c_t[d]), v0, H, W, alpha)
                    speed_errors.append(abs(vmag - exp_speed))
                elif mode == "center_direction":
                    exp_dir = expected_direction_center(tuple(c_t[d]), H, W)
                    dir_errors.append(angle_between(exp_dir, v))
                else:
                    pass
        results["num_sequences"] += 1

    if speed_errors:
        results["avg_speed_error"] = float(np.mean(speed_errors))
    if dir_errors:
        results["avg_direction_error"] = float(np.mean(dir_errors))
    return results


def main():
    variants_root = Path(cfg.VARIANTS_ROOT)
    out_dir = Path(cfg.OUTPUT_ROOT) / "eval"
    ensure_dir(str(out_dir))

    summaries = []
    for vdir in sorted(p for p in variants_root.iterdir() if p.is_dir()):
        summaries.append(analyze_variant_dir(vdir))

    out_path = out_dir / "variants_summary.json"
    with open(out_path, "w") as f:
        json.dump({"summaries": summaries}, f, indent=2)

    print(str(out_path))


if __name__ == "__main__":
    main()
