import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2

import sys
from pathlib import Path as _Path
CURRENT_DIR = _Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import config as cfg


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_manifest() -> Dict:
    manifest_path = Path(cfg.OUTPUT_ROOT) / "patch_detach" / "manifest.json"
    with open(manifest_path, "r") as f:
        return json.load(f)


def group_by_seq(manifest: Dict) -> Dict[int, Dict[int, Dict]]:
    grouped: Dict[int, Dict[int, Dict]] = {}
    for item in manifest.get("items", []):
        s = int(item["seq"])
        t = int(item["t"])
        grouped.setdefault(s, {})[t] = item
    return grouped


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return float(x + w / 2.0), float(y + h / 2.0)


def load_patch_image(out_dir: Path, patch_record: Dict) -> np.ndarray:
    p_path = out_dir / patch_record["patch_path"]
    img = cv2.imread(str(p_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(p_path))
    return img


def render_canvas(H: int, W: int, placements: List[Tuple[np.ndarray, Tuple[float, float]]]) -> np.ndarray:
    canvas = np.zeros((H, W), dtype=np.uint8)
    for patch_img, center in placements:
        ph, pw = patch_img.shape[:2]
        cx, cy = center
        x0 = int(round(cx - pw / 2.0))
        y0 = int(round(cy - ph / 2.0))
        # clipping
        x1 = max(0, x0)
        y1 = max(0, y0)
        x2 = min(W, x0 + pw)
        y2 = min(H, y0 + ph)
        if x1 >= x2 or y1 >= y2:
            continue
        px1 = x1 - x0
        py1 = y1 - y0
        px2 = px1 + (x2 - x1)
        py2 = py1 + (y2 - y1)
        roi = canvas[y1:y2, x1:x2]
        patch_roi = patch_img[py1:py2, px1:px2]
        # composite by max
        np.maximum(roi, patch_roi, out=roi)
    return canvas


def reflect_direction(cx: float, cy: float, vx: float, vy: float, ph: int, pw: int, H: int, W: int) -> Tuple[float, float]:
    # If next position would leave bounds, flip component
    nx = cx + vx
    ny = cy + vy
    half_w = pw / 2.0
    half_h = ph / 2.0
    if nx - half_w < 0 or nx + half_w > W:
        vx = -vx
    if ny - half_h < 0 or ny + half_h > H:
        vy = -vy
    return vx, vy


def unit(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n == 0:
        return 1.0, 0.0
    return vx / n, vy / n


def speed_field_center_speed(cx: float, cy: float, v0: float, H: int, W: int, alpha: float) -> float:
    cx0, cy0 = W / 2.0, H / 2.0
    dist = math.hypot(cx - cx0, cy - cy0)
    rmax = math.hypot(cx0, cy0)
    return float(v0 * (1.0 + alpha * (dist / max(rmax, 1e-6))))


def direction_field_center(cx: float, cy: float, H: int, W: int) -> Tuple[float, float]:
    cx0, cy0 = W / 2.0, H / 2.0
    dx = cx0 - cx
    dy = cy0 - cy
    return unit(dx, dy)


def infer_initial_velocity(rec_t0: Dict, rec_t1: Dict) -> Tuple[float, float]:
    # Use first patch centers difference to infer direction; fallback to (1,0)
    if not rec_t0 or not rec_t1:
        return 1.0, 0.0
    b0 = rec_t0.get("patches", [])
    b1 = rec_t1.get("patches", [])
    if len(b0) == 0 or len(b1) == 0:
        return 1.0, 0.0
    c0 = bbox_center(tuple(b0[0]["bbox"]))
    c1 = bbox_center(tuple(b1[0]["bbox"]))
    return c1[0] - c0[0], c1[1] - c0[1]


def run_variant(mode: str, variant_name, grouped: Dict[int, Dict[int, Dict]], T_out: int) -> None:
    H, W = cfg.CANVAS_SIZE[1], cfg.CANVAS_SIZE[0]
    variant_meta = variant_name if isinstance(variant_name, dict) else {"name": str(variant_name)}
    out_root = Path(cfg.VARIANTS_ROOT) / variant_meta["name"]
    ensure_dir(str(out_root))
    det_root = Path(cfg.OUTPUT_ROOT) / "patch_detach"

    # Choose sequences present in manifest (keys of grouped)
    seq_ids = sorted(grouped.keys())
    # For speed, limit to first S sequences in manifest
    S = min(cfg.MOTION_SAMPLE_SEQUENCES, len(seq_ids))
    seq_ids = seq_ids[:S]

    v0 = float(cfg.MOTION_BASE_SPEED)

    for s in seq_ids:
        recs = grouped[s]
        rec_t0 = recs.get(0)
        # pick a later record (6, then 1) for velocity inference
        rec_t1 = recs.get(6) or recs.get(1)
        vx0, vy0 = infer_initial_velocity(rec_t0, rec_t1)
        dvx, dvy = unit(vx0, vy0)

        # load patches at t0
        if rec_t0 is None:
            continue
        patches_meta = rec_t0.get("patches", [])
        if len(patches_meta) == 0:
            continue
        patch_imgs = [load_patch_image(det_root, p) for p in patches_meta]
        centers = [bbox_center(tuple(p["bbox"])) for p in patches_meta]
        dirs = [(dvx, dvy) for _ in patch_imgs]

        frames = []
        centers_log: List[List[Tuple[float, float]]] = []
        for t in range(T_out):
            placements = list(zip(patch_imgs, centers))
            frame = render_canvas(H, W, placements)
            frames.append(frame)
            centers_log.append([(float(cx), float(cy)) for (_, (cx, cy)) in placements])

            # update positions
            new_centers = []
            for idx, (patch_img, (cx, cy)) in enumerate(placements):
                ph, pw = patch_img.shape[:2]
                dx, dy = dirs[idx]

                # compute direction and speed
                if mode == "center_direction":
                    dx, dy = direction_field_center(cx, cy, H, W)
                speed = v0
                if mode == "center_speed":
                    speed = speed_field_center_speed(cx, cy, v0, H, W, variant_meta.get("alpha", 0.0))
                step_mult = variant_meta.get("s", 1.0)
                vx, vy = dx * speed * step_mult, dy * speed * step_mult

                # reflect if going out of bounds
                vx, vy = reflect_direction(cx, cy, vx, vy, ph, pw, H, W)
                new_centers.append((cx + vx, cy + vy))
                dirs[idx] = unit(vx, vy) if mode == "baseline" else (dx, dy)
            centers = new_centers

        arr = np.stack(frames, axis=0)  # (T, H, W)
        np.save(out_root / f"seq{s:04d}.npy", arr)
        meta = {
            "mode": mode,
            "variant": variant_meta,
            "H": H,
            "W": W,
            "centers": centers_log,
        }
        with open(out_root / f"seq{s:04d}_centers.json", "w") as f:
            json.dump(meta, f)


def main():
    manifest = load_manifest()
    grouped = group_by_seq(manifest)

    # Keep generation lightweight for now
    T_out = 20

    # Run a minimal set to verify pipeline
    run_variant(mode="baseline", variant_name={"name": "baseline", "s": 1.0}, grouped=grouped, T_out=T_out)
    run_variant(mode="baseline", variant_name={"name": "fast_s1.5", "s": 1.5}, grouped=grouped, T_out=T_out)
    for alpha in cfg.CENTER_SPEED_ALPHAS:
        run_variant(mode="center_speed", variant_name={"name": f"center_speed_a{alpha}", "alpha": float(alpha)}, grouped=grouped, T_out=T_out)
    run_variant(mode="center_direction", variant_name={"name": "center_direction"}, grouped=grouped, T_out=T_out)

    print(str(Path(cfg.VARIANTS_ROOT)))


if __name__ == "__main__":
    main()
