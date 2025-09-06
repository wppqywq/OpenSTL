import json
from pathlib import Path
from typing import List, Tuple

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


def to_gray_2d(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return frame[..., 0]
    raise ValueError(f"Unexpected frame shape: {frame.shape}")


def threshold_frame(frame: np.ndarray, use_otsu: bool, T: int) -> np.ndarray:
    src = frame.astype(np.uint8)
    if use_otsu:
        _, th = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(src, T, 255, cv2.THRESH_BINARY)
    return th


def morph_open(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def largest_components(binary: np.ndarray, max_components: int, min_area: int) -> List[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            comp_mask = (labels == label).astype(np.uint8) * 255
            components.append((area, comp_mask))
    components.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in components[:max_components]]


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)


def extract_patch(frame: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    x, y, w, h = bbox_from_mask(mask)
    if w == 0 or h == 0:
        return None, None, (0, 0, 0, 0)
    sub_img = frame[y:y+h, x:x+w].copy()
    sub_mask = mask[y:y+h, x:x+w].copy()
    sub_img[sub_mask == 0] = 0
    return sub_img, sub_mask, (x, y, w, h)


def overlay_debug(frame: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    vis = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    colors = [(0,255,0), (0,0,255)]
    for idx, m in enumerate(masks):
        contours, _ = cv2.findContours((m>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, colors[idx % len(colors)], 1)
    return vis


def load_array() -> np.ndarray:
    root = Path(cfg.DATA_ROOT)
    candidates = []
    for p in root.glob("**/*.npy"):
        if "mnist" in p.name.lower():
            candidates.append(p)
    if not candidates:
        candidates = list(root.glob("**/*.npy"))
    if not candidates:
        raise FileNotFoundError("No .npy files under data root")
    preferred = [p for p in candidates if "test" in p.name.lower()]
    path = preferred[0] if preferred else candidates[0]
    arr = np.load(path)
    return arr


def get_accessor(arr: np.ndarray):
    if arr.ndim == 5:
        T_first = arr.shape[0] < arr.shape[1]
        if T_first:
            T, N = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                return to_gray_2d(arr[t, seq_idx])
            return get_frame, N, T
        else:
            N, T = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                return to_gray_2d(arr[seq_idx, t])
            return get_frame, N, T
    elif arr.ndim == 4:
        T_first = arr.shape[0] < arr.shape[1]
        if T_first:
            T, N = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                return arr[t, seq_idx]
            return get_frame, N, T
        else:
            N, T = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                return arr[seq_idx, t]
            return get_frame, N, T
    elif arr.ndim == 3:
        if arr.shape[0] >= 10 and arr.shape[1] == 64 and arr.shape[2] == 64:
            T = arr.shape[0]
            def get_frame(seq_idx, t):
                return arr[t]
            return get_frame, 1, T
        else:
            N = arr.shape[0]
            def get_frame(seq_idx, t):
                return arr[seq_idx]
            return get_frame, N, 1
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


def main():
    out_dir = Path(cfg.OUTPUT_ROOT) / "patch_detach"
    ensure_dir(str(out_dir))

    arr = load_array()
    get_frame, N, T = get_accessor(arr)

    S = min(cfg.PATCH_SAMPLE_SEQUENCES, N)
    K = min(cfg.PATCH_FRAMES_PER_SEQUENCE, T)
    seq_indices = list(range(0, N, max(1, N // S)))[:S]
    frame_indices = list(range(0, T, max(1, T // K)))[:K]

    manifest = []

    for s in seq_indices:
        for t in frame_indices:
            frame = get_frame(s, t)
            gray = to_gray_2d(frame)
            th = threshold_frame(gray, cfg.PATCH_USE_OTSU, cfg.PATCH_THRESHOLD)
            if cfg.PATCH_MORPH_OPEN:
                th = morph_open(th)
            comps = largest_components(th, cfg.PATCH_MAX_COMPONENTS, cfg.PATCH_MIN_AREA)
            patches = []
            for m in comps:
                p_img, p_mask, bbox = extract_patch(gray, m)
                if p_img is None:
                    continue
                patches.append({
                    "bbox": bbox,
                    "patch_path": f"patch_seq{s:04d}_t{t:03d}_x{bbox[0]}_y{bbox[1]}.png",
                    "mask_path": f"mask_seq{s:04d}_t{t:03d}_x{bbox[0]}_y{bbox[1]}.png",
                })
                cv2.imwrite(str(out_dir / patches[-1]["patch_path"]), p_img)
                cv2.imwrite(str(out_dir / patches[-1]["mask_path"]), p_mask)
            record = {"seq": int(s), "t": int(t), "num_components": len(comps), "patches": patches}
            if cfg.PATCH_SAVE_OVERLAYS:
                overlay = overlay_debug(gray, comps)
                overlay_path = out_dir / f"overlay_seq{s:04d}_t{t:03d}.png"
                cv2.imwrite(str(overlay_path), overlay)
                record["overlay"] = overlay_path.name
            manifest.append(record)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump({"items": manifest}, f, indent=2)

    print(str(out_dir / "manifest.json"))


if __name__ == "__main__":
    main()
