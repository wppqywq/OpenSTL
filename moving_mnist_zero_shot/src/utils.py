"""Common utilities for Moving MNIST zero-shot benchmark."""
import json
import sys
from pathlib import Path
import numpy as np


def add_path():
    """Add current directory to sys.path if not already present."""
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def to_gray_2d(frame: np.ndarray) -> np.ndarray:
    """Convert any frame format to 2D grayscale."""
    if frame.ndim == 2:
        return frame
    elif frame.ndim == 3:
        # (H, W, C) -> (H, W)
        if frame.shape[-1] == 3:
            return frame[..., 0]
        elif frame.shape[0] == 3:
            return frame[0]
        else:
            return frame[..., 0]
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")


def standardize_array(arr: np.ndarray):
    """Standardize array layout to access frames as (seq_idx, t)."""
    if arr.ndim == 3:
        # (N, H, W) or (T, H, W)
        if arr.shape[0] > 100:  # Likely N sequences
            def get_frame(seq_idx, t):
                return arr[seq_idx]
            return get_frame, arr.shape[0], 1
        else:  # Likely T timesteps
            def get_frame(seq_idx, t):
                return arr[t]
            return get_frame, 1, arr.shape[0]
    
    elif arr.ndim == 4:
        # (N, T, H, W) or (T, N, H, W)
        if arr.shape[0] < arr.shape[1]:  # T first
            T, N = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                return arr[t, seq_idx]
            return get_frame, N, T
        else:  # N first
            N, T = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                return arr[seq_idx, t]
            return get_frame, N, T
    
    elif arr.ndim == 5:
        # (T, N, H, W, C) or (N, T, H, W, C)
        if arr.shape[0] < arr.shape[1]:  # T first
            T, N = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                return arr[t, seq_idx, ..., 0]  # Take first channel
            return get_frame, N, T
        else:  # N first
            N, T = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                return arr[seq_idx, t, ..., 0]  # Take first channel
            return get_frame, N, T
    
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


def load_patch_manifest(output_root: str):
    """Load and parse patch detection manifest into grouped structure."""
    det_root = Path(output_root) / "patch_detach"
    manifest_path = det_root / "manifest.json"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    grouped = {}
    for entry in manifest["entries"]:
        seq_idx = entry["seq_idx"]
        t = entry["t"]
        if seq_idx not in grouped:
            grouped[seq_idx] = {}
        grouped[seq_idx][t] = entry
    
    return grouped, manifest
