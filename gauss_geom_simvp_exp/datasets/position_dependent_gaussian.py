#!/usr/bin/env python3
"""
Simplified MVP generator for center-inward Gaussian trajectories.

Update per step:
  Delta ~ N( s * mu(y, x), s^2 * Sigma(y, x) )
  pos_{t+1} = clamp( pos_t + Delta, [0, H-1] )

Mode: center_inward only
  - mu(y,x) = alpha * unit_vector_to_center (no decay)
  - Sigma(y,x) = lam1 * u*u^T + lam2 * n*n^T (outer product construction)

Coordinate convention: (y, x). All functions use consistent (y,x) ordering.
"""

import math
import os
import torch
from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader
from .config import IMG_SIZE, SEQUENCE_LENGTH, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, HEATMAP_SIGMA, FIELD_CONFIG, RANDOM_SEED

RANDOM_SEED = 42
# Optional global seed (keep per-call seed override higher priority)
torch.manual_seed(RANDOM_SEED)

def _compute_mu_sigma(y, x, img_size, config):
    """
    Compute mu [dy, dx] and Sigma (2x2) for center_inward mode with center-weak/edge-strong behavior.
    Consistent (y,x) ordering throughout. Uses outer product for covariance (no rotation matrix).
    """
    # Base params for position-dependent behavior (use FIELD_CONFIG as fallback)
    alpha0 = float(config.get('alpha', FIELD_CONFIG['alpha']))
    lam1_min = float(config.get('lambda1_min', FIELD_CONFIG['lambda1_min']))
    lam1_max = float(config.get('lambda1_max', FIELD_CONFIG['lambda1_max']))
    lam2_min = float(config.get('lambda2_min', FIELD_CONFIG['lambda2_min']))  # ensure positive definite
    T_total = float(config.get('lambda_total', FIELD_CONFIG['lambda_total']))

    center = img_size / 2.0
    dy = center - float(y)
    dx = center - float(x)
    norm = math.hypot(dy, dx) + 1e-12
    
    # Unit vectors
    uy, ux = dy / norm, dx / norm           # unit direction toward center
    u = torch.tensor([uy, ux], dtype=torch.float32)
    n = torch.tensor([-ux, uy], dtype=torch.float32)   # perpendicular direction

    # Radius-dependent weight: center weak (w~0), edge strong (w~1)
    Rmax = math.sqrt(2) * (img_size / 2.0)  # maximum distance to corner
    rho = min(1.0, math.hypot(dy, dx) / Rmax)  # normalized radius ∈ [0,1]
    w = rho  # simple linear weight function

    # Mean: center-weak / edge-strong
    mu = alpha0 * w * u

    # Anisotropy: keep trace constant, increase λ1 with radius
    lam1 = lam1_min + (lam1_max - lam1_min) * w
    lam2 = max(lam2_min, T_total - lam1)

    # Sigma = lam1 * u*u^T + lam2 * n*n^T (ellipse along u direction)
    Sigma = lam1 * torch.outer(u, u) + lam2 * torch.outer(n, n)
    
    return mu, Sigma

def _clamp_to_bounds(pos, img_size):
    """Clamp [y, x] within [0, img_size-1]."""
    y = max(0.0, min(float(pos[0]), img_size - 1.0))
    x = max(0.0, min(float(pos[1]), img_size - 1.0))
    return torch.tensor([y, x])

def sample_position_dependent_gaussian(batch_size, sequence_length, img_size, config=None, seed=None):
    """Sample coordinates step-by-step from (0,0) using center-inward field."""
    # Always start with FIELD_CONFIG as base, then update with passed config
    merged_config = FIELD_CONFIG.copy()
    if config is not None:
        merged_config.update(config)
    config = merged_config

    coords = torch.zeros(batch_size, sequence_length, 2)

    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))

    step_scale = float(config.get('step_scale', 3.0))

    for b in range(batch_size):
        # Always start at (0,0)
        coords[b, 0] = torch.tensor([0.0, 0.0])

        for t in range(1, sequence_length):
            current_pos = coords[b, t-1]
            mu_vec, sigma = _compute_mu_sigma(float(current_pos[0]), float(current_pos[1]), img_size, config)

            cov_stable = sigma + 1e-6 * torch.eye(2)
            L = torch.linalg.cholesky(cov_stable)
            z = torch.randn(2, generator=gen) if gen is not None else torch.randn(2)
            displacement = step_scale * (mu_vec + L @ z)
            new_pos = current_pos + displacement
            coords[b, t] = _clamp_to_bounds(new_pos, img_size)

    return coords.to(torch.float32)

def render_sparse_frames(coords, img_size):
    """Render binary frames with a single active pixel per coordinate."""
    batch_size, sequence_length, _ = coords.shape
    frames = torch.zeros(batch_size, sequence_length, 1, img_size, img_size)
    for b in range(batch_size):
        for t in range(sequence_length):
            y, x = coords[b, t]
            y_int, x_int = int(round(y.item())), int(round(x.item()))
            y_int = max(0, min(img_size - 1, y_int))
            x_int = max(0, min(img_size - 1, x_int))
            frames[b, t, 0, y_int, x_int] = 1.0
    return frames

def render_gaussian_frames(coords, img_size, sigma=2.0):
    """Render Gaussian heatmaps centered at coordinates."""
    batch_size, sequence_length, _ = coords.shape
    frames = torch.zeros(batch_size, sequence_length, 1, img_size, img_size)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32),
        torch.arange(img_size, dtype=torch.float32),
        indexing='ij'
    )
    for b in range(batch_size):
        for t in range(sequence_length):
            y, x = coords[b, t]
            gaussian = torch.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            frames[b, t, 0] = gaussian
    return frames

def generate_dataset(batch_size, sequence_length, img_size, config=None, representation='sparse'):
    """Generate frames, coordinates, and mask using simplified center-inward sampling."""
    seed = None if config is None else config.get('seed', None)
    coords = sample_position_dependent_gaussian(batch_size, sequence_length, img_size, config, seed=seed)
    mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)

    if representation == 'sparse':
        frames = render_sparse_frames(coords, img_size)
    elif representation == 'gaussian':
        sigma = config.get('heatmap_sigma', 2.0) if config else 2.0
        frames = render_gaussian_frames(coords, img_size, sigma)
    else:
        raise ValueError(f"Unknown representation: {representation}")

    return {'frames': frames, 'coordinates': coords, 'mask': mask}


# Dataset and data loader classes

class GaussianFieldDataset(Dataset):
    """Dataset of trajectories sampled from position-dependent Gaussian field."""
    
    def __init__(self, split='train', data_path=None, generate=False, config=None, representation='gaussian'):
        """Initialize dataset; load from disk or generate on the fly."""
        self.split = split
        self.generate = generate
        self.config = config or {}
        self.representation = representation
        
        if generate:
            num_sequences = self.config.get('num_sequences', TRAIN_SIZE if split == 'train' else 
                                          VAL_SIZE if split == 'val' else TEST_SIZE)
            sequence_length = self.config.get('sequence_length', SEQUENCE_LENGTH)
            img_size = self.config.get('img_size', IMG_SIZE)
            
            data = generate_dataset(num_sequences, sequence_length, img_size,
                                  config=self.config, representation=self.representation)
            self.frames = data['frames']
            self.coordinates = data['coordinates']
            self.mask = data['mask']
        else:
            if data_path is None:
                data_path = Path("data/position_dependent_gaussian")
            else:
                data_path = Path(data_path)
            
            rep = representation.lower() if isinstance(representation, str) else representation
            rep = 'gaussian' if rep in ('heatmap', 'gauss') else rep
            file_path = data_path / f"{split}_{rep}_data.pt"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            data = torch.load(file_path)
            self.frames = data['frames']
            self.coordinates = data['coordinates']
            self.mask = data['mask']
        
        print(f"Loaded {split} data ({representation}): {self.coordinates.shape[0]} sequences, "
              f"{self.coordinates.shape[1]} frames each")
    
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        """Get a single sequence."""
        return {
            'frames': self.frames[idx],
            'coordinates': self.coordinates[idx],
            'mask': self.mask[idx]
        }


class GaussianFieldGenerator:
    """Generate and save Gaussian field datasets (train/val/test)."""
    
    def __init__(self, output_dir=None, config=None):
        """Initialize generator."""
        self.config = config or FIELD_CONFIG.copy()
        
        self.img_size = self.config.get('img_size', IMG_SIZE)
        self.sequence_length = self.config.get('sequence_length', SEQUENCE_LENGTH)
        self.train_size = self.config.get('train_size', TRAIN_SIZE)
        self.val_size = self.config.get('val_size', VAL_SIZE)
        self.test_size = self.config.get('test_size', TEST_SIZE)
        self.mode = self.config.get('mode', 'center_inward')
        
        if output_dir is None:
            output_dir = Path("data/position_dependent_gaussian")
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_datasets(self):
        """Generate train/val/test datasets and write to disk."""
        print(f"Generating datasets with mode: {self.mode}")
        print(f"Training size: {self.train_size}, Validation size: {self.val_size}, Test size: {self.test_size}")
        
        base_seed = self.config.get('seed', None)
        
        print("Generating training dataset...")
        seed_train = None if base_seed is None else int(base_seed)
        coords_train = sample_position_dependent_gaussian(self.train_size, self.sequence_length, self.img_size, config=self.config, seed=seed_train)
        train_data_sparse = {
            'frames': render_sparse_frames(coords_train, self.img_size),
            'coordinates': coords_train,
            'mask': torch.ones(self.train_size, self.sequence_length, dtype=torch.bool)
        }
        train_data_gaussian = {
            'frames': render_gaussian_frames(coords_train, self.img_size, self.config.get('heatmap_sigma', 2.0)),
            'coordinates': coords_train,
            'mask': torch.ones(self.train_size, self.sequence_length, dtype=torch.bool)
        }
        
        print("Generating validation dataset...")
        seed_val = None if base_seed is None else int(base_seed) + 1
        coords_val = sample_position_dependent_gaussian(self.val_size, self.sequence_length, self.img_size, config=self.config, seed=seed_val)
        val_data_sparse = {
            'frames': render_sparse_frames(coords_val, self.img_size),
            'coordinates': coords_val,
            'mask': torch.ones(self.val_size, self.sequence_length, dtype=torch.bool)
        }
        val_data_gaussian = {
            'frames': render_gaussian_frames(coords_val, self.img_size, self.config.get('heatmap_sigma', 2.0)),
            'coordinates': coords_val,
            'mask': torch.ones(self.val_size, self.sequence_length, dtype=torch.bool)
        }
        
        print("Generating test dataset...")
        seed_test = None if base_seed is None else int(base_seed) + 2
        coords_test = sample_position_dependent_gaussian(self.test_size, self.sequence_length, self.img_size, config=self.config, seed=seed_test)
        test_data_sparse = {
            'frames': render_sparse_frames(coords_test, self.img_size),
            'coordinates': coords_test,
            'mask': torch.ones(self.test_size, self.sequence_length, dtype=torch.bool)
        }
        test_data_gaussian = {
            'frames': render_gaussian_frames(coords_test, self.img_size, self.config.get('heatmap_sigma', 2.0)),
            'coordinates': coords_test,
            'mask': torch.ones(self.test_size, self.sequence_length, dtype=torch.bool)
        }
        
        # Save datasets
        print("Saving datasets...")
        torch.save(train_data_sparse, self.output_dir / "train_sparse_data.pt")
        torch.save(train_data_gaussian, self.output_dir / "train_gaussian_data.pt")
        torch.save(val_data_sparse, self.output_dir / "val_sparse_data.pt")
        torch.save(val_data_gaussian, self.output_dir / "val_gaussian_data.pt")
        torch.save(test_data_sparse, self.output_dir / "test_sparse_data.pt")
        torch.save(test_data_gaussian, self.output_dir / "test_gaussian_data.pt")
        
        # Save configuration
        with open(self.output_dir / "config.json", "w") as f:
            config_dict = {k: v if not isinstance(v, tuple) else list(v) 
                           for k, v in self.config.items()}
            json.dump(config_dict, f, indent=4)
        
        print(f"All datasets saved to {self.output_dir}")
        
        return {
            'train': {'sparse': train_data_sparse, 'gaussian': train_data_gaussian},
            'val': {'sparse': val_data_sparse, 'gaussian': val_data_gaussian},
            'test': {'sparse': test_data_sparse, 'gaussian': test_data_gaussian}
        }


def calculate_displacement_vectors(coordinates, mask):
    """
    Compute per-step displacement vectors and validity mask.
    
    Args:
        coordinates: [B, T, 2] or [T, 2] tensor
        mask: [B, T] or [T] tensor
        
    Returns:
        displacements: [B, T-1, 2] or [T-1, 2] 
        displacement_mask: [B, T-1] or [T-1]
    """
    if coordinates.dim() == 3:  # [B, T, 2] format
        B, T = coordinates.shape[:2]
        displacements = torch.zeros(B, T-1, 2)
        displacement_mask = torch.zeros(B, T-1, dtype=torch.bool)
        
        for b in range(B):
            for t in range(T-1):
                if mask[b, t] and mask[b, t+1]:
                    displacements[b, t] = coordinates[b, t+1] - coordinates[b, t]
                    displacement_mask[b, t] = True
                    
    elif coordinates.dim() == 2:  # [T, 2] format (single sequence)
        T = coordinates.shape[0]
        displacements = torch.zeros(T-1, 2)
        displacement_mask = torch.zeros(T-1, dtype=torch.bool)
        
        for t in range(T-1):
            if mask[t] and mask[t+1]:
                displacements[t] = coordinates[t+1] - coordinates[t]
                displacement_mask[t] = True
    else:
        raise ValueError(f"Expected coordinates shape [B,T,2] or [T,2], got {coordinates.shape}")
    
    return displacements, displacement_mask


def create_data_loaders(batch_size=32, data_path=None, generate=False, config=None, representation='gaussian'):
    """Create train/val/test DataLoaders for center-inward dataset."""
    if config is None:
        config = {}

    train_dataset = GaussianFieldDataset('train', data_path, generate, config, representation)
    val_dataset = GaussianFieldDataset('val', data_path, generate, config, representation)
    test_dataset = GaussianFieldDataset('test', data_path, generate, config, representation)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main():
    """Config-first entry via env vars."""
    import os
    cfg = FIELD_CONFIG.copy()
    cfg_path = os.environ.get('GAUSS_MVP_CONFIG')
    output_dir = os.environ.get('GAUSS_MVP_OUTPUT_DIR')
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            file_cfg = json.load(f)
        cfg.update(file_cfg)
    generator = GaussianFieldGenerator(output_dir, cfg)
    generator.generate_datasets()


if __name__ == "__main__":
    main()
