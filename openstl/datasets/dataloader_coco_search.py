# openstl/datasets/dataloader_coco_search.py
"""
COCO-Search18 eye tracking dataset loader for OpenSTL
Supports both coordinate and heatmap representations
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class COCOSearchDataset(Dataset):
    """COCO-Search18 eye tracking dataset"""
    
    def __init__(self, data_root, split='train', pre_seq_length=5, 
                 aft_seq_length=5, spatial_size=32, representation='heatmap',
                 dataset_config='short', use_augment=False):
        """
        Args:
            data_root: Path to processed data directory
            split: 'train' or 'val'
            pre_seq_length: Input sequence length
            aft_seq_length: Output sequence length
            spatial_size: Size of spatial heatmap (if using heatmap representation)
            representation: 'coordinate' or 'heatmap'
            dataset_config: 'short', 'medium', 'standard', or 'success_only'
            use_augment: Whether to apply data augmentation
        """
        self.data_root = Path(data_root)
        self.split = split
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.spatial_size = spatial_size
        self.representation = representation
        self.use_augment = use_augment
        
        # Load sequences
        prefix = f"{dataset_config}_" if dataset_config != 'standard' else ''
        seq_file = self.data_root / f'{prefix}{split}_sequences.npy'
        meta_file = self.data_root / f'{prefix}{split}_metadata.json'
        
        if not seq_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {seq_file}")
        
        self.sequences = np.load(seq_file)
        
        # Load metadata if available
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = [{}] * len(self.sequences)
        
        # Validate sequence length
        self.total_length = self.pre_seq_length + self.aft_seq_length
        if self.sequences.shape[1] < self.total_length:
            print(f"WARNING: Sequence length mismatch!")
            print(f"  Data has sequences of length: {self.sequences.shape[1]}")
            print(f"  Requested: pre={self.pre_seq_length} + aft={self.aft_seq_length} = {self.total_length}")
            print(f"  Dataset config: {dataset_config}")
            
            # Auto-adjust if possible
            if self.sequences.shape[1] == 10 and self.total_length == 20:
                print(f"  Auto-adjusting to pre=5, aft=5")
                self.pre_seq_length = 5
                self.aft_seq_length = 5
                self.total_length = 10
            else:
                raise ValueError(f"Cannot auto-adjust. Please check your configuration.")
        
        print(f"Loaded {len(self.sequences)} {split} sequences")
        print(f"Representation: {representation}, Spatial size: {spatial_size}")
        
        # Calculate statistics for normalization
        if self.representation == 'heatmap':
            sample_heatmap = self.coords_to_heatmap(self.sequences[0])
            self.mean = np.mean(sample_heatmap)
            self.std = np.std(sample_heatmap) + 1e-6
        else:
            self.mean = 0.5
            self.std = 0.5
        
        # Add required attribute for OpenSTL compatibility
        self.data_name = 'coco_search'
    
    def __len__(self):
        return len(self.sequences)
    
    def coords_to_heatmap(self, coords):
        """Convert coordinate sequence to spatial heatmap representation"""
        seq_len = len(coords)
        heatmaps = np.zeros((seq_len, 1, self.spatial_size, self.spatial_size), 
                           dtype=np.float32)
        
        sigma = 2.0  # Gaussian spread
        
        for t in range(seq_len):
            x, y = coords[t]
            
            # Convert to pixel coordinates
            px = int(x * (self.spatial_size - 1))
            py = int(y * (self.spatial_size - 1))
            
            # Ensure within bounds
            px = np.clip(px, 0, self.spatial_size - 1)
            py = np.clip(py, 0, self.spatial_size - 1)
            
            # Create Gaussian heatmap
            for i in range(max(0, px - 5), min(self.spatial_size, px + 6)):
                for j in range(max(0, py - 5), min(self.spatial_size, py + 6)):
                    dist_sq = (i - px)**2 + (j - py)**2
                    heatmaps[t, 0, j, i] = np.exp(-dist_sq / (2 * sigma**2))
        
        return heatmaps
    
    def augment_coords(self, coords):
        """Apply data augmentation to coordinate sequence"""
        if not self.use_augment:
            return coords
        
        augmented = coords.copy()
        
        # Random translation
        if np.random.rand() > 0.5:
            offset = np.random.uniform(-0.05, 0.05, 2)
            augmented = augmented + offset
        
        # Random scaling around center
        if np.random.rand() > 0.7:
            scale = np.random.uniform(0.9, 1.1)
            center = np.array([0.5, 0.5])
            augmented = (augmented - center) * scale + center
        
        # Clip to valid range
        augmented = np.clip(augmented, 0, 1)
        
        return augmented
    
    def __getitem__(self, idx):
        """Get a training sample"""
        # Get coordinate sequence
        full_seq = self.sequences[idx].copy()
        
        # Apply augmentation
        if self.use_augment and self.split == 'train':
            full_seq = self.augment_coords(full_seq)
        
        # Split into input and target
        input_seq = full_seq[:self.pre_seq_length]
        target_seq = full_seq[self.pre_seq_length:self.total_length]
        
        # Convert representation if needed
        if self.representation == 'heatmap':
            input_seq = self.coords_to_heatmap(input_seq)
            target_seq = self.coords_to_heatmap(target_seq)
        else:
            # Keep as coordinates but add channel dimension
            input_seq = input_seq[np.newaxis, :, :]  # [1, T, 2]
            target_seq = target_seq[np.newaxis, :, :]
            
            # Transpose to [T, C, ...] format
            input_seq = np.transpose(input_seq, (1, 0, 2))
            target_seq = np.transpose(target_seq, (1, 0, 2))
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_seq).float()
        target_tensor = torch.from_numpy(target_seq).float()
        
        return input_tensor, target_tensor


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=10, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False,
              drop_last=False, representation='heatmap', dataset_config='short',
              **kwargs):
    """Load COCO-Search18 data for OpenSTL
    
    Returns:
        train_loader, val_loader, test_loader, test_mean, test_std
    """
    # Debug print
    print(f"Loading COCO-Search18 data:")
    print(f"  data_root: {data_root}")
    print(f"  pre_seq_length: {pre_seq_length}")
    print(f"  aft_seq_length: {aft_seq_length}")
    print(f"  dataset_config: {dataset_config}")
    
    # Fix data_root path
    data_root_path = Path(data_root)
    
    # Check if we need to add the coco_search18_tp/processed path
    if not (data_root_path / f'{dataset_config}_train_sequences.npy').exists():
        # Try common paths
        possible_paths = [
            data_root_path / 'coco_search18_tp' / 'processed',
            data_root_path / 'coco_search' / 'processed',
            Path('./data/coco_search18_tp/processed'),
        ]
        
        for path in possible_paths:
            if (path / f'{dataset_config}_train_sequences.npy').exists():
                data_root = str(path)
                print(f"  Found data at: {data_root}")
                break
        else:
            print(f"  WARNING: Could not find {dataset_config}_train_sequences.npy")
            print(f"  Searched in: {data_root} and common paths")
    
    # For 'short' config, force correct lengths
    if dataset_config == 'short':
        pre_seq_length = 5
        aft_seq_length = 5
        print(f"  Using short config: pre=5, aft=5")
    
    # Extract spatial size from in_shape if provided
    if in_shape is not None and len(in_shape) >= 3:
        spatial_size = in_shape[-1]  # Assuming square spatial dimensions
    else:
        spatial_size = 32
    
    # Create datasets
    train_set = COCOSearchDataset(
        data_root=data_root,
        split='train',
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        spatial_size=spatial_size,
        representation=representation,
        dataset_config=dataset_config,
        use_augment=use_augment
    )
    
    val_set = COCOSearchDataset(
        data_root=data_root,
        split='val',
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        spatial_size=spatial_size,
        representation=representation,
        dataset_config=dataset_config,
        use_augment=False
    )
    
    # Use validation set as test set
    test_set = val_set
    
    # Get normalization statistics
    test_mean = train_set.mean
    test_std = train_set.std
    
    # Create data loaders
    train_sampler = None
    val_sampler = None
    
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set, shuffle=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=drop_last
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = val_loader
    
    return train_loader, val_loader, test_loader