import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class COCOSearchDataset(Dataset):
    """COCO-Search18 eye tracking dataset for OpenSTL
    
    This dataset contains eye movement scanpaths from the COCO-Search18 dataset.
    Each sample consists of a sequence of (x, y) coordinates representing gaze positions.
    """
    
    def __init__(self, data_root, split='train', seq_length=20, 
                 pre_seq_length=10, aft_seq_length=10,
                 spatial_size=32, use_augment=False):
        """
        Args:
            data_root: Path to data directory
            split: 'train' or 'test'
            seq_length: Total sequence length
            pre_seq_length: Input sequence length
            aft_seq_length: Output sequence length
            spatial_size: Size of spatial representation
            use_augment: Whether to use data augmentation
        """
        self.data_root = data_root
        self.split = split
        self.seq_length = seq_length
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.spatial_size = spatial_size
        self.use_augment = use_augment
        
        # Load data
        if split == 'train':
            self.data = np.load(os.path.join(data_root, 'coco_search_train.npy'))
        else:
            self.data = np.load(os.path.join(data_root, 'coco_search_test.npy'))
        
        print(f"Loaded {split} set with {len(self.data)} sequences")
        
        # Calculate mean and std for compatibility with OpenSTL
        # Convert a sample to spatial and calculate stats
        sample_spatial = self.coords_to_spatial(self.data[0])
        self.mean = np.mean(sample_spatial)
        self.std = np.std(sample_spatial)
        print(f"Dataset mean: {self.mean:.4f}, std: {self.std:.4f}")
        
        # Add required attributes for OpenSTL compatibility
        self.data_name = 'coco_search'
        self.data_root = data_root
        
    def __len__(self):
        return len(self.data)
    
    def coords_to_spatial(self, coords):
        """Convert (x, y) coordinates to spatial heatmap representation
        
        Args:
            coords: Array of shape [seq_len, 2] with normalized coordinates
            
        Returns:
            spatial: Array of shape [seq_len, 1, H, W]
        """
        seq_len = len(coords)
        spatial = np.zeros((seq_len, 1, self.spatial_size, self.spatial_size), dtype=np.float32)
        
        for t in range(seq_len):
            x, y = coords[t]
            
            # Convert normalized [0, 1] to pixel coordinates
            px = int(x * (self.spatial_size - 1))
            py = int(y * (self.spatial_size - 1))
            
            # Ensure within bounds
            px = np.clip(px, 0, self.spatial_size - 1)
            py = np.clip(py, 0, self.spatial_size - 1)
            
            # Create Gaussian-like heatmap centered at gaze position
            sigma = 2.0
            for i in range(max(0, px - 5), min(self.spatial_size, px + 6)):
                for j in range(max(0, py - 5), min(self.spatial_size, py + 6)):
                    dist = np.sqrt((i - px)**2 + (j - py)**2)
                    spatial[t, 0, j, i] = np.exp(-dist**2 / (2 * sigma**2))
        
        return spatial
    
    def augment_sequence(self, coords):
        """Apply data augmentation to coordinate sequence"""
        if not self.use_augment:
            return coords
            
        # Random small translation
        if np.random.rand() > 0.5:
            offset = np.random.uniform(-0.05, 0.05, 2)
            coords = coords + offset
            coords = np.clip(coords, 0, 1)
        
        # Random small scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            center = 0.5
            coords = (coords - center) * scale + center
            coords = np.clip(coords, 0, 1)
            
        return coords
    
    def __getitem__(self, idx):
        """Get a single sample
        
        Returns:
            input_seq: Tensor of shape [pre_seq_length, 1, H, W]
            target_seq: Tensor of shape [aft_seq_length, 1, H, W]
        """
        # Get coordinate sequence
        coords = self.data[idx].copy()  # [seq_length, 2]
        
        # Apply augmentation
        coords = self.augment_sequence(coords)
        
        # Convert to spatial representation
        spatial = self.coords_to_spatial(coords)
        
        # Split into input and target
        input_seq = spatial[:self.pre_seq_length]
        target_seq = spatial[self.pre_seq_length:self.pre_seq_length + self.aft_seq_length]
        
        return torch.from_numpy(input_seq), torch.from_numpy(target_seq)


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=10, in_shape=None,
              distributed=False, use_augment=False, test_mean=None, test_std=None, 
              dataset_config='short', **kwargs):
    """Load COCO-Search data
    
    Args:
        batch_size: Training batch size
        val_batch_size: Validation batch size  
        data_root: Path to data directory
        num_workers: Number of data loading workers
        pre_seq_length: Input sequence length
        aft_seq_length: Output sequence length
        in_shape: Input shape (not used, for compatibility)
        distributed: Whether using distributed training
        use_augment: Whether to use data augmentation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_set = COCOSearchDataset(
        data_root=data_root,
        split='train',
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        use_augment=use_augment
    )
    
    test_set = COCOSearchDataset(
        data_root=data_root,
        split='test',
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        use_augment=False
    )
    
    # Create data loaders
    dataloader_num_workers = num_workers
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=dataloader_num_workers
    )
    
    val_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=dataloader_num_workers
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=dataloader_num_workers
    )
    
    return train_loader, val_loader, test_loader