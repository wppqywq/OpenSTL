import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import create_loader


class COCOSearchDataset(Dataset):
    def __init__(self, data_root, is_training=True, pre_seq_length=10, aft_seq_length=10, 
                 in_shape=None, transform=None):
        super(COCOSearchDataset, self).__init__()
        
        self.data_root = data_root
        self.is_training = is_training
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.total_length = pre_seq_length + aft_seq_length
        self.transform = transform
        self.in_shape = in_shape
        
        # Data normalization parameters
        self.mean = 0.0
        self.std = 1.0
        self.data_name = 'coco_search'
        
        # Load scanpath data
        self._load_data()
        
    def _load_data(self):
        """Load COCO-Search scanpath data"""
        split = 'train' if self.is_training else 'test'
        data_path = os.path.join(self.data_root, f'coco_search_{split}.npy')
        
        if not os.path.exists(data_path):
            # If data doesn't exist, create dummy data for testing
            print(f"Warning: {data_path} not found, generating dummy data for testing")
            self._generate_dummy_data()
        else:
            self.data = np.load(data_path, allow_pickle=True)
            
        self.length = len(self.data)
        
    def _generate_dummy_data(self):
        """Generate dummy scanpath data for testing"""
        # Dummy data: [num_samples, seq_length, 2] for (x, y) coordinates
        num_samples = 1000 if self.is_training else 200
        seq_length = self.total_length
        
        # Generate random scanpaths normalized to [0, 1]
        self.data = np.random.rand(num_samples, seq_length, 2).astype(np.float32)
        
    def _normalize_coordinates(self, coords):
        """Normalize coordinates to [0, 1] range"""
        # Assuming coords are already normalized or implement normalization
        return coords
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        """Get a single scanpath sequence"""
        scanpath = self.data[index]  # [seq_length, 2]
        
        # Normalize coordinates
        scanpath = self._normalize_coordinates(scanpath)
        
        # Split into input and target sequences
        input_seq = scanpath[:self.pre_seq_length]    # [pre_seq_length, 2]
        target_seq = scanpath[self.pre_seq_length:self.pre_seq_length + self.aft_seq_length]  # [aft_seq_length, 2]
        
        # Convert to tensors
        input_seq = torch.from_numpy(input_seq).float()
        target_seq = torch.from_numpy(target_seq).float()
        
        # Reshape to match SimVP input format [T, C, H, W]
        # For scanpath, we treat coordinates as 2D spatial maps
        if self.in_shape:
            _, _, H, W = self.in_shape
        else:
            H, W = 64, 64  # Default spatial resolution
        
        # Get spatial dimensions from config
        if self.in_shape:
            _, _, H, W = self.in_shape
        else:
            H, W = 64, 64

        # Convert coordinates to spatial heatmaps
        input_spatial = self._coords_to_spatial(input_seq, H, W)   # [pre_seq_length, 2, H, W]
        target_spatial = self._coords_to_spatial(target_seq, H, W) # [aft_seq_length, 2, H, W]
        
        return input_spatial, target_spatial
        
    def _coords_to_spatial(self, coords, H, W):
        """Convert coordinate sequence to spatial representation"""
        T = coords.shape[0]
        
        # Determine number of channels from in_shape
        if self.in_shape:
            _, C, _, _ = self.in_shape
        else:
            C = 1  # Default to 1 channel
            
        spatial = torch.zeros(T, C, H, W)
        
        for t in range(T):
            x, y = coords[t]
            # Convert normalized coordinates to spatial indices  
            h_idx = max(0, min(int(y * (H - 1)), H-1))
            w_idx = max(0, min(int(x * (W - 1)), W-1))
            
            if C == 1:
                # Single channel: just mark presence
                spatial[t, 0, h_idx, w_idx] = 1.0
            else:
                # Two channels: separate X and Y
                spatial[t, 0, h_idx, w_idx] = 1.0  # X presence
                spatial[t, 1, h_idx, w_idx] = 1.0  # Y presence
            
        return spatial


def load_data(batch_size, val_batch_size, data_root, num_workers, 
              pre_seq_length=10, aft_seq_length=10, in_shape=(20, 2, 64, 64),
              distributed=False, use_augment=False, use_prefetcher=False, 
              drop_last=False, **kwargs):
    """Load COCO-Search dataset"""
    
    # Create datasets
    train_set = COCOSearchDataset(
        data_root=data_root,
        is_training=True,
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        in_shape=in_shape
    )
    
    test_set = COCOSearchDataset(
        data_root=data_root,
        is_training=False,
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        in_shape=in_shape
    )
    
    # Create data loaders
    train_loader = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True, 
        is_training=True,
        pin_memory=True, 
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher
    )
    
    test_loader = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False, 
        is_training=False,
        pin_memory=True, 
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher
    )
    
    return train_loader, None, test_loader