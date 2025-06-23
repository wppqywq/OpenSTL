import torch
import numpy as np
from torch.utils.data import Dataset

class SimpleEyeTrackingDataset(Dataset):
    def __init__(self, data_root, is_training=True, pre_seq_length=10, aft_seq_length=10):
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        
        # Load data
        split = 'train' if is_training else 'test'
        data_path = f'{data_root}/coco_search_{split}.npy'
        self.data = np.load(data_path)  # [N, 20, 2]
        
        self.mean = 0.0
        self.std = 1.0
        self.data_name = 'simple_eye_tracking'
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]  # [20, 2]
        
        # Split into input and target
        input_seq = sequence[:self.pre_seq_length]    # [10, 2]
        target_seq = sequence[self.pre_seq_length:self.pre_seq_length + self.aft_seq_length]  # [10, 2]
        
        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)
