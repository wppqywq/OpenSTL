#!/usr/bin/env python

import torch
import torch.nn as nn
import os

class SimpleEyeTrackingLSTM(nn.Module):
    """
    Simple LSTM for eye tracking sequence prediction
    Input: [batch, seq_len, 2] coordinates  
    Output: [batch, seq_len, 2] predicted coordinates
    """
    
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # x: [batch, seq_len, 2]
        batch_size, seq_len = x.shape[:2]
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        
        # Project to coordinates
        output = self.output_proj(lstm_out)  # [batch, seq_len, 2]
        
        return output

def create_simple_dataset():
    """Create a simple dataset that works directly with coordinates"""
    
    dataset_code = '''import torch
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
'''
    
    with open('simple_eye_dataset.py', 'w') as f:
        f.write(dataset_code)
    
    print("✅ Created simple_eye_dataset.py")

def create_simple_training_script():
    """Create a standalone training script that doesn't depend on OpenSTL"""
    
    training_code = '''#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from simple_eye_dataset import SimpleEyeTrackingDataset

class SimpleEyeTrackingLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_proj = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.output_proj(lstm_out)
        return output

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output = model(input_seq)
            loss = criterion(output, target_seq)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    print("🚀 Simple LSTM Eye Tracking Training")
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using MPS")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU")
    
    # Data setup
    train_dataset = SimpleEyeTrackingDataset('./data/coco_search', is_training=True)
    test_dataset = SimpleEyeTrackingDataset('./data/coco_search', is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Model setup
    model = SimpleEyeTrackingLSTM(
        input_size=2,
        hidden_size=64,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    num_epochs = 10
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'best_eye_model.pth')
            print('✅ Saved best model')
    
    print(f'\\n🎉 Training completed! Best test loss: {best_loss:.4f}')
    
    # Quick evaluation
    model.load_state_dict(torch.load('best_eye_model.pth'))
    model.eval()
    
    with torch.no_grad():
        sample_input, sample_target = test_dataset[0]
        sample_input = sample_input.unsqueeze(0).to(device)
        sample_target = sample_target.unsqueeze(0).to(device)
        
        prediction = model(sample_input)
        
        print(f'\\nSample prediction:')
        print(f'Input shape: {sample_input.shape}')
        print(f'Target shape: {sample_target.shape}')
        print(f'Prediction shape: {prediction.shape}')
        print(f'First predicted point: ({prediction[0,0,0]:.3f}, {prediction[0,0,1]:.3f})')
        print(f'First true point: ({sample_target[0,0,0]:.3f}, {sample_target[0,0,1]:.3f})')

if __name__ == '__main__':
    main()
'''
    
    with open('train_simple_lstm.py', 'w') as f:
        f.write(training_code)
    
    print("✅ Created train_simple_lstm.py")

def main():
    print("🔧 Creating Simple LSTM Alternative for Eye Tracking\n")
    
    # Create simple dataset
    create_simple_dataset()
    
    # Create training script  
    create_simple_training_script()
    
    print("\n🎯 Simple LSTM solution created!")
    print("This is a clean, standalone solution that:")
    print("  ✅ Works directly with coordinate data")
    print("  ✅ Doesn't depend on complex OpenSTL configurations")
    print("  ✅ Is optimized for M2 MacBook")
    print("  ✅ Provides clear training feedback")
    
    print("\n🚀 To run:")
    print("  python train_simple_lstm.py")
    
    print("\nThis approach:")
    print("  • Uses raw coordinates directly (no spatial conversion)")
    print("  • Simple LSTM architecture")
    print("  • MSE loss on coordinate differences")
    print("  • Should train quickly on M2")

if __name__ == '__main__':
    main()