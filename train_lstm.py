#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from simple_eye_dataset import SimpleEyeTrackingDataset

class ImprovedEyeTrackingLSTM(nn.Module):
    """
    Improved LSTM with attention mechanism for better eye tracking
    """
    
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embed = nn.Linear(input_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, input_size)
        )
        
    def forward(self, x):
        # Input embedding
        x_embed = self.input_embed(x)  # [batch, seq_len, hidden_size]
        
        # LSTM forward
        lstm_out, _ = self.lstm(x_embed)  # [batch, seq_len, hidden_size]
        
        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        attended = attended + lstm_out
        
        # Output projection
        attended = self.output_norm(attended)
        output = self.output_proj(attended)  # [batch, seq_len, 2]
        
        return output

class ResidualEyeTrackingLSTM(nn.Module):
    """
    Residual LSTM that predicts displacement instead of absolute coordinates
    """
    
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Predict displacement
        self.displacement_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, input_size)
        )
        
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Predict displacement
        displacement = self.displacement_proj(lstm_out)
        
        # Add to last known position (residual connection)
        last_position = x[:, -1:, :].expand(-1, x.size(1), -1)
        output = last_position + displacement
        
        return output

def train_improved_model(model_type='improved'):
    """Train improved model variants"""
    
    print(f"🚀 Training {model_type} LSTM model")
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Data setup
    train_dataset = SimpleEyeTrackingDataset('./data/coco_search', is_training=True)
    test_dataset = SimpleEyeTrackingDataset('./data/coco_search', is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Larger batch
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Model setup
    if model_type == 'improved':
        model = ImprovedEyeTrackingLSTM(
            input_size=2,
            hidden_size=128,
            num_layers=3,
            dropout=0.2
        ).to(device)
    elif model_type == 'residual':
        model = ResidualEyeTrackingLSTM(
            input_size=2,
            hidden_size=64,
            num_layers=2,
            dropout=0.1
        ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Training loop
    num_epochs = 20
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_seq, target_seq in test_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                output = model(input_seq)
                loss = criterion(output, target_seq)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'best_{model_type}_model.pth')
            print('  ✅ Saved best model')
    
    print(f'Best validation loss: {best_loss:.4f}')
    return model_type, best_loss

def compare_models():
    """Compare different model architectures"""
    
    print("🔬 Comparing Model Architectures\n")
    
    results = {}
    
    # Train different models
    model_types = ['improved', 'residual']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        try:
            model_name, best_loss = train_improved_model(model_type)
            results[model_name] = best_loss
        except Exception as e:
            print(f"❌ {model_type} model failed: {e}")
            results[model_type] = float('inf')
    
    # Compare with baseline
    baseline_loss = 0.0138  # From previous training
    results['baseline'] = baseline_loss
    
    print(f"\n🏆 Model Comparison Results:")
    print(f"{'Model':<15} {'Best Loss':<12} {'Improvement':<12}")
    print("-" * 40)
    
    for model_name, loss in sorted(results.items(), key=lambda x: x[1]):
        if loss == float('inf'):
            improvement = "Failed"
        else:
            improvement = f"{((baseline_loss - loss) / baseline_loss * 100):+.1f}%"
        print(f"{model_name:<15} {loss:<12.4f} {improvement:<12}")
    
    return results

def main():
    print("🎯 Improving LSTM Eye Tracking Model\n")
    
    # Compare different architectures
    results = compare_models()
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1])
    
    print(f"\n🏅 Best model: {best_model[0]} (loss: {best_model[1]:.4f})")
    
    if best_model[1] < 0.01:
        print("🎉 Excellent performance! Ready for real data.")
    elif best_model[1] < 0.02:
        print("✅ Good performance! Consider further optimization.")
    else:
        print("⚠️  Moderate performance. May need architectural changes.")

if __name__ == '__main__':
    main()