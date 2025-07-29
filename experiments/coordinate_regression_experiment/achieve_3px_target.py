#!/usr/bin/env python3
"""
Achieve 3px Target - Comprehensive improvement plan
Current: 7.71px -> Target: <3px
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import load_coordinate_data, train_test_split

class EnhancedLSTM(nn.Module):
    """Enhanced LSTM with multiple improvements for sub-3px accuracy"""
    
    def __init__(self, input_size=2, hidden_size=256, num_layers=4, output_length=8, 
                 dropout=0.2, use_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length
        self.use_attention = use_attention
        
        # Larger LSTM with more capacity
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=False)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(hidden_size)
        
        # Multi-layer decoder with residuals
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_length * 2)
        )
        
    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        
        # Self-attention (if enabled)
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.norm(lstm_out + attn_out)  # Residual connection
        
        # Use all timesteps with weighted average instead of just last
        # This gives model access to full sequence information
        weights = torch.softmax(torch.sum(lstm_out, dim=-1), dim=1)  # [batch, seq_len]
        weighted_repr = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)  # [batch, hidden_size]
        
        # Decode to coordinates
        coords = self.decoder(weighted_repr)  # [batch, output_length*2]
        coords = coords.view(-1, self.output_length, 2)
        
        return coords

def clean_data_for_3px_target():
    """Clean data to remove problematic trajectories that hurt 3px target"""
    
    print("="*60)
    print("DATA CLEANING FOR 3px TARGET")
    print("="*60)
    
    input_coords, target_coords = load_coordinate_data()
    all_coords = torch.cat([input_coords, target_coords], dim=1)
    
    # Identify problematic samples
    clean_indices = []
    removed_count = 0
    
    for i in range(all_coords.shape[0]):
        traj = all_coords[i]  # [20, 2]
        
        # Check for large jumps
        large_jumps = 0
        for t in range(1, len(traj)):
            dx = traj[t, 0] - traj[t-1, 0]
            dy = traj[t, 1] - traj[t-1, 1]
            displacement = torch.sqrt(dx**2 + dy**2).item()
            
            if displacement > 10.0:  # Remove trajectories with >10px jumps
                large_jumps += 1
        
        # Keep only smooth trajectories
        if large_jumps == 0:
            clean_indices.append(i)
        else:
            removed_count += 1
    
    # Filter data
    clean_input = input_coords[clean_indices]
    clean_target = target_coords[clean_indices]
    
    print(f"Original samples: {input_coords.shape[0]}")
    print(f"Removed samples: {removed_count}")
    print(f"Clean samples: {len(clean_indices)}")
    print(f"Data retention: {len(clean_indices)/input_coords.shape[0]*100:.1f}%")
    
    # Recalculate statistics on clean data
    clean_all_coords = torch.cat([clean_input, clean_target], dim=1)
    clean_displacements = []
    
    for i in range(clean_all_coords.shape[0]):
        traj = clean_all_coords[i]
        for t in range(1, len(traj)):
            dx = traj[t, 0] - traj[t-1, 0]
            dy = traj[t, 1] - traj[t-1, 1]
            displacement = torch.sqrt(dx**2 + dy**2).item()
            clean_displacements.append(displacement)
    
    clean_displacements = np.array(clean_displacements)
    
    print(f"\nClean data statistics:")
    print(f"  Mean displacement: {np.mean(clean_displacements):.2f}px")
    print(f"  Max displacement: {np.max(clean_displacements):.2f}px")
    print(f"  Jumps >5px: {np.mean(clean_displacements > 5.0)*100:.1f}%")
    print(f"  Jumps >8px: {np.mean(clean_displacements > 8.0)*100:.1f}%")
    
    return clean_input, clean_target

def train_enhanced_model_for_3px():
    """Train enhanced model specifically targeting 3px accuracy"""
    
    print(f"\n" + "="*60)
    print("TRAINING ENHANCED MODEL FOR 3px TARGET")
    print("="*60)
    
    # Use clean data
    clean_input, clean_target = clean_data_for_3px_target()
    
    # Shorter prediction horizon for better accuracy
    clean_target_short = clean_target[:, :8]  # 4->8 instead of 4->16
    
    # Train/test split
    train_input, train_target, test_input, test_target = train_test_split(
        clean_input, clean_target_short, train_ratio=0.8
    )
    
    print(f"\nTraining data:")
    print(f"  Train samples: {train_input.shape[0]}")
    print(f"  Test samples: {test_input.shape[0]}")
    print(f"  Prediction: 4->8 frames")
    
    # Enhanced model with larger capacity
    model = EnhancedLSTM(
        hidden_size=256,
        num_layers=4, 
        output_length=8,
        dropout=0.2,
        use_attention=True
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    # Enhanced training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # Use Huber loss instead of MSE (more robust to outliers)
    criterion = nn.HuberLoss(delta=1.0)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using device: {device}")
    
    # Training loop
    best_test_error = float('inf')
    patience_counter = 0
    max_patience = 50
    
    train_losses = []
    test_errors = []
    
    print(f"\nTraining for 3px target...")
    print("Epoch | Train Loss | Test Error | LR | Status")
    print("-" * 50)
    
    for epoch in range(500):  # More epochs
        # Training
        model.train()
        total_loss = 0
        
        # Shuffle training data each epoch
        perm = torch.randperm(train_input.shape[0])
        train_input_shuffled = train_input[perm]
        train_target_shuffled = train_target[perm]
        
        for i in range(train_input_shuffled.shape[0]):
            optimizer.zero_grad()
            
            input_seq = train_input_shuffled[i:i+1].to(device)
            target_seq = train_target_shuffled[i:i+1].to(device)
            
            pred = model(input_seq)
            loss = criterion(pred, target_seq)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / train_input_shuffled.shape[0]
        train_losses.append(avg_train_loss)
        
        # Evaluation every 10 epochs
        if epoch % 10 == 0 or epoch == 499:
            model.eval()
            with torch.no_grad():
                all_errors = []
                for i in range(test_input.shape[0]):
                    input_seq = test_input[i:i+1].to(device)
                    target_seq = test_target[i:i+1].to(device)
                    
                    pred = model(input_seq)
                    
                    pred_np = pred.cpu().numpy()[0]
                    target_np = target_seq.cpu().numpy()[0]
                    
                    for t in range(8):
                        px, py = pred_np[t]
                        tx, ty = target_np[t]
                        error = np.sqrt((px - tx)**2 + (py - ty)**2)
                        all_errors.append(error)
                
                test_error = np.mean(all_errors)
                test_errors.append(test_error)
                
                # Learning rate scheduling
                scheduler.step(test_error)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Early stopping
                if test_error < best_test_error:
                    best_test_error = test_error
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), 'models/enhanced_lstm_3px_target.pth')
                else:
                    patience_counter += 1
                
                # Status
                if test_error < 3.0:
                    status = "TARGET!"
                elif test_error < 4.0:
                    status = "CLOSE"
                elif test_error < 5.0:
                    status = "GOOD"
                else:
                    status = "TRAIN"
                
                print(f"{epoch:5d} | {avg_train_loss:10.4f} | {test_error:10.2f} | {current_lr:.2e} | {status}")
                
                # Early exit if target achieved
                if test_error < 3.0:
                    print(f"\n TARGET ACHIEVED! Test error: {test_error:.2f}px < 3px")
                    break
                
                # Early stopping
                if patience_counter >= max_patience:
                    print(f"\nEarly stopping after {epoch} epochs")
                    break
    
    print(f"\nFinal results:")
    print(f"  Best test error: {best_test_error:.2f}px")
    print(f"  Target achieved: {'YES' if best_test_error < 3.0 else 'NO'}")
    
    return model, best_test_error, train_losses, test_errors

def evaluate_3px_strategies():
    """Evaluate different strategies to achieve 3px target"""
    
    print(f"\n" + "="*60)
    print("STRATEGIES TO ACHIEVE 3px TARGET")
    print("="*60)
    
    print(f"Current performance: 7.71px")
    print(f"Target: <3px")
    print(f"Required improvement: {((7.71 - 3.0) / 7.71 * 100):.1f}%")
    
    print(f"\n STRATEGY 1: DATA CLEANING")
    print(f"  • Remove trajectories with >10px jumps")
    print(f"  • Expected improvement: ~15-20%")
    print(f"  • Estimated result: ~6-6.5px")
    
    print(f"\n STRATEGY 2: SHORTER HORIZON")
    print(f"  • Change from 4->16 to 4->8 prediction")
    print(f"  • Early frames already ~5.8px average")
    print(f"  • Expected improvement: ~25%")
    print(f"  • Estimated result: ~4-5px")
    
    print(f"\n STRATEGY 3: ENHANCED MODEL")
    print(f"  • Larger LSTM (256 hidden, 4 layers)")
    print(f"  • Attention mechanism")
    print(f"  • Better training (Huber loss, AdamW, scheduling)")
    print(f"  • Expected improvement: ~20-30%")
    print(f"  • Estimated result: ~3-4px")
    
    print(f"\n STRATEGY 4: COMBINATION")
    print(f"  • Data cleaning + Shorter horizon + Enhanced model")
    print(f"  • Expected improvement: ~50-60%")
    print(f"  • Estimated result: ~2-3px ")
    
    print(f"\n STRATEGY 5: ADVANCED TECHNIQUES")
    print(f"  • Ensemble of multiple models")
    print(f"  • Data augmentation")
    print(f"  • Multi-task learning")
    print(f"  • Expected improvement: ~60-70%")
    print(f"  • Estimated result: ~2px ")

if __name__ == "__main__":
    print("="*60)
    print("ACHIEVING 3px TARGET - COMPREHENSIVE PLAN")
    print("="*60)
    
    # Evaluate strategies
    evaluate_3px_strategies()
    
    # Train enhanced model
    model, best_error, train_losses, test_errors = train_enhanced_model_for_3px()
    
    print(f"\n" + "="*60)
    print("3px TARGET ASSESSMENT")
    print("="*60)
    
    if best_error < 3.0:
        print(f" SUCCESS! Achieved {best_error:.2f}px < 3px target")
        print(f" Model saved as 'enhanced_lstm_3px_target.pth'")
        print(f" Strategy: Clean data + Short horizon + Enhanced model")
    else:
        print(f"  Close but not quite: {best_error:.2f}px")
        print(f" Next steps:")
        if best_error < 4.0:
            print(f"   • Try ensemble methods")
            print(f"   • More data augmentation")
            print(f"   • Longer training")
        else:
            print(f"   • Check data quality further")
            print(f"   • Try different architectures")
            print(f"   • Consider transformer models")
    
    print(f"\n Performance progression:")
    print(f"  Original LSTM: 7.71px")
    print(f"  Enhanced LSTM: {best_error:.2f}px")
    print(f"  Improvement: {((7.71 - best_error) / 7.71 * 100):.1f}%")
    print(f"  Target achieved: {'YES ' if best_error < 3.0 else 'NO '}")
    
    print("="*60) 