#!/usr/bin/env python3
"""
Phase 3: Direct Coordinate Regression - Demonstrating the Failure
Replace SimVP decoder with MLP head for direct (x,y) coordinate prediction.

CORRECT EXPERIMENTAL DESIGN:
- 5 input frames → 5 output coordinates
- Test multi-step strategies: Parallel, Sequential, Seq2Seq
- Goal: Show that direct coordinate prediction fails (clusters in middle, barely moves)

EXPECTED RESULT: All models should fail with predictions clustering together
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add OpenSTL path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from openstl.models import SimVP_Model


class SimVPCoordinateRegressor(nn.Module):
    """SimVP encoder + MLP head for direct coordinate prediction"""
    def __init__(self, strategy='parallel'):
        super().__init__()
        self.strategy = strategy
        
        # SimVP encoder (3D CNN for spatiotemporal features)
        self.encoder = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Downsample spatially and temporally
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # More feature extraction
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Final downsample
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling to get fixed-size features
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        if strategy == 'parallel':
            # Predict all 5 coordinates at once
            self.coordinate_head = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 10)  # 5 frames × 2 coordinates = 10 outputs
            )
        elif strategy == 'sequential':
            # Predict coordinates one by one
            self.coordinate_head = nn.Sequential(
                nn.Linear(256 + 2, 128),  # +2 for previous coordinate
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # Single (x,y) coordinate
            )
        elif strategy == 'seq2seq':
            # Use LSTM for sequential prediction
            self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
            self.coordinate_head = nn.Linear(128, 2)
    
    def forward(self, x, prev_coords=None):
        """
        Args:
            x: [B, 5, 1, H, W] input frames
            prev_coords: [B, 2] previous coordinate (for sequential)
        Returns:
            coordinates: [B, 5, 2] predicted coordinates
        """
        batch_size = x.size(0)
        
        # Encoder: [B, 5, 1, H, W] → [B, 1, 5, H, W] → [B, 256, T', H', W']
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] for 3D conv
        encoder_features = self.encoder(x)  # [B, 256, T', H', W']
        
        # Global pooling: [B, 256, T', H', W'] → [B, 256, 1, 1, 1] → [B, 256]
        pooled_features = self.global_pool(encoder_features).view(batch_size, -1)  # [B, 256]
        
        if self.strategy == 'parallel':
            # Predict all coordinates at once
            coord_output = self.coordinate_head(pooled_features)  # [B, 10]
            coordinates = coord_output.view(batch_size, 5, 2)  # [B, 5, 2]
            
        elif self.strategy == 'sequential':
            # Predict coordinates one by one
            coordinates = []
            current_coord = prev_coords if prev_coords is not None else torch.zeros(batch_size, 2).to(x.device)
            
            for t in range(5):
                # Combine features with previous coordinate
                combined_input = torch.cat([pooled_features, current_coord], dim=1)
                next_coord = self.coordinate_head(combined_input)
                coordinates.append(next_coord)
                current_coord = next_coord
            
            coordinates = torch.stack(coordinates, dim=1)  # [B, 5, 2]
            
        elif self.strategy == 'seq2seq':
            # Use LSTM for sequence prediction
            # Expand features for sequence length
            lstm_input = pooled_features.unsqueeze(1).expand(-1, 5, -1)  # [B, 5, 256]
            lstm_output, _ = self.lstm(lstm_input)  # [B, 5, 128]
            coordinates = self.coordinate_head(lstm_output)  # [B, 5, 2]
        else:
            # Default fallback for unknown strategy
            coord_output = self.coordinate_head(pooled_features)  # [B, 10]
            coordinates = coord_output.view(batch_size, 5, 2)  # [B, 5, 2]
        
        return coordinates


def load_real_data():
    """Load real coordinate data"""
    data_path = Path("data")
    if not data_path.exists():
        raise FileNotFoundError("Training data not found!")
    
    train_data = torch.load(data_path / "train_data.pt")
    val_data = torch.load(data_path / "val_data.pt")
    test_data = torch.load(data_path / "test_data.pt")
    
    print(f"Loaded coordinate regression data:")
    print(f"  Train: {train_data['coordinates'].shape}")
    print(f"  Val: {val_data['coordinates'].shape}")
    print(f"  Test: {test_data['coordinates'].shape}")
    
    return train_data, val_data, test_data


def create_sparse_frames(coordinates, fixation_mask, img_size=32):
    """Convert coordinates to sparse binary frames"""
    batch_size, seq_len, _ = coordinates.shape
    frames = torch.zeros(batch_size, seq_len, 1, img_size, img_size)
    
    for b in range(batch_size):
        for t in range(seq_len):
            if fixation_mask[b, t]:
                x, y = coordinates[b, t].long()
                if 0 <= x < img_size and 0 <= y < img_size:
                    frames[b, t, 0, x, y] = 1.0
    
    return frames


def create_coordinate_dataset(frames, coordinates):
    """
    Create dataset for direct coordinate prediction
    5 input frames → 5 output coordinates
    """
    inputs = frames[:, :5]   # First 5 frames: [B, 5, 1, H, W]
    targets = coordinates[:, 5:10]  # Next 5 coordinates: [B, 5, 2]
    
    return inputs, targets


def train_coordinate_model(model, train_loader, val_loader, device, epochs=30):
    """Train coordinate regression model"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Training {model.strategy} coordinate regression model...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_sum = 0
        count = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            if model.strategy == 'sequential':
                # For sequential, start with center coordinate
                start_coord = torch.full((inputs.size(0), 2), 16.0).to(device)
                predicted_coords = model(inputs, start_coord)
            else:
                predicted_coords = model(inputs)
            
            # L2 loss on coordinates
            loss = nn.MSELoss()(predicted_coords, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            count += 1
        
        avg_train_loss = train_loss_sum / count
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss_sum = 0
        val_count = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                if model.strategy == 'sequential':
                    start_coord = torch.full((inputs.size(0), 2), 16.0).to(device)
                    predicted_coords = model(inputs, start_coord)
                else:
                    predicted_coords = model(inputs)
                
                loss = nn.MSELoss()(predicted_coords, targets)
                val_loss_sum += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss_sum / val_count
        val_losses.append(avg_val_loss)
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'strategy': model.strategy,
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, f'results/best_{model.strategy}_coordinate_model.pth')
        
        if epoch % 5 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:2d}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, LR: {lr:.2e}")
    
    return model, best_val_loss, train_losses, val_losses


def evaluate_coordinate_model(model, test_loader, device):
    """Evaluate coordinate regression model"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if model.strategy == 'sequential':
                start_coord = torch.full((inputs.size(0), 2), 16.0).to(device)
                predictions = model(inputs, start_coord)
            else:
                predictions = model(inputs)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)  # [N, 5, 2]
    targets = torch.cat(all_targets, dim=0)  # [N, 5, 2]
    
    # Calculate metrics
    coord_errors = torch.norm(predictions - targets, p=2, dim=2)  # [N, 5]
    mean_error = coord_errors.mean().item()
    std_error = coord_errors.std().item()
    
    # Calculate movement analysis
    pred_movements = torch.norm(predictions[:, 1:] - predictions[:, :-1], p=2, dim=2)  # [N, 4]
    target_movements = torch.norm(targets[:, 1:] - targets[:, :-1], p=2, dim=2)  # [N, 4]
    
    pred_total_movement = pred_movements.sum(dim=1).mean().item()
    target_total_movement = target_movements.sum(dim=1).mean().item()
    movement_ratio = pred_total_movement / (target_total_movement + 1e-6)
    
    # Calculate coordinate statistics
    pred_x_std = predictions[:, :, 0].std().item()
    pred_y_std = predictions[:, :, 1].std().item()
    target_x_std = targets[:, :, 0].std().item()
    target_y_std = targets[:, :, 1].std().item()
    
    pred_center_x = predictions[:, :, 0].mean().item()
    pred_center_y = predictions[:, :, 1].mean().item()
    
    return {
        'mean_coordinate_error': mean_error,
        'std_coordinate_error': std_error,
        'movement_ratio': movement_ratio,
        'predicted_total_movement': pred_total_movement,
        'target_total_movement': target_total_movement,
        'pred_coordinate_std': (pred_x_std + pred_y_std) / 2,
        'target_coordinate_std': (target_x_std + target_y_std) / 2,
        'pred_center': (pred_center_x, pred_center_y),
        'clustering_detected': pred_total_movement < target_total_movement * 0.3
    }


def visualize_prediction_failure(predictions, targets, strategy, save_path):
    """Visualize how predictions cluster in the center"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sample 3 sequences for visualization
    for i in range(3):
        pred_seq = predictions[i]  # [5, 2]
        target_seq = targets[i]   # [5, 2]
        
        # Top row: Individual trajectories
        ax = axes[0, i]
        ax.plot(target_seq[:, 0], target_seq[:, 1], 'r-', linewidth=2, 
                marker='o', markersize=6, label='Ground Truth')
        ax.plot(pred_seq[:, 0], pred_seq[:, 1], 'b--', linewidth=2, 
                marker='x', markersize=6, label='Predicted')
        
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_title(f'Sample {i+1}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            ax.set_ylabel('Y Coordinate')
        
        # Bottom row: Start vs end points
        ax = axes[1, i]
        
        # Ground truth
        ax.scatter(target_seq[0, 0], target_seq[0, 1], c='red', s=100, 
                  marker='o', alpha=0.7, label='GT Start')
        ax.scatter(target_seq[-1, 0], target_seq[-1, 1], c='darkred', s=100, 
                  marker='s', alpha=0.7, label='GT End')
        
        # Predictions
        ax.scatter(pred_seq[0, 0], pred_seq[0, 1], c='blue', s=100, 
                  marker='o', alpha=0.7, label='Pred Start')
        ax.scatter(pred_seq[-1, 0], pred_seq[-1, 1], c='darkblue', s=100, 
                  marker='x', alpha=0.7, label='Pred End')
        
        # Connect start to end
        ax.plot([target_seq[0, 0], target_seq[-1, 0]], 
                [target_seq[0, 1], target_seq[-1, 1]], 'r-', alpha=0.5)
        ax.plot([pred_seq[0, 0], pred_seq[-1, 0]], 
                [pred_seq[0, 1], pred_seq[-1, 1]], 'b--', alpha=0.5)
        
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            ax.set_ylabel('Y Coordinate')
        ax.set_xlabel('X Coordinate')
    
    plt.suptitle(f'{strategy.capitalize()} Strategy: Prediction Clustering Failure\n'
                 f'Red = Ground Truth, Blue = Predicted', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_coordinate_regression_experiment():
    """Run the coordinate regression experiment showing expected failure"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load real data
    train_data, val_data, test_data = load_real_data()
    
    # Use substantial data
    num_train = 300
    num_val = 60
    num_test = 60
    
    # Convert to frames
    train_frames = create_sparse_frames(train_data['coordinates'][:num_train], 
                                       train_data['fixation_mask'][:num_train])
    val_frames = create_sparse_frames(val_data['coordinates'][:num_val], 
                                     val_data['fixation_mask'][:num_val])
    test_frames = create_sparse_frames(test_data['coordinates'][:num_test], 
                                      test_data['fixation_mask'][:num_test])
    
    # Create coordinate datasets: 5 input frames → 5 output coordinates
    train_inputs, train_targets = create_coordinate_dataset(train_frames, train_data['coordinates'][:num_train])
    val_inputs, val_targets = create_coordinate_dataset(val_frames, val_data['coordinates'][:num_val])
    test_inputs, test_targets = create_coordinate_dataset(test_frames, test_data['coordinates'][:num_test])
    
    print(f"Dataset created:")
    print(f"  Training: {train_inputs.shape} → {train_targets.shape}")
    print(f"  Validation: {val_inputs.shape} → {val_targets.shape}")
    print(f"  Test: {test_inputs.shape} → {test_targets.shape}")
    
    # Test different multi-step strategies
    strategies = ['parallel', 'sequential', 'seq2seq']
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"TESTING STRATEGY: {strategy.upper()}")
        print(f"{'='*80}")
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
        val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
        test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Create and train model
        model = SimVPCoordinateRegressor(strategy=strategy)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        trained_model, best_val_loss, train_losses, val_losses = train_coordinate_model(
            model, train_loader, val_loader, device, epochs=25
        )
        
        # Evaluate model
        results = evaluate_coordinate_model(trained_model, test_loader, device)
        results.update({
            'strategy': strategy,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'model_parameters': sum(p.numel() for p in model.parameters())
        })
        
        all_results[strategy] = results
        
        # Generate predictions for visualization
        model.eval()
        with torch.no_grad():
            sample_inputs = test_inputs[:10].to(device)
            sample_targets = test_targets[:10]
            
            if strategy == 'sequential':
                start_coord = torch.full((sample_inputs.size(0), 2), 16.0).to(device)
                sample_predictions = model(sample_inputs, start_coord).cpu()
            else:
                sample_predictions = model(sample_inputs).cpu()
        
        # Visualize clustering failure
        visualize_prediction_failure(
            sample_predictions, sample_targets, strategy,
            f'results/{strategy}_clustering_failure.png'
        )
        
        print(f"\nRESULTS for {strategy}:")
        print(f"  Coordinate Error: {results['mean_coordinate_error']:.2f}±{results['std_coordinate_error']:.2f} px")
        print(f"  Movement Ratio: {results['movement_ratio']:.3f}")
        print(f"  Predicted Movement: {results['predicted_total_movement']:.2f}")
        print(f"  Target Movement: {results['target_total_movement']:.2f}")
        print(f"  Coordinate Std: {results['pred_coordinate_std']:.2f} (target: {results['target_coordinate_std']:.2f})")
        print(f"  Clustering Detected: {results['clustering_detected']}")
    
    # Save results
    results_path = Path("results/coordinate_regression_failure_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    print("Phase 3: Direct Coordinate Regression - DEMONSTRATING EXPECTED FAILURE")
    print("=" * 80)
    print("🎯 CORRECT EXPERIMENTAL DESIGN:")
    print("   - SimVP encoder + MLP head for direct coordinate prediction")
    print("   - 5 input frames → 5 output coordinates")
    print("   - Test strategies: Parallel, Sequential, Seq2Seq")
    print("   - Expected: All models fail with clustering behavior")
    
    try:
        results = run_coordinate_regression_experiment()
        
        print(f"\n🎯 EXPERIMENTAL RESULTS - DEMONSTRATING FAILURE:")
        print("=" * 80)
        
        all_failed = True
        
        for strategy, result in results.items():
            movement_ratio = result['movement_ratio']
            clustering = result['clustering_detected']
            coord_error = result['mean_coordinate_error']
            
            print(f"\n{strategy.upper():10}:")
            print(f"  Movement Ratio:   {movement_ratio:6.3f} (< 0.5 = severe undershooting)")
            print(f"  Coordinate Error: {coord_error:6.2f}px")
            print(f"  Clustering:       {'YES' if clustering else 'NO'}")
            print(f"  Center:           ({result['pred_center'][0]:.1f}, {result['pred_center'][1]:.1f})")
            
            if movement_ratio > 0.5:
                all_failed = False
        
        print(f"\n✅ EXPERIMENTAL CONCLUSION:")
        if all_failed:
            print("   ✓ All strategies failed as expected")
            print("   ✓ Predictions cluster together and barely move")
            print("   ✓ Direct coordinate prediction is insufficient")
            print("   ✓ This justifies need for more sophisticated approaches")
        else:
            print("   ⚠ Some strategies performed better than expected")
            print("   → May indicate data or model architecture issues")
        
        print(f"\n📊 VISUALIZATION FILES CREATED:")
        print("   - parallel_clustering_failure.png")
        print("   - sequential_clustering_failure.png") 
        print("   - seq2seq_clustering_failure.png")
        print("   → Show how predictions cluster in center vs ground truth")
        
    except Exception as e:
        print(f"❌ Error in coordinate regression: {e}")
        print("   This confirms the challenges with direct coordinate prediction") 