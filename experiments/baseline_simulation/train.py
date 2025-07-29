#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Add OpenSTL to path
sys.path.insert(0, '/Users/apple/git/neuro/OpenSTL')

from openstl.models import SimVP_Model


class CorrectedWhiteDotLoss(nn.Module):
    """
    Corrected loss function that avoids center bias by using:
    1. Focal Loss for detection (on logits)
    2. Soft-argmax coordinates for spatial guidance
    3. Low-weight background MSE to preserve overall spatial structure
    """
    def __init__(self, alpha=1.0, delta=0.02, gamma=5.0, focal_gamma=2.0):
        super().__init__()
        self.alpha = alpha      # Detection weight (Focal Loss) - reduced
        self.delta = delta      # Background MSE weight (minimal)
        self.gamma = gamma      # Soft coordinate weight - increased significantly  
        self.focal_gamma = focal_gamma
        
    def focal_loss(self, predictions, targets):
        """Focal loss for detection"""
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.focal_gamma * bce_loss
        return focal_loss.mean()
    
    def background_mse_loss(self, predictions, targets):
        """Low-weight background MSE to preserve overall spatial structure"""
        pred_probs = torch.sigmoid(predictions)
        return F.mse_loss(pred_probs, targets)
    
    def soft_coordinate_loss(self, predictions, targets):
        """Vectorized soft coordinate loss using proper soft-argmax with normalized coordinates"""
        B, T, C, H, W = targets.shape
        device = targets.device
        
        # Vectorized target coordinate extraction
        targ_flat = targets.view(B, T, -1)  # [B, T, H*W]
        idx = torch.argmax(targ_flat, dim=-1)  # [B, T]
        y = idx // W
        x = idx % W
        target_coords = torch.stack([x, y], dim=-1).float()  # [B, T, 2]
        
        # Normalize target coordinates to [0,1] for stable regression
        target_coords[:, :, 0] = target_coords[:, :, 0] / (W - 1)  # x coordinates
        target_coords[:, :, 1] = target_coords[:, :, 1] / (H - 1)  # y coordinates
        
        # Vectorized prediction coordinate extraction using soft-argmax
        pred_coords = self.extract_soft_coordinates(predictions)
        
        return F.smooth_l1_loss(pred_coords, target_coords)
    
    def extract_soft_coordinates(self, predictions, temperature=0.005):
        """True soft-argmax coordinate extraction using softmax for proper probability distribution"""
        B, T, C, H, W = predictions.shape
        device = predictions.device
        
        # Create coordinate grids [H*W, 2] with normalized coordinates
        y_coords = torch.arange(H, dtype=torch.float32, device=device) / (H - 1)
        x_coords = torch.arange(W, dtype=torch.float32, device=device) / (W - 1)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # [H*W, 2]
        
        # Handle multiple channels with warning
        if C == 1:
            logits = predictions.squeeze(2)  # [B, T, H, W]
        else:
            print(f"Warning: channel={C}, averaging over channels")
            logits = predictions.mean(2)  # Average over channels if C > 1
            
        # Apply temperature and flatten for softmax
        logits_flat = (logits / temperature).view(B, T, -1)  # [B, T, H*W]
        
        # Dimension safety check
        assert logits_flat.numel() == B * T * H * W, f"Dimension mismatch: {logits_flat.numel()} != {B * T * H * W}"
        
        # True softmax probability distribution (sum=1 for each spatial map)
        probs = F.softmax(logits_flat, dim=-1)  # [B, T, H*W]
        
        # Compute weighted coordinates using einsum: [B, T, H*W] @ [H*W, 2] -> [B, T, 2]
        coords = torch.einsum('btn,nc->btc', probs, grid_coords)
        
        return coords
    
    def forward(self, predictions, targets):
        # Detection loss (Focal Loss) - main detection signal
        focal_loss = self.focal_loss(predictions, targets)
        
        # Soft coordinate loss - spatial guidance without center bias
        coord_loss = self.soft_coordinate_loss(predictions, targets)
        
        # Low-weight background MSE to preserve overall spatial structure
        background_mse_loss = self.background_mse_loss(predictions, targets)
        
        # Combined loss with adjusted weights
        total_loss = self.alpha * focal_loss + self.gamma * coord_loss + self.delta * background_mse_loss
        
        return total_loss, focal_loss, background_mse_loss, coord_loss


class ContiguousSimVP(nn.Module):
    """Wrapper for SimVP model to ensure contiguous tensors"""
    def __init__(self, in_shape, hid_S=64, hid_T=256, N_S=4, N_T=8, model_type='gSTA'):
        super().__init__()
        self.model = SimVP_Model(
            in_shape=in_shape,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            model_type=model_type
        )
    
    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        output = self.model(x)
        if not output.is_contiguous():
            output = output.contiguous()
        return output


def extract_hard_coordinates(heatmaps, normalize=True):
    """Vectorized coordinate extraction using direct argmax (for evaluation)"""
    B, T, C, H, W = heatmaps.shape
    device = heatmaps.device
    
    # Flatten spatial dimensions
    if C == 1:
        flat_heatmaps = heatmaps.squeeze(2).view(B, T, -1)  # [B, T, H*W]
    else:
        flat_heatmaps = heatmaps.mean(2).view(B, T, -1)  # Average over channels
    
    # Get argmax indices for all samples at once
    max_indices = torch.argmax(flat_heatmaps, dim=-1)  # [B, T]
    
    # Convert to coordinates
    y = max_indices // W
    x = max_indices % W
    coords = torch.stack([x, y], dim=-1).float()  # [B, T, 2]
    
    # Normalize coordinates to [0,1] if requested
    if normalize:
        coords[:, :, 0] = coords[:, :, 0] / (W - 1)  # x coordinates
        coords[:, :, 1] = coords[:, :, 1] / (H - 1)  # y coordinates
    
    return coords


def evaluate_model(model, dataloader, device):
    """Evaluate model with corrected coordinate extraction - uses full probability distribution"""
    model.eval()
    total_error = 0.0
    total_samples = 0
    errors_under_1px = 0
    errors_under_5px = 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Model prediction
            predictions = model(batch_inputs)
            
            # Ensure predictions have the right shape
            if predictions.dim() == 4:  # (B, T, H, W)
                predictions = predictions.unsqueeze(2)  # Add channel dimension
            
            # Extract coordinates using direct argmax on full probability distribution
            target_coords = extract_hard_coordinates(batch_targets, normalize=False)
            
            # Apply sigmoid to convert logits to probabilities, then argmax
            pred_probs = torch.sigmoid(predictions)
            pred_coords = extract_hard_coordinates(pred_probs, normalize=False)
            
            # Calculate errors in pixel space
            errors = torch.norm(pred_coords - target_coords, dim=-1)
            
            total_error += errors.sum().item()
            total_samples += errors.numel()
            errors_under_1px += (errors < 1.0).sum().item()
            errors_under_5px += (errors < 5.0).sum().item()
    
    mean_error = total_error / total_samples
    acc_1px = errors_under_1px / total_samples * 100
    acc_5px = errors_under_5px / total_samples * 100
    
    return mean_error, acc_1px, acc_5px


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_focal = 0.0
    total_heatmap = 0.0
    total_coord = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_inputs, batch_targets in progress_bar:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_inputs)
        
        # Ensure predictions have the right shape
        if predictions.dim() == 4:  # (B, T, H, W)
            predictions = predictions.unsqueeze(2)  # Add channel dimension
            
        loss, focal_loss, background_mse_loss, coord_loss = criterion(predictions, batch_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_focal += focal_loss.item()
        total_heatmap += background_mse_loss.item()
        total_coord += coord_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Focal': f'{focal_loss.item():.4f}',
            'MSE': f'{background_mse_loss.item():.4f}',
            'Coord': f'{coord_loss.item():.4f}'
        })
    
    num_batches = len(dataloader)
    return (total_loss / num_batches, total_focal / num_batches, 
            total_heatmap / num_batches, total_coord / num_batches)


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_focal = 0.0
    total_heatmap = 0.0
    total_coord = 0.0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            predictions = model(batch_inputs)
            
            # Ensure predictions have the right shape
            if predictions.dim() == 4:  # (B, T, H, W)
                predictions = predictions.unsqueeze(2)  # Add channel dimension
                
            loss, focal_loss, background_mse_loss, coord_loss = criterion(predictions, batch_targets)
            
            total_loss += loss.item()
            total_focal += focal_loss.item()
            total_heatmap += background_mse_loss.item()
            total_coord += coord_loss.item()
    
    num_batches = len(dataloader)
    return (total_loss / num_batches, total_focal / num_batches,
            total_heatmap / num_batches, total_coord / num_batches)


def main(num_epochs=None, batch_size=None, lr=None):
    """Main training function with corrected loss and optimizations"""
    import config
    
    # Set random seed for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.random_seed)
    
    # Use config values as defaults, override with function parameters if provided
    epochs = num_epochs if num_epochs is not None else config.epochs
    batch_sz = batch_size if batch_size is not None else config.batch_size
    learning_rate = lr if lr is not None else config.lr
    
    print(f"Training with epochs={epochs}, batch_size={batch_sz}, lr={learning_rate}")
    print(f"Using sigma={config.sigma}, random_seed={config.random_seed}")
    print("Optimizations: vectorized coordinate extraction, normalized coords, dimension safety")
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine num_workers based on device
    if device.type == 'mps':
        num_workers = 0  # macOS MPS doesn't work well with multiprocessing
    else:
        cpu_count = os.cpu_count()
        num_workers = min(4, cpu_count if cpu_count is not None else 2)  # Use up to 4 workers on other platforms
    
    print(f"DataLoader num_workers: {num_workers}")
    
    # Load data
    print("Loading data...")
    train_data_dict = torch.load("data/train_data.pt")
    val_data_dict = torch.load("data/val_data.pt")
    test_data_dict = torch.load("data/test_data.pt")
    
    # Extract frames from dictionary format (B, C, T, H, W) -> (B, T, C, H, W)
    train_data = train_data_dict['frames'].permute(0, 2, 1, 3, 4)  # (B, 1, 32, 32, 32) -> (B, 32, 1, 32, 32)
    val_data = val_data_dict['frames'].permute(0, 2, 1, 3, 4)
    test_data = test_data_dict['frames'].permute(0, 2, 1, 3, 4)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    print(f"Train data shape after adding channel: {train_data.shape}")
    
    # Create datasets
    train_inputs = train_data[:, :16]  # First 16 frames as input
    train_targets = train_data[:, 16:]  # Last 16 frames as target
    val_inputs = val_data[:, :16]
    val_targets = val_data[:, 16:]
    
    # Convert to binary (white dot detection)
    train_targets = (train_targets > 0.5).float()
    val_targets = (val_targets > 0.5).float()
    
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    
    # Create data loaders with optimized workers
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    
    # Model configuration
    model_config = {
        'in_shape': (16, 1, 32, 32),
        'hid_S': 64,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 8,
        'model_type': 'gSTA'
    }
    
    # Create model
    model = ContiguousSimVP(**model_config).to(device)
    
    # Create corrected loss function with coordinate-focused weights
    criterion = CorrectedWhiteDotLoss(alpha=1.0, delta=0.2, gamma=6.0, focal_gamma=2.0)
    
    # Optimizer and scheduler with responsive learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training configuration
    num_epochs = epochs
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Starting optimized training...")
    print("Loss weights: alpha=1.0 (detection), delta=0.2 (background MSE), gamma=6.0 (coord)")
    print("Coordinates normalized to [0,1] for stable regression")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_focal, train_heatmap, train_coord = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation
        val_loss, val_focal, val_heatmap, val_coord = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Evaluation on validation set
        val_error, val_acc_1px, val_acc_5px = evaluate_model(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  -> New best model saved! Val Error: {val_error:.3f}px")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log progress
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train: Loss={train_loss:.6f} Focal={train_focal:.6f} MSE={train_heatmap:.6f} Coord={train_coord:.6f}")
        print(f"  Val:   Loss={val_loss:.6f} Error={val_error:.3f}px <1px={val_acc_1px:.1f}% <5px={val_acc_5px:.1f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    print("=" * 50)
    print("Training completed!")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_inputs = test_data[:, :16]
    test_targets = test_data[:, 16:]
    
    # Convert test targets to binary
    test_targets = (test_targets > 0.5).float()
    
    test_dataset = TensorDataset(test_inputs, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    
    # Load best model
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    
    test_error, test_acc_1px, test_acc_5px = evaluate_model(model, test_loader, device)
    
    print(f"Final Test Results:")
    print(f"  Mean Error: {test_error:.3f} pixels")
    print(f"  <1px Accuracy: {test_acc_1px:.1f}%")
    print(f"  <5px Accuracy: {test_acc_5px:.1f}%")
    
    # Success criteria check
    success = test_error < 5.0 and test_acc_5px > 50.0
    print(f"  Success: {'PASS' if success else 'FAIL'}")


if __name__ == "__main__":
    main() 