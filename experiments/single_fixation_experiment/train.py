#!/usr/bin/env python3
"""
Main training script for single fixation experiment.
Supports resume from checkpoint and early stopping.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import time
import numpy as np
from tqdm import tqdm

# Add OpenSTL to path
sys.path.insert(0, '/Users/apple/git/neuro/OpenSTL')

from openstl.models import SimVP_Model
import config


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class SingleFixationLoss(nn.Module):
    """Combined loss function for single fixation prediction"""
    
    def __init__(self, sparsity_weight=0.8, concentration_weight=1.5, 
                 focal_alpha=1.0, coordinate_weight=1.0, background_weight=0.1):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.concentration_weight = concentration_weight
        self.focal_alpha = focal_alpha
        self.coordinate_weight = coordinate_weight
        self.background_weight = background_weight
    
    def focal_loss(self, predictions, targets, alpha=1.0, gamma=2.0):
        """Focal loss for sparse targets"""
        if predictions.dim() == 5:
            predictions = predictions.squeeze(2)
        if targets.dim() == 5:
            targets = targets.squeeze(2)
        
        predictions_sigmoid = torch.sigmoid(predictions)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none')
        
        p_t = torch.where(targets == 1, predictions_sigmoid, 1 - predictions_sigmoid)
        focal_weight = alpha * (1 - p_t) ** gamma
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()
    
    def sparsity_loss(self, predictions):
        """Encourage winner-take-all sparsity using entropy"""
        if predictions.dim() == 5:
            predictions = predictions.squeeze(2)
        probs = torch.sigmoid(predictions)
        
        B, T, H, W = probs.shape
        probs_flat = probs.view(B, T, -1)
        probs_flat = probs_flat / (probs_flat.sum(dim=-1, keepdim=True) + 1e-8)
        
        entropy = -torch.sum(probs_flat * torch.log(probs_flat + 1e-8), dim=-1)
        return entropy.mean()
    
    def concentration_loss(self, predictions):
        """Encourage sharp peaks"""
        if predictions.dim() == 5:
            predictions = predictions.squeeze(2)
        probs = torch.sigmoid(predictions)
        
        B, T, H, W = probs.shape
        probs_flat = probs.view(B, T, -1)
        max_vals, _ = torch.max(probs_flat, dim=-1)
        concentration = 1.0 - max_vals
        return concentration.mean()
    
    def soft_coordinate_loss(self, predictions, targets):
        """Soft coordinate loss using center of mass"""
        device = predictions.device
        
        if predictions.dim() == 5:
            predictions = predictions.squeeze(2)
        if targets.dim() == 5:
            targets = targets.squeeze(2)
        
        pred_probs = torch.sigmoid(predictions)
        B, T, H, W = pred_probs.shape
        
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
        x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
        
        total_loss = 0.0
        for b in range(B):
            for t in range(T):
                pred_mass = pred_probs[b, t] / (pred_probs[b, t].sum() + 1e-8)
                pred_x = (pred_mass * x_coords).sum()
                pred_y = (pred_mass * y_coords).sum()
                
                target_mass = targets[b, t] / (targets[b, t].sum() + 1e-8)
                target_x = (target_mass * x_coords).sum()
                target_y = (target_mass * y_coords).sum()
                
                distance = torch.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
                total_loss += distance
        
        return total_loss / (B * T)
    
    def background_mse_loss(self, predictions, targets):
        """MSE loss for background regions"""
        if predictions.dim() == 5:
            predictions = predictions.squeeze(2)
        if targets.dim() == 5:
            targets = targets.squeeze(2)
        
        pred_probs = torch.sigmoid(predictions)
        background_mask = (targets == 0).float()
        background_pred = pred_probs * background_mask
        background_loss = torch.mean(background_pred ** 2)
        
        return background_loss
    
    def forward(self, predictions, targets):
        """Combined loss function"""
        focal = self.focal_loss(predictions, targets, alpha=self.focal_alpha)
        sparsity = self.sparsity_loss(predictions)
        concentration = self.concentration_loss(predictions)
        coordinate = self.soft_coordinate_loss(predictions, targets)
        background = self.background_mse_loss(predictions, targets)
        
        total_loss = (focal + 
                     self.sparsity_weight * sparsity + 
                     self.concentration_weight * concentration +
                     self.coordinate_weight * coordinate +
                     self.background_weight * background)
        
        return total_loss, {
            'focal': focal.item() if isinstance(focal, torch.Tensor) else focal,
            'sparsity': sparsity.item() if isinstance(sparsity, torch.Tensor) else sparsity,
            'concentration': concentration.item() if isinstance(concentration, torch.Tensor) else concentration,
            'coordinate': coordinate.item() if isinstance(coordinate, torch.Tensor) else coordinate,
            'background': background.item() if isinstance(background, torch.Tensor) else background,
            'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        }


def create_data_loaders():
    """Create data loaders with correct baseline splitting"""
    train_data = torch.load(config.train_data_file)
    val_data = torch.load(config.val_data_file)
    test_data = torch.load(config.test_data_file)
    
    # Extract frames and reformat: (B,C,T,H,W) -> (B,T,C,H,W)
    train_frames = train_data['frames'].permute(0, 2, 1, 3, 4)
    val_frames = val_data['frames'].permute(0, 2, 1, 3, 4)
    test_frames = test_data['frames'].permute(0, 2, 1, 3, 4)
    
    # Split into input and target: first 16 frames -> last 16 frames
    train_input = train_frames[:, :config.input_frames]
    train_target = train_frames[:, config.input_frames:]
    
    val_input = val_frames[:, :config.input_frames]
    val_target = val_frames[:, config.input_frames:]
    
    test_input = test_frames[:, :config.input_frames]
    test_target = test_frames[:, config.input_frames:]
    
    # Create datasets
    train_dataset = TensorDataset(train_input, train_target)
    val_dataset = TensorDataset(val_input, val_target)
    test_dataset = TensorDataset(test_input, test_target)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


def create_model(device):
    """Create SimVP model"""
    model = SimVP_Model(
        in_shape=(config.input_frames, 1, config.img_size, config.img_size),
        hid_S=config.model_hid_S,
        hid_T=config.model_hid_T,
        N_S=config.model_N_S,
        N_T=config.model_N_T,
        model_type=config.model_type,
        mlp_ratio=config.model_mlp_ratio,
        drop=config.model_drop,
        drop_path=config.model_drop_path,
        spatio_kernel_enc=config.model_spatio_kernel_enc,
        spatio_kernel_dec=config.model_spatio_kernel_dec
    ).to(device)
    
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path):
    """Load training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def main():
    """Main training function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path to resume from')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(config.random_seed)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders()
    print(f"Data loaded: {config.train_size} train, {config.val_size} val, {config.test_size} test samples")
    
    # Create model
    model = create_model(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function and optimizer
    criterion = SingleFixationLoss(
        sparsity_weight=config.sparsity_weight,
        concentration_weight=config.concentration_weight,
        focal_alpha=config.focal_alpha,
        coordinate_weight=config.coordinate_weight,
        background_weight=config.background_weight
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=config.lr_scheduler_patience, factor=config.lr_scheduler_factor)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume or config.resume_training:
        checkpoint_path = args.checkpoint or config.resume_checkpoint or config.checkpoint_path
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
            start_epoch += 1
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience, 
        min_delta=config.early_stopping_min_delta)
    
    # Training loop
    print(f"Starting training from epoch {start_epoch}")
    
    best_val_loss = float('inf')
    
    # Create log file
    with open(config.training_log_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Focal,Sparsity,Concentration,Coordinate,Background,LR,Time\n")
    
    for epoch in range(start_epoch, config.max_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_components_sum = {}
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss, loss_components = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            for key, val in loss_components.items():
                if key not in train_components_sum:
                    train_components_sum[key] = 0.0
                train_components_sum[key] += val
        
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_components = {k: v/len(train_loader) for k, v in train_components_sum.items()}
        
        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss, _ = criterion(output, target)
                val_loss_sum += loss.item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.final_model_path)
            print(f"New best model saved: {avg_val_loss:.6f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, avg_val_loss, config.checkpoint_path)
        
        epoch_time = time.time() - start_time
        
        # Logging
        print(f"Epoch {epoch+1}/{config.max_epochs} ({epoch_time:.1f}s)")
        print(f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, LR: {current_lr:.2e}")
        
        # Log to file
        with open(config.training_log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},"
                   f"{avg_train_components['focal']:.6f},{avg_train_components['sparsity']:.6f},"
                   f"{avg_train_components['concentration']:.6f},{avg_train_components['coordinate']:.6f},"
                   f"{avg_train_components['background']:.6f},{current_lr:.2e},{epoch_time:.1f}\n")
        
        # Early stopping check
        if early_stopping(avg_val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            early_stopping.restore_best(model)
            torch.save(model.state_dict(), config.final_model_path)
            break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    return best_val_loss


if __name__ == "__main__":
    main() 