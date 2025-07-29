#!/usr/bin/env python3
"""
Train SimVP model on Gaussian Enhanced dataset.
STEP 2 REFACTORING: Simplified loss function from 5 components to 2 components.
STEP 3 REFACTORING: Added history trails ablation support.
"""

import os
import sys
from pathlib import Path
import time
import argparse

# Compatibility patch for NumPy 2.0
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

# Stub torch_optimizer if needed
import types, torch
if 'torch_optimizer' not in sys.modules:
    torch_opt_stub = types.ModuleType('torch_optimizer')
    torch_opt_stub.RAdam = torch.optim.Adam
    sys.modules['torch_optimizer'] = torch_opt_stub

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root for OpenSTL import
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openstl.methods import SimVP
import config


# Simplified Loss Functions


def gaussian_mse_loss(pred_logits, target_heatmaps):
    """Direct MSE loss between predictions and Gaussian targets"""
    pred_probs = torch.sigmoid(pred_logits)
    return torch.nn.functional.mse_loss(pred_probs, target_heatmaps)


def _expected_coordinates(heatmap: torch.Tensor):
    """Extract coordinates using soft-argmax (center of mass)"""
    B, T, H, W = heatmap.shape
    
    # Normalize to probability distribution
    heatmap_flat = heatmap.view(B, T, -1)
    heatmap_norm = heatmap_flat / (heatmap_flat.sum(dim=-1, keepdim=True) + 1e-8)
    heatmap_norm = heatmap_norm.view(B, T, H, W)
    
    # Create coordinate grids
    ys = torch.linspace(0, H - 1, H, device=heatmap.device)
    xs = torch.linspace(0, W - 1, W, device=heatmap.device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')
    x_grid = x_grid.view(1, 1, H, W)
    y_grid = y_grid.view(1, 1, H, W)
    
    # Expected coordinates
    exp_x = (heatmap_norm * x_grid).sum(dim=(2, 3))
    exp_y = (heatmap_norm * y_grid).sum(dim=(2, 3))
    coords = torch.stack([exp_x, exp_y], dim=-1)
    
    return coords


def coordinate_mse_loss(pred_logits, target_heatmaps):
    """MSE between soft-argmax coordinates"""
    pred_coords = _expected_coordinates(torch.sigmoid(pred_logits))
    target_coords = _expected_coordinates(target_heatmaps)
    return torch.nn.functional.mse_loss(pred_coords, target_coords)


class SimplifiedLoss(nn.Module):
    """Simplified loss function with only MSE and coordinate components"""
    
    def __init__(self, mse_weight=1.0, coord_weight=2.0):
        super().__init__()
        self.weights = {
            'mse': mse_weight,
            'coord': coord_weight,
        }
    
    def forward(self, pred, target):
        if pred.dim() == 5:
            pred = pred.squeeze(2)
        if target.dim() == 5:
            target = target.squeeze(2)
        
        mse_loss = gaussian_mse_loss(pred, target)
        coord_loss = coordinate_mse_loss(pred, target)
        
        total_loss = (
            self.weights['mse'] * mse_loss +
            self.weights['coord'] * coord_loss
        )
        return total_loss


# Training parameters
PRE_SEQ_LEN = 4
AFT_SEQ_LEN = 16
TOTAL_FRAMES = PRE_SEQ_LEN + AFT_SEQ_LEN

BATCH_SIZE = 8
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset(split: str, use_history_trails: bool):
    """Load dataset for the specified split"""
    history_suffix = "_no_history" if not use_history_trails else "_with_history"
    filename = f"{split}_data{history_suffix}.pt"
    file_path = os.path.join(Path(__file__).parent, config.data_dir, filename)
    
    data = torch.load(file_path)
    frames = data['frames'].permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)
    
    assert frames.shape[1] >= TOTAL_FRAMES, (
        f"Dataset has {frames.shape[1]} frames, but TOTAL_FRAMES={TOTAL_FRAMES} required.")
    
    inp = frames[:, :PRE_SEQ_LEN]
    tgt = frames[:, PRE_SEQ_LEN:TOTAL_FRAMES]
    
    return TensorDataset(inp.float(), tgt.float())


def create_dataloaders(use_history_trails):
    """Create data loaders for training"""
    train_set = load_dataset('train', use_history_trails)
    val_set = load_dataset('val', use_history_trails)
    test_set = load_dataset('test', use_history_trails)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Evaluate model on given data loader"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = model.criterion(pred, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train SimVP on Gaussian Enhanced data")
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--no_history_trails', action='store_true',
                       help='Train without history trails')
    args = parser.parse_args()
    
    use_history_trails = not args.no_history_trails
    history_suffix = "_no_history" if args.no_history_trails else "_with_history"
    
    print(f"Using device: {DEVICE}")
    print(f"Gaussian sigma: {config.gaussian_sigma}")
    print(f"History trails: {use_history_trails}")
    
    # Load data
    print("Loading enhanced data...")
    train_loader, val_loader, test_loader = create_dataloaders(use_history_trails)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # Create model
    model = SimVP(
        in_shape=(PRE_SEQ_LEN, 1, config.img_size, config.img_size),
        pre_seq_length=PRE_SEQ_LEN,
        aft_seq_length=AFT_SEQ_LEN,
        hid_S=config.model_hid_S,
        hid_T=config.model_hid_T,
        N_S=config.model_N_S,
        N_T=config.model_N_T,
        model_type=config.model_type,
        lr=args.lr,
        dataname='gaussian_enhanced',
        metrics=[],
    ).to(DEVICE)
    
    # Use simplified loss function
    model.criterion = SimplifiedLoss(
        mse_weight=config.simple_mse_weight,
        coord_weight=config.simple_coord_weight,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val = float('inf')
    os.makedirs(config.models_dir, exist_ok=True)
    ckpt_path = os.path.join(config.models_dir, f'best_gaussian_enhanced{history_suffix}.pth')
    
    # Setup simplified logging
    log_file = f'training_log_gaussian{history_suffix}.txt'
    with open(log_file, 'w') as f:
        f.write('Epoch,Train_Loss,Val_Loss,Time\n')
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        tic = time.time()
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_x)
            
            loss = model.criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            batch_size = batch_x.size(0)
            epoch_loss += loss.item() * batch_size
        
        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, DEVICE)
        scheduler.step()
        toc = time.time() - tic
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.2e} | time={toc:.1f}s")
        
        # Log to CSV
        with open(log_file, 'a') as f:
            f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},{toc:.1f}\n')
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved new best model -> {ckpt_path}")
    
    # Final evaluation
    print("Training complete. Best val loss: {:.6f}".format(best_val))
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(ckpt_path))
    test_loss = evaluate(model, test_loader, DEVICE)
    print(f"Test loss: {test_loss:.6f}")


if __name__ == '__main__':
    main() 