#!/usr/bin/env python
"""Plot training curves from logs."""

import re
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def parse_logs(log_dir):
    """Parse all training logs to extract loss values."""
    log_dir = Path(log_dir)
    
    all_train_losses = []
    all_val_losses = []
    all_lrs = []
    epochs = []
    
    # Parse all log files
    for log_file in sorted(log_dir.glob("train_*.log")):
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract epoch data: "Epoch X: Lr: Y | Train Loss: Z | Vali Loss: W"
        # Note: Some entries span multiple lines
        pattern = r"Epoch (\d+): Lr: ([\d.]+)\s*\| Train Loss: ([\d.]+) \| Vali Loss: ([\d.]+)"
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            epoch, lr, train_loss, val_loss = match
            epochs.append(int(epoch))
            all_lrs.append(float(lr))
            all_train_losses.append(float(train_loss))
            all_val_losses.append(float(val_loss))
    
    return epochs, all_train_losses, all_val_losses, all_lrs

def plot_curves(epochs, train_losses, val_losses, lrs, save_path="moving_mnist_zero_shot/results/training_curves.png"):
    """Plot training curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot losses
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss difference
    ax2 = axes[1]
    loss_diff = np.array(train_losses) - np.array(val_losses)
    ax2.plot(epochs, loss_diff, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Train - Val Loss')
    ax2.set_title('Overfitting Monitor')
    ax2.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax3 = axes[2]
    ax3.plot(epochs, lrs, 'orange', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    
    # Print summary
    if epochs:
        print(f"\nTraining Summary:")
        print(f"Latest Epoch: {epochs[-1]}")
        print(f"Latest Train Loss: {train_losses[-1]:.6f}")
        print(f"Latest Val Loss: {val_losses[-1]:.6f}")
        print(f"Best Val Loss: {min(val_losses):.6f} (Epoch {epochs[np.argmin(val_losses)]})")
        print(f"Current Learning Rate: {lrs[-1]:.6f}")
    
    return fig

if __name__ == "__main__":
    log_dir = "../work_dirs/convlstm_mmnist_m2"
    print(f"Looking for logs in: {Path(log_dir).absolute()}")
    epochs, train_losses, val_losses, lrs = parse_logs(log_dir)
    print(f"Found {len(epochs)} epochs of data")
    
    if epochs:
        save_path = "results/convlstm_training_curves.png"
        plot_curves(epochs, train_losses, val_losses, lrs, save_path=save_path)
        plt.show()
    else:
        print("No training data found in logs.")
