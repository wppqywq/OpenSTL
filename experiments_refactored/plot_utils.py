#!/usr/bin/env python3
"""
Unified plotting utilities for different representation types.

Three specialized plotters:
- PixelPlotter: Binary frames with single pred/gt markers
- HeatPlotter: Normalized heatmaps with markers
- CoordPlotter: Trajectory-only visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple


class BasePlotter:
    """Base class for representation-specific plotters."""
    
    def __init__(self, img_size: int = 32, pre_seq_length: int = 10):
        self.img_size = img_size
        self.pre_seq_length = pre_seq_length
        self.dpi = 150
        
    def plot_trajectory(self, 
                       true_coords: torch.Tensor,
                       pred_coords: torch.Tensor, 
                       sample_idx: int,
                       save_path: Optional[Path] = None,
                       valid_mask: Optional[torch.Tensor] = None) -> None:
        """Plot trajectory visualization with points at each frame."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Convert to numpy
        true_coords_np = true_coords.numpy()
        pred_coords_np = pred_coords.numpy()
        
        # Input trajectory (first pre_seq_length frames)
        input_coords = true_coords_np[:self.pre_seq_length]
        ax.plot(input_coords[:, 0], input_coords[:, 1], '-', 
               color='black', linewidth=2, label='Input', alpha=0.8)
        # Add points for each input frame
        ax.scatter(input_coords[:, 0], input_coords[:, 1], 
                  color='black', s=30, alpha=0.6, zorder=3)
        
        # Ground truth future trajectory (dashed line, semi-transparent)
        true_future = true_coords_np[self.pre_seq_length:]
        ax.plot(true_future[:, 0], true_future[:, 1], ':',
               color='blue', linewidth=2.5, label='Ground Truth', alpha=0.7)
        # Add points for each ground truth frame
        ax.scatter(true_future[:, 0], true_future[:, 1],
                  color='blue', s=30, alpha=0.8, zorder=3)
        
        # Predicted trajectory (only valid prefix, solid red)
        if valid_mask is not None:
            # use continuous prefix length determined by cumulative product
            prefix_len = int(valid_mask.float().cumprod(0).sum().item())
            if prefix_len > 0:
                pred_coords_np = pred_coords_np[:prefix_len]
                true_future = true_future[:prefix_len]
            else:
                pred_coords_np = pred_coords_np[:0]
                true_future = true_future[:0]
        ax.plot(pred_coords_np[:, 0], pred_coords_np[:, 1], '-',
               color='red', linewidth=2, label='Prediction', alpha=0.7)
        # Add points for each predicted frame
        ax.scatter(pred_coords_np[:, 0], pred_coords_np[:, 1],
                  color='red', s=30, alpha=0.8, zorder=3)
        
        # Format
        ax.set_xlim(0, self.img_size)
        ax.set_ylim(0, self.img_size)
        ax.invert_yaxis()
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'Sample {sample_idx + 1} Trajectory ({self.__class__.__name__.replace("Plotter", "")})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class PixelPlotter(BasePlotter):
    """Plotter for pixel/binary representation."""
    
    def plot_frames(self,
                   input_frames: torch.Tensor,
                   pred_frames: torch.Tensor,
                   true_coords: torch.Tensor,
                   pred_coords: torch.Tensor,
                   sample_idx: int,
                   save_path: Optional[Path] = None,
                   valid_mask: Optional[torch.Tensor] = None) -> None:
        """Plot frame grid for pixel representation."""
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        fig.suptitle(f'Sample {sample_idx + 1}: Pixel Input and Output', fontsize=14)
        
        # Input frames (raw binary)
        for t in range(10):
            if t < len(input_frames):
                frame = input_frames[t, 0].numpy()
                vmax = float(frame.max()) if frame.max() > 0 else 1.0
                axes[0, t].imshow(frame / (vmax if vmax > 0 else 1.0), cmap='gray', vmin=0, vmax=1)
            axes[0, t].set_title(f'Input {t+1}')
            axes[0, t].axis('off')
        
        # Predicted frames with markers
        for t in range(10):
            if t < len(pred_frames):
                # Apply sigmoid for display
                frame = torch.sigmoid(pred_frames[t, 0]).numpy()
                vmax = float(frame.max()) if frame.max() > 0 else 1.0
                axes[1, t].imshow(frame / (vmax if vmax > 0 else 1.0), cmap='gray', vmin=0, vmax=1)
                
                # Add single pred and gt markers (smaller)
                if (valid_mask is None or (t < len(valid_mask) and valid_mask[t])) and t < len(pred_coords):
                    pred_coord = pred_coords[t].numpy()
                    axes[1, t].plot(pred_coord[0], pred_coord[1], 'r+', 
                                  markersize=6, markeredgewidth=1, alpha=0.7, label='Prediction' if t == 0 else "_")
                
                if (self.pre_seq_length + t) < len(true_coords):
                    true_coord = true_coords[self.pre_seq_length + t].numpy()
                    axes[1, t].plot(true_coord[0], true_coord[1], 'gx',
                                  markersize=6, markeredgewidth=1, alpha=0.7, label='Ground Truth' if t == 0 else "_")
                
            axes[1, t].set_title(f'Pred {t+1}', fontsize=10)
            axes[1, t].axis('off')
        
        # Add legend inside the first prediction frame
        handles, labels = axes[1, 0].get_legend_handles_labels()
        if not handles:
            axes[1, 0].plot([], [], 'r+', label='Prediction')
            axes[1, 0].plot([], [], 'gx', label='Ground Truth')
        axes[1, 0].legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class HeatPlotter(BasePlotter):
    """Plotter for heatmap/Gaussian representation."""
    
    def plot_frames(self,
                   input_frames: torch.Tensor,
                   pred_frames: torch.Tensor,
                   true_coords: torch.Tensor,
                   pred_coords: torch.Tensor,
                   sample_idx: int,
                   save_path: Optional[Path] = None,
                   valid_mask: Optional[torch.Tensor] = None) -> None:
        """Plot frame grid for heatmap representation with raw output."""
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        fig.suptitle(f'Sample {sample_idx + 1}: Heatmap Input and Output', fontsize=14)
        
        # Input frames (show heatmaps)
        for t in range(10):
            if t < len(input_frames):
                frame = input_frames[t, 0].numpy()
                vmax = float(frame.max()) if frame.max() > 0 else 1.0
                axes[0, t].imshow(frame / (vmax if vmax > 0 else 1.0), cmap='gray', vmin=0, vmax=1.0)
            axes[0, t].set_title(f'Input {t+1}', fontsize=10)
            axes[0, t].axis('off')
        
        # Predicted frames - show raw heatmap output
        for t in range(10):
            if t < len(pred_frames):
                frame = pred_frames[t, 0].numpy()
                vmax = float(frame.max()) if frame.max() > 0 else 1.0
                im = axes[1, t].imshow(frame / (vmax if vmax > 0 else 1.0), cmap='gray', vmin=0, vmax=1.0)
                # markers only on valid frames
                if (valid_mask is None or (t < len(valid_mask) and valid_mask[t])) and t < len(pred_coords):
                    pred_coord = pred_coords[t].numpy()
                    axes[1, t].plot(pred_coord[0], pred_coord[1], 'r+',
                                  markersize=6, markeredgewidth=1, alpha=0.7)
                
                if (self.pre_seq_length + t) < len(true_coords):
                    true_coord = true_coords[self.pre_seq_length + t].numpy()
                    axes[1, t].plot(true_coord[0], true_coord[1], 'gx',
                                  markersize=6, markeredgewidth=1, alpha=0.7)
                
            axes[1, t].set_title(f'Pred {t+1}', fontsize=10)
            axes[1, t].axis('off')
        
        # Legend inside first pred frame (force even if no markers in panel)
        handles, labels = axes[1, 0].get_legend_handles_labels()
        if not handles:
            axes[1, 0].plot([], [], 'r+', label='Prediction')
            axes[1, 0].plot([], [], 'gx', label='Ground Truth')
        axes[1, 0].legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class CoordPlotter(BasePlotter):
    """Plotter for coordinate representation - trajectory only."""
    
    def plot_frames(self,
                   input_coords: torch.Tensor,
                   pred_coords: torch.Tensor,
                   true_coords: torch.Tensor,
                   sample_idx: int,
                   save_path: Optional[Path] = None) -> None:
        """For coord representation, we only plot trajectory."""
        # Redirect to trajectory plot
        self.plot_trajectory(true_coords, pred_coords, sample_idx, save_path)


def create_plotter(representation: str, img_size: int = 32, pre_seq_length: int = 10) -> BasePlotter:
    """Factory function to create appropriate plotter."""
    if representation == 'pixel':
        return PixelPlotter(img_size, pre_seq_length)
    elif representation == 'heat':
        return HeatPlotter(img_size, pre_seq_length)
    elif representation == 'coord':
        return CoordPlotter(img_size, pre_seq_length)
    else:
        raise ValueError(f"Unknown representation: {representation}")