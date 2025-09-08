import torch
import torch.nn as nn
import torch.nn.functional as F

class EarthMoverDistanceLoss(nn.Module):
    """
    Earth Mover Distance (Wasserstein) loss for heatmap predictions.
    Simplified implementation using MSE on cumulative distributions.
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Calculate EMD between predicted and target heatmaps.
        
        Args:
            pred: Predicted heatmap [B, T, C, H, W]
            target: Target heatmap [B, T, C, H, W]
        """
        # Flatten spatial dimensions
        B, T, C, H, W = pred.shape
        pred_flat = pred.view(B, T, C, -1)  # [B, T, C, H*W]
        target_flat = target.view(B, T, C, -1)
        
        # Normalize to probability distributions
        pred_prob = F.softmax(pred_flat, dim=-1)
        target_prob = F.softmax(target_flat, dim=-1)
        
        # Sort both distributions and compute MSE on sorted values
        # This is a simplified approximation of EMD
        pred_sorted, _ = torch.sort(pred_prob, dim=-1)
        target_sorted, _ = torch.sort(target_prob, dim=-1)
        
        # Calculate cumulative distributions
        pred_cumsum = torch.cumsum(pred_sorted, dim=-1)
        target_cumsum = torch.cumsum(target_sorted, dim=-1)
        
        # MSE on cumulative distributions approximates EMD
        loss = F.mse_loss(pred_cumsum, target_cumsum, reduction=self.reduction)
        
        return loss