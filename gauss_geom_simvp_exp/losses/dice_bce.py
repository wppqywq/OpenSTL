import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    """
    Combination of Dice loss and Binary Cross Entropy loss.
    """
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6, reduction='mean'):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def dice_loss(self, pred, target):
        """Calculate Dice loss."""
        pred = torch.sigmoid(pred)  # Apply sigmoid to logits
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # Dice loss is 1 - Dice coefficient
        return 1.0 - dice_coeff
    
    def forward(self, pred, target):
        """Forward pass."""
        # Calculate individual losses
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        # Combine losses
        total_loss = self.dice_weight * dice + self.bce_weight * bce
        
        return total_loss