import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss) for coordinate regression.
    
    This loss is less sensitive to outliers than MSE and provides
    more stable gradients for large errors.
    """
    
    def __init__(self, delta=1.0, reduction='mean'):
        """
        Initialize the HuberLoss.
        
        Args:
            delta (float): Threshold parameter that determines the boundary
                          between L1 and L2 behavior.
            reduction (str): Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted displacement vectors of shape (B, 2).
            target (torch.Tensor): Target displacement vectors of shape (B, 2).
            
        Returns:
            torch.Tensor: Loss value.
        """
        # Calculate absolute difference
        abs_diff = torch.abs(pred - target)
        
        # Apply Huber loss formula
        loss = torch.where(
            abs_diff < self.delta,
            0.5 * abs_diff.pow(2),
            self.delta * (abs_diff - 0.5 * self.delta)
        )
        
        # Sum over vector components (x, y)
        loss = loss.sum(dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class PolarDecoupledLoss(nn.Module):
    """
    Polar-Decoupled Loss for coordinate regression.
    
    This loss decouples the direction and magnitude components of displacement vectors,
    allowing for separate treatment of these aspects.
    """
    
    def __init__(self, direction_weight=1.0, magnitude_weight=1.0, reduction='mean'):
        """
        Initialize the PolarDecoupledLoss.
        
        Args:
            direction_weight (float): Weight for the direction component.
            magnitude_weight (float): Weight for the magnitude component.
            reduction (str): Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted displacement vectors of shape (B, 2).
            target (torch.Tensor): Target displacement vectors of shape (B, 2).
            
        Returns:
            torch.Tensor: Loss value.
        """
        # Get direction and magnitude components
        direction_loss, magnitude_loss = self.get_components(pred, target)
        
        # Combine losses with weights
        combined_loss = self.direction_weight * direction_loss + self.magnitude_weight * magnitude_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:  # 'none'
            return combined_loss
    
    def get_components(self, pred, target):
        """
        Decompose the loss into direction and magnitude components.
        
        Args:
            pred (torch.Tensor): Predicted displacement vectors of shape (B, 2).
            target (torch.Tensor): Target displacement vectors of shape (B, 2).
            
        Returns:
            tuple: (direction_loss, magnitude_loss)
        """
        # Calculate magnitudes
        pred_magnitude = torch.norm(pred, dim=1)
        target_magnitude = torch.norm(target, dim=1)
        
        # Calculate magnitude loss (L1)
        magnitude_loss = torch.abs(pred_magnitude - target_magnitude)
        
        # Calculate direction loss (1 - cosine similarity)
        # Handle zero vectors to avoid NaN
        zero_mask = (pred_magnitude == 0) | (target_magnitude == 0)
        
        if zero_mask.any():
            # For zero vectors, set direction loss to 1 (maximum)
            direction_loss = torch.ones_like(magnitude_loss)
            
            # Calculate cosine similarity for non-zero vectors
            non_zero_mask = ~zero_mask
            if non_zero_mask.any():
                pred_normalized = F.normalize(pred[non_zero_mask], dim=1)
                target_normalized = F.normalize(target[non_zero_mask], dim=1)
                cosine_sim = torch.sum(pred_normalized * target_normalized, dim=1)
                direction_loss[non_zero_mask] = 1.0 - cosine_sim
        else:
            # All vectors are non-zero
            pred_normalized = F.normalize(pred, dim=1)
            target_normalized = F.normalize(target, dim=1)
            cosine_sim = torch.sum(pred_normalized * target_normalized, dim=1)
            direction_loss = 1.0 - cosine_sim
        
        return direction_loss, magnitude_loss

class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-Weighted Multitask Loss for coordinate regression.
    
    This loss learns to balance the direction and magnitude components
    by predicting task-specific uncertainties.
    
    Reference:
        "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
        by Kendall et al.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize the UncertaintyWeightedLoss.
        
        Args:
            reduction (str): Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.reduction = reduction
        
        # Initialize log variances (uncertainties) as learnable parameters
        # Start with equal weighting (log(1) = 0)
        self.log_var_direction = nn.Parameter(torch.zeros(1))
        self.log_var_magnitude = nn.Parameter(torch.zeros(1))
        
    def forward(self, pred, target):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted displacement vectors of shape (B, 2).
            target (torch.Tensor): Target displacement vectors of shape (B, 2).
            
        Returns:
            torch.Tensor: Loss value.
        """
        # Get direction and magnitude components using PolarDecoupledLoss
        polar_loss = PolarDecoupledLoss(reduction='none')
        direction_loss, magnitude_loss = polar_loss.get_components(pred, target)
        
        # Calculate precision (inverse variance) terms
        precision_direction = torch.exp(-self.log_var_direction)
        precision_magnitude = torch.exp(-self.log_var_magnitude)
        
        # Calculate weighted losses
        weighted_direction_loss = precision_direction * direction_loss + self.log_var_direction
        weighted_magnitude_loss = precision_magnitude * magnitude_loss + self.log_var_magnitude
        
        # Combine losses
        combined_loss = weighted_direction_loss + weighted_magnitude_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:  # 'none'
            return combined_loss 