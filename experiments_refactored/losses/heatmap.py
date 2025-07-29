import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    """
    Mean Squared Error Loss for heatmap regression.
    
    This is a simple MSE loss that treats each pixel in the heatmap as 
    an independent regression target. Suitable for dense Gaussian representation.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize the MSELoss.
        
        Args:
            reduction (str): Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
        
    def forward(self, pred, target):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted heatmap.
            target (torch.Tensor): Target heatmap.
            
        Returns:
            torch.Tensor: Loss value.
        """
        return self.mse(pred, target)

class KLDivergenceLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss for heatmap regression.
    
    This loss treats heatmaps as probability distributions and measures
    the divergence between the predicted and target distributions.
    """
    
    def __init__(self, reduction='batchmean', eps=1e-10):
        """
        Initialize the KLDivergenceLoss.
        
        Args:
            reduction (str): Reduction method ('batchmean', 'sum', 'mean', 'none').
            eps (float): Small value to avoid log(0).
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        
    def forward(self, pred, target):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted heatmap.
            target (torch.Tensor): Target heatmap.
            
        Returns:
            torch.Tensor: Loss value.
        """
        # Normalize target and prediction to sum to 1
        # This ensures they are proper probability distributions
        
        # Handle zero-sum case for target
        target_sum = target.sum(dim=(2, 3), keepdim=True)
        target_sum = torch.where(target_sum == 0, torch.ones_like(target_sum), target_sum)
        target_normalized = target / target_sum
        
        # Handle zero-sum case for prediction
        pred_sum = pred.sum(dim=(2, 3), keepdim=True)
        pred_sum = torch.where(pred_sum == 0, torch.ones_like(pred_sum), pred_sum)
        pred_normalized = pred / pred_sum
        
        # Apply log to prediction for KL divergence
        log_pred = torch.log(pred_normalized + self.eps)
        
        # Calculate KL divergence
        return self.kl_div(log_pred, target_normalized)

class EarthMoverDistanceLoss(nn.Module):
    """
    Earth Mover Distance (EMD) Loss for heatmap regression.
    
    This loss treats heatmaps as probability distributions and calculates
    the minimum "work" required to transform one distribution into another.
    
    Note: This is a simplified approximation of EMD using center of mass distance,
    which is equivalent to 2-Wasserstein distance for single-peaked distributions
    like Gaussian fixation heatmaps.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize the EarthMoverDistanceLoss.
        
        Args:
            reduction (str): Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted heatmap.
            target (torch.Tensor): Target heatmap.
            
        Returns:
            torch.Tensor: Loss value.
        """
        B, C, H, W = pred.shape
        
        # Create coordinate grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=pred.device),
            torch.arange(W, dtype=torch.float32, device=pred.device),
            indexing='ij'
        )
        
        # Normalize target and prediction to sum to 1
        # This ensures they are proper probability distributions
        
        # Handle zero-sum case for target
        target_sum = target.sum(dim=(2, 3), keepdim=True)
        target_sum = torch.where(target_sum == 0, torch.ones_like(target_sum), target_sum)
        target_normalized = target / target_sum
        
        # Handle zero-sum case for prediction
        pred_sum = pred.sum(dim=(2, 3), keepdim=True)
        pred_sum = torch.where(pred_sum == 0, torch.ones_like(pred_sum), pred_sum)
        pred_normalized = pred / pred_sum
        
        # Calculate expected coordinates using normalized distributions directly
        # No need for additional softmax since pred_normalized is already normalized
        target_x = (target_normalized * x_grid).sum(dim=(2, 3))
        target_y = (target_normalized * y_grid).sum(dim=(2, 3))
        pred_x = (pred_normalized * x_grid).sum(dim=(2, 3))
        pred_y = (pred_normalized * y_grid).sum(dim=(2, 3))
        
        # Calculate squared Euclidean distance between expected coordinates
        squared_distance = (target_x - pred_x).pow(2) + (target_y - pred_y).pow(2)
        
        # Calculate Earth Mover Distance (2-Wasserstein distance)
        emd = torch.sqrt(squared_distance + 1e-6)  # Add small epsilon to avoid sqrt(0)
        
        # Apply reduction
        if self.reduction == 'mean':
            return emd.mean()
        elif self.reduction == 'sum':
            return emd.sum()
        else:  # 'none'
            return emd 