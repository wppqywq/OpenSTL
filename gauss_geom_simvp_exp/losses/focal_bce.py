import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross Entropy Loss.
    
    This loss is designed for binary classification tasks with class imbalance.
    It down-weights well-classified examples and focuses on hard examples.
    
    Reference:
        "Focal Loss for Dense Object Detection" by Lin et al.
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize the FocalBCELoss.
        
        Args:
            alpha (float): Weighting factor for the positive class.
            gamma (float): Focusing parameter. Higher values give more weight to hard examples.
            reduction (str): Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted logits.
            target (torch.Tensor): Binary target tensor.
            
        Returns:
            torch.Tensor: Loss value.
        """
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Calculate binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Calculate focal weights
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        # Apply focal weights to BCE loss
        loss = focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.
    
    This loss applies a higher weight to the positive class to address class imbalance.
    """
    
    def __init__(self, pos_weight=1000.0, reduction='mean'):
        """
        Initialize the WeightedBCELoss.
        
        Args:
            pos_weight (float): Weight for the positive class.
            reduction (str): Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted logits.
            target (torch.Tensor): Binary target tensor.
            
        Returns:
            torch.Tensor: Loss value.
        """
        # Use built-in weighted BCE loss
        loss = F.binary_cross_entropy_with_logits(
            pred, target, 
            pos_weight=torch.tensor(self.pos_weight, device=pred.device),
            reduction=self.reduction
        )
        
        return loss

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss.
    
    This loss combines the Tversky index with focal modulation.
    It's especially useful for highly imbalanced segmentation tasks.
    
    Reference:
        "A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation"
        by Abraham & Khan
    """
    
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6, reduction='mean'):
        """
        Initialize the FocalTverskyLoss.
        
        Args:
            alpha (float): Weight for false positives.
            beta (float): Weight for false negatives.
            gamma (float): Focusing parameter.
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted logits.
            target (torch.Tensor): Binary target tensor.
            
        Returns:
            torch.Tensor: Loss value.
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten the tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate true positives, false positives, and false negatives
        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()
        
        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Calculate Focal Tversky loss
        focal_tversky = (1 - tversky).pow(self.gamma)
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_tversky.mean()
        elif self.reduction == 'sum':
            return focal_tversky.sum()
        else:  # 'none'
            return focal_tversky 