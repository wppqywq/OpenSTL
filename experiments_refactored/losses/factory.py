import torch
import torch.nn as nn
from experiments_refactored.losses.focal_bce import FocalBCELoss, WeightedBCELoss, FocalTverskyLoss
from experiments_refactored.losses.heatmap import MSELoss, KLDivergenceLoss, EarthMoverDistanceLoss
from experiments_refactored.losses.vector import HuberLoss, PolarDecoupledLoss, UncertaintyWeightedLoss

def get_loss(name, **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        name (str): Name of the loss function.
        **kwargs: Additional arguments for the loss function.
        
    Returns:
        nn.Module: The requested loss function.
        
    Raises:
        ValueError: If the loss name is not recognized.
    """
    # Pixel-based losses (for binary pixel representation)
    if name == 'weighted_bce':
        return WeightedBCELoss(
            pos_weight=kwargs.get('pos_weight', 1000.0),
            reduction=kwargs.get('reduction', 'mean')
        )
    elif name == 'focal_bce':
        return FocalBCELoss(
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2.0),
            reduction=kwargs.get('reduction', 'mean')
        )
    elif name == 'focal_tversky':
        return FocalTverskyLoss(
            alpha=kwargs.get('alpha', 0.7),
            beta=kwargs.get('beta', 0.3),
            gamma=kwargs.get('gamma', 0.75),
            smooth=kwargs.get('smooth', 1e-6),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    # Heatmap-based losses (for dense Gaussian representation)
    elif name == 'mse':
        return MSELoss(
            reduction=kwargs.get('reduction', 'mean')
        )
    elif name == 'kl':
        return KLDivergenceLoss(
            reduction=kwargs.get('reduction', 'batchmean'),
            eps=kwargs.get('eps', 1e-10)
        )
    elif name == 'emd':
        return EarthMoverDistanceLoss(
            reduction=kwargs.get('reduction', 'mean')
        )
    
    # Vector-based losses (for coordinate displacement representation)
    elif name == 'huber':
        return HuberLoss(
            delta=kwargs.get('delta', 1.0),
            reduction=kwargs.get('reduction', 'mean')
        )
    elif name == 'polar_decoupled':
        return PolarDecoupledLoss(
            direction_weight=kwargs.get('direction_weight', 1.0),
            magnitude_weight=kwargs.get('magnitude_weight', 1.0),
            reduction=kwargs.get('reduction', 'mean'),
            eps=kwargs.get('eps', 1e-6)
        )
    elif name == 'uncertainty_weighted':
        return UncertaintyWeightedLoss(
            reduction=kwargs.get('reduction', 'mean')
        )
    
    # Default PyTorch losses
    elif name == 'l1':
        return nn.L1Loss(reduction=kwargs.get('reduction', 'mean'))
    elif name == 'l2':
        return nn.MSELoss(reduction=kwargs.get('reduction', 'mean'))
    elif name == 'smooth_l1':
        return nn.SmoothL1Loss(beta=kwargs.get('beta', 1.0), reduction=kwargs.get('reduction', 'mean'))
    
    else:
        raise ValueError(f"Unknown loss function: {name}")

def get_loss_for_representation(representation, **kwargs):
    """
    Get the default loss function for a given representation type.
    
    Args:
        representation (str): Representation type ('pixel', 'heat', or 'coord').
        **kwargs: Additional arguments for the loss function.
        
    Returns:
        nn.Module: The default loss function for the representation.
        
    Raises:
        ValueError: If the representation is not recognized.
    """
    if representation == 'pixel':
        return get_loss('focal_bce', **kwargs)
    elif representation == 'heat':
        return get_loss('kl', **kwargs)
    elif representation == 'coord':
        return get_loss('huber', **kwargs)
    else:
        raise ValueError(f"Unknown representation: {representation}")

def get_all_losses_for_representation(representation):
    """
    Get all available loss functions for a given representation type.
    
    Args:
        representation (str): Representation type ('pixel', 'heat', or 'coord').
        
    Returns:
        dict: Dictionary mapping loss names to loss functions.
        
    Raises:
        ValueError: If the representation is not recognized.
    """
    if representation == 'pixel':
        return {
            'weighted_bce': get_loss('weighted_bce'),
            'focal_bce': get_loss('focal_bce'),
            'focal_tversky': get_loss('focal_tversky')
        }
    elif representation == 'heat':
        return {
            'mse': get_loss('mse'),
            'kl': get_loss('kl'),
            'emd': get_loss('emd')
        }
    elif representation == 'coord':
        return {
            'huber': get_loss('huber'),
            'polar_decoupled': get_loss('polar_decoupled'),
            'uncertainty_weighted': get_loss('uncertainty_weighted')
        }
    else:
        raise ValueError(f"Unknown representation: {representation}") 