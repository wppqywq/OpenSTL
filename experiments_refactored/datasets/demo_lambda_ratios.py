#!/usr/bin/env python3
"""
Demo script to visualize the effect of lambda1/lambda2 ratios on covariance matrix ellipticity.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from .position_dependent_gaussian import build_cov_matrix

def visualize_covariance_ellipses():
    """Visualize how different lambda ratios affect the shape of the covariance matrix."""
    
    # Test different lambda ratios
    lambda_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    theta = 0  # No rotation for simplicity
    lambda1 = 2.0  # Fixed major axis
    
    fig, axes = plt.subplots(1, len(lambda_ratios), figsize=(15, 3))
    fig.suptitle('Effect of lambda2/lambda1 ratio on covariance matrix shape', fontsize=14)
    
    for i, ratio in enumerate(lambda_ratios):
        lambda2 = lambda1 * ratio
        cov_matrix = build_cov_matrix(theta, lambda1, lambda2)
        
        # Generate points from this distribution
        n_points = 1000
        points = torch.distributions.MultivariateNormal(
            torch.zeros(2), cov_matrix
        ).sample((n_points,))
        
        # Plot the points
        axes[i].scatter(points[:, 0], points[:, 1], alpha=0.6, s=10)
        axes[i].set_title(f'Ratio: {ratio:.1f}\n(λ₂/λ₁ = {ratio})')
        axes[i].set_xlim(-4, 4)
        axes[i].set_ylim(-4, 4)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('lambda_ratio_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Lambda ratio explanation:")
    print("- Ratio closer to 1.0 = more circular distribution")
    print("- Ratio closer to 0.0 = more elliptical distribution")
    print("- Current settings: short steps (0.4), long steps (0.3)")

if __name__ == "__main__":
    visualize_covariance_ellipses() 