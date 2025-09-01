#!/usr/bin/env python3

import os
import sys
import json

# Ensure project root is on sys.path when running this file directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from experiments_refactored.datasets.position_dependent_gaussian import GaussianFieldGenerator
from experiments_refactored.datasets.config import *

def main():
    """Generate Gaussian field datasets using centralized config."""
    
    cfg = {
        'step_scale': 1.0,
        'mode': 'center_inward',
        'alpha': 0.5,
        'decay_length': 10.0,
        'lambda1_directed': 1.0,
        'lambda2_perpendicular': 0.25,
        'lambda_iso': 0.5,
        'bias_alpha': 0.5,
        'heatmap_sigma': HEATMAP_SIGMA,
        'img_size': IMG_SIZE,
        'sequence_length': SEQUENCE_LENGTH,
        'train_size': TRAIN_SIZE,
        'val_size': VAL_SIZE,
        'test_size': TEST_SIZE,
    }
    
    output_dir = "data/position_dependent_gaussian"
    generator = GaussianFieldGenerator(output_dir, cfg)
    generator.generate_datasets()

if __name__ == "__main__":
    main()