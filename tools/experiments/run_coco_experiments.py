#!/usr/bin/env python
"""
Main experiment runner for COCO-Search18 eye tracking prediction
Supports both coordinate-based and heatmap-based experiments
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add OpenSTL to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser, load_config, update_config


class COCOSearchExperiment:
    """Experiment runner for COCO-Search18"""
    
    def __init__(self, args):
        self.args = args
        self.experiment_dir = Path(f"experiments/{args.ex_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment configuration
        self.save_experiment_config()
        
    def save_experiment_config(self):
        """Save experiment configuration for reproducibility"""
        config = {
            'method': self.args.method,
            'dataset': self.args.dataname,
            'config_file': self.args.config_file,
            'representation': getattr(self.args, 'representation', 'heatmap'),
            'dataset_config': getattr(self.args, 'dataset_config', 'short'),
            'timestamp': datetime.now().isoformat(),
            'device': self.args.device,
            'seed': self.args.seed
        }
        
        with open(self.experiment_dir / 'experiment_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def run_training(self):
        """Run training experiment"""
        print("="*60)
        print(f"Starting training experiment: {self.args.ex_name}")
        print(f"Method: {self.args.method}")
        print(f"Dataset: {self.args.dataname}")
        print("="*60)
        
        # Create experiment
        exp = BaseExperiment(self.args)
        
        # Train model
        exp.train()
        
        # Test model
        print("\nEvaluating on test set...")
        test_results = exp.test()
        
        # Save results
        self.save_results(test_results)
        
        return test_results
    
    def save_results(self, results):
        """Save experiment results"""
        results_file = self.experiment_dir / 'results.json'
        
        # Convert numpy values to Python types
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                clean_results[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                clean_results[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                clean_results[k] = int(v)
            else:
                clean_results[k] = v
        
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def analyze_results(self):
        """Analyze and visualize experiment results"""
        # Load training logs
        log_dir = Path(f"work_dirs/{self.args.ex_name}")
        
        if not log_dir.exists():
            print("No logs found for analysis")
            return
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training loss curve
        # Plot 2: Validation metrics
        # Plot 3: Coordinate error by timestep
        # Plot 4: Sample predictions
        
        # Save analysis
        analysis_path = self.experiment_dir / 'analysis.png'
        plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
        print(f"Analysis saved to: {analysis_path}")


def create_experiment_parser():
    """Create argument parser for experiments"""
    parser = argparse.ArgumentParser(description='COCO-Search18 Experiments')
    
    # Basic arguments
    parser.add_argument('-d', '--dataname', type=str, default='coco_search',
                       help='Dataset name')
    parser.add_argument('-m', '--method', type=str, default='SimVP',
                       help='Method name')
    parser.add_argument('-c', '--config_file', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--ex_name', type=str, required=True,
                       help='Experiment name')
    
    # Dataset specific arguments
    parser.add_argument('--representation', type=str, default='heatmap',
                       choices=['heatmap', 'coordinate'],
                       help='Data representation type')
    parser.add_argument('--dataset_config', type=str, default='short',
                       choices=['short', 'medium', 'standard', 'success_only'],
                       help='Dataset configuration')
    
    # Training arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--epoch', type=int, default=100,
                       help='Number of epochs')
    
    # Other arguments
    parser.add_argument('--test', action='store_true',
                       help='Test only mode')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite config values')
    parser.add_argument('--no_display_method_info', action='store_true',
                       help='Do not display method info')
    
    return parser


def run_comparison_experiments():
    """Run comparison experiments between different methods"""
    experiments = [
        {
            'name': 'simvp_heatmap',
            'method': 'SimVP',
            'config': 'configs/coco_search/simvp/SimVP_gSTA_M2.py',
            'representation': 'heatmap'
        },
        {
            'name': 'convlstm_heatmap',
            'method': 'ConvLSTM',
            'config': 'configs/coco_search/ConvLSTM_coco.py',
            'representation': 'heatmap'
        }
    ]
    
    results = {}
    
    for exp_config in experiments:
        print(f"\nRunning experiment: {exp_config['name']}")
        
        # Create args
        args = create_experiment_parser().parse_args([
            '-d', 'coco_search',
            '-m', exp_config['method'],
            '-c', exp_config['config'],
            '--ex_name', exp_config['name'],
            '--representation', exp_config['representation']
        ])
        
        # Update with default values
        default_values = default_parser()
        for attr, value in default_values.items():
            if not hasattr(args, attr):
                setattr(args, attr, value)
        
        # Load config
        config = load_config(args.config_file)
        args.__dict__.update(config)
        
        # Run experiment
        experiment = COCOSearchExperiment(args)
        results[exp_config['name']] = experiment.run_training()
    
    # Compare results
    print("\n" + "="*60)
    print("Experiment Comparison Results")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name}:")
        for metric, value in result.items():
            print(f"  {metric}: {value:.4f}")


def main():
    """Main entry point"""
    parser = create_experiment_parser()
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison experiments')
    
    args = parser.parse_args()
    
    if args.compare:
        run_comparison_experiments()
    else:
        # Update with default values
        default_values = default_parser()
        for attr, value in default_values.items():
            if not hasattr(args, attr):
                setattr(args, attr, value)
        
        # Load config
        config = load_config(args.config_file)
        args.__dict__.update(config)
        
        # Run single experiment
        experiment = COCOSearchExperiment(args)
        results = experiment.run_training()
        experiment.analyze_results()
        
        print("\nExperiment completed successfully!")
        print(f"Results saved in: experiments/{args.ex_name}/")


if __name__ == '__main__':
    main()