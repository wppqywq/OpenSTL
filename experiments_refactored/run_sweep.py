#!/usr/bin/env python3
"""
Script to run experiments defined in sweep.yaml.

This script loads the sweep configuration and runs the specified experiments.

Example usage:
    # Run all experiments
    python run_sweep.py

    # Run specific experiment by name
    python run_sweep.py --exp P1_Gauss_Pixel_FocalBCE

    # Run specific experiment by index
    python run_sweep.py --idx 0
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run experiments defined in sweep.yaml')
    
    # Experiment selection arguments
    parser.add_argument('--exp', type=str, default=None,
                        help='Name of experiment to run')
    parser.add_argument('--idx', type=int, default=None,
                        help='Index of experiment to run')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    parser.add_argument('--sweep', action='store_true',
                        help='Run hyperparameter sweep for specified experiment')
    parser.add_argument('--param', type=str, default=None,
                        help='Parameter to sweep (e.g., sigma, gamma, w_dir, w_mag)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--sweep_file', type=str, default='sweep.yaml',
                        help='Path to sweep configuration file')
    
    args = parser.parse_args()
    
    # Check that exactly one of --exp, --idx, or --all is specified
    if sum([args.exp is not None, args.idx is not None, args.all]) != 1:
        parser.error('Exactly one of --exp, --idx, or --all must be specified')
    
    # Check that --param is specified if --sweep is specified
    if args.sweep and args.param is None:
        parser.error('--param must be specified if --sweep is specified')
    
    return args

def load_sweep_config(sweep_file: str) -> Dict[str, Any]:
    """
    Load sweep configuration from YAML file.
    
    Args:
        sweep_file (str): Path to sweep configuration file.
        
    Returns:
        dict: Sweep configuration.
    """
    with open(sweep_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_experiment_by_name(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """
    Get experiment by name.
    
    Args:
        config (dict): Sweep configuration.
        name (str): Name of experiment.
        
    Returns:
        dict: Experiment configuration.
    """
    for exp in config['experiments']:
        if exp['name'] == name:
            return exp
    
    raise ValueError(f"Experiment '{name}' not found in sweep configuration")

def get_experiment_by_index(config: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Get experiment by index.
    
    Args:
        config (dict): Sweep configuration.
        idx (int): Index of experiment.
        
    Returns:
        dict: Experiment configuration.
    """
    if idx < 0 or idx >= len(config['experiments']):
        raise ValueError(f"Experiment index {idx} out of range (0-{len(config['experiments'])-1})")
    
    return config['experiments'][idx]

def build_command(exp: Dict[str, Any], common_params: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Build command to run experiment.
    
    Args:
        exp (dict): Experiment configuration.
        common_params (dict): Common parameters for all experiments.
        output_dir (str): Output directory.
        
    Returns:
        list: Command to run experiment.
    """
    cmd = ['python', 'train.py']
    
    # Add experiment-specific parameters
    cmd.extend(['--data', exp['data']])
    cmd.extend(['--repr', exp['repr']])
    cmd.extend(['--model', exp['model']])
    cmd.extend(['--loss', exp['loss']])
    
    # Add loss-specific parameters
    if exp['loss'] == 'focal_bce' and 'alpha' in exp and 'gamma' in exp:
        cmd.extend(['--alpha', str(exp['alpha'])])
        cmd.extend(['--gamma', str(exp['gamma'])])
    elif exp['loss'] == 'weighted_bce' and 'pos_weight' in exp:
        cmd.extend(['--pos_weight', str(exp['pos_weight'])])
    elif exp['loss'] == 'huber' and 'delta' in exp:
        cmd.extend(['--delta', str(exp['delta'])])
    elif exp['loss'] == 'polar_decoupled' and 'w_dir' in exp and 'w_mag' in exp:
        cmd.extend(['--w_dir', str(exp['w_dir'])])
        cmd.extend(['--w_mag', str(exp['w_mag'])])
    
    # Add representation-specific parameters
    if exp['repr'] == 'heat' and 'sigma' in exp:
        cmd.extend(['--sigma', str(exp['sigma'])])
    
    # Add common parameters
    for k, v in common_params.items():
        cmd.extend([f'--{k}', str(v)])
    
    # Add output directory
    cmd.extend(['--output_dir', output_dir])
    
    # Add experiment name
    cmd.extend(['--exp_name', exp['name']])
    
    return cmd

def run_experiment(cmd: List[str]) -> int:
    """
    Run experiment.
    
    Args:
        cmd (list): Command to run experiment.
        
    Returns:
        int: Return code.
    """
    print(f"Running command: {' '.join(cmd)}")
    
    # Run command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print output in real-time
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end='')
    
    # Wait for process to finish
    process.wait()
    
    return process.returncode

def run_sweep(exp: Dict[str, Any], param: str, sweep_values: List[Any], 
              common_params: Dict[str, Any], output_dir: str) -> List[int]:
    """
    Run hyperparameter sweep for experiment.
    
    Args:
        exp (dict): Experiment configuration.
        param (str): Parameter to sweep.
        sweep_values (list): Values to sweep over.
        common_params (dict): Common parameters for all experiments.
        output_dir (str): Output directory.
        
    Returns:
        list: Return codes.
    """
    return_codes = []
    
    for value in sweep_values:
        # Create copy of experiment configuration
        exp_copy = exp.copy()
        
        # Update parameter value
        exp_copy[param] = value
        
        # Update experiment name
        exp_copy['name'] = f"{exp['name']}_{param}_{value}"
        
        # Build command
        cmd = build_command(exp_copy, common_params, output_dir)
        
        # Run experiment
        return_code = run_experiment(cmd)
        return_codes.append(return_code)
    
    return return_codes

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load sweep configuration
    config = load_sweep_config(args.sweep_file)
    
    # Get common parameters
    common_params = config.get('common', {})
    
    # Get experiment(s) to run
    if args.exp is not None:
        experiments = [get_experiment_by_name(config, args.exp)]
    elif args.idx is not None:
        experiments = [get_experiment_by_index(config, args.idx)]
    else:  # args.all
        experiments = config['experiments']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'sweep_{timestamp}.log'
    
    # Log sweep configuration
    with open(log_file, 'w') as f:
        f.write(f"Sweep started at: {datetime.now()}\n\n")
        f.write(f"Sweep configuration:\n{yaml.dump(config, default_flow_style=False)}\n\n")
        f.write(f"Running experiments: {[exp['name'] for exp in experiments]}\n\n")
    
    # Run experiments
    results = []
    
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"Running experiment: {exp['name']}")
        print(f"Description: {exp.get('description', 'No description')}")
        print(f"{'='*80}\n")
        
        # Record start time
        start_time = time.time()
        
        # Initialize variables that might not be set in some conditions
        sweep_values = []
        return_codes = []
        return_code = 0
        
        if args.sweep:
            # Get sweep values
            sweep_values = config['sweeps'].get(args.param, {}).get('values', [])
            if not sweep_values:
                raise ValueError(f"No sweep values found for parameter '{args.param}'")
            
            # Run sweep
            return_codes = run_sweep(exp, args.param, sweep_values, common_params, args.output_dir)
            
            # Record result
            result = {
                'name': exp['name'],
                'param': args.param,
                'values': sweep_values,
                'return_codes': return_codes,
                'duration': time.time() - start_time
            }
        else:
            # Build command
            cmd = build_command(exp, common_params, args.output_dir)
            
            # Run experiment
            return_code = run_experiment(cmd)
            
            # Record result
            result = {
                'name': exp['name'],
                'return_code': return_code,
                'duration': time.time() - start_time
            }
        
        results.append(result)
        
        # Log result
        with open(log_file, 'a') as f:
            f.write(f"Experiment: {exp['name']}\n")
            f.write(f"Duration: {result['duration']:.2f} seconds\n")
            if args.sweep:
                f.write(f"Parameter: {args.param}\n")
                f.write(f"Values: {sweep_values}\n")
                f.write(f"Return codes: {return_codes}\n")
            else:
                f.write(f"Return code: {return_code}\n")
            f.write("\n")
    
    # Log summary
    with open(log_file, 'a') as f:
        f.write(f"\nSweep completed at: {datetime.now()}\n")
        f.write(f"Total duration: {sum([r['duration'] for r in results]):.2f} seconds\n")
    
    print(f"\nSweep completed. Log saved to: {log_file}")

if __name__ == '__main__':
    main() 