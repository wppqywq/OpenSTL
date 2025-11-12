#!/usr/bin/env python
"""Train SimVP_gSTA on Moving MNIST using MacBook M2."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openstl.api import BaseExperiment
from openstl.utils import create_parser
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == '__main__':
    # Create parser and set arguments for M2 Mac
    parser = create_parser()
    
    # Parse with our custom settings
    args = parser.parse_args([
        '--dataname', 'mmnist',
        '--method', 'SimVP',
        '--model_type', 'gSTA',
        '--config_file', 'configs/mmnist/simvp/SimVP_gSTA.py', 
        '--ex_name', 'simvp_gsta_mmnist_m2',
        '--batch_size', '32', 
        '--val_batch_size', '32', 
        '--epoch', '200', 
        '--log_step', '20',
        '--device', 'mps',  # here
        '--num_workers', '7',  # Increased workers
        '--seed', '42',
        '--lr', '0.005',
        '--sched', 'onecycle',  # 
        '--ckpt_path', 'work_dirs/simvp_gsta_mmnist_m2/checkpoints/last.ckpt',  # Resume 
    ])

    print(f"Resume from: {args.ckpt_path if os.path.exists(args.ckpt_path) else 'scratch'}")
    
    # Add callbacks for early stopping and better checkpointing
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='work_dirs/simvp_gsta_mmnist_m2/checkpoints',
        filename='epoch{epoch:02d}-val_loss{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    # Initialize experiment
    exp = BaseExperiment(args)
    
    # Add callbacks to trainer
    if hasattr(exp, 'trainer') and exp.trainer is not None:
        exp.trainer.callbacks.extend([early_stop_callback, checkpoint_callback])
    
    exp.train()
    mse = exp.test()
    print(f"Test MSE: {mse}")
