#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import os
import torch
from openstl.api import BaseExperiment
from openstl.utils import create_parser

def main():
    print("🚀 Starting Simple M2 Training with Fixed Config")
    
    # Check if working config exists
    working_config = 'configs/coco_search/simvp/SimVP_gSTA_M2_working.py'
    if not os.path.exists(working_config):
        print("❌ Working config not found. Run 'python deep_debug_simvp.py' first")
        return
    
    # Create basic args
    parser = create_parser()
    args = parser.parse_args([
        '-d', 'coco_search',
        '-c', working_config,
        '--ex_name', 'coco_search_m2_working',
        '--device', 'mps' if torch.backends.mps.is_available() else 'cpu'
    ])
    
    print(f"Using config: {working_config}")
    print(f"Device: {args.device}")
    
    try:
        exp = BaseExperiment(args)
        print("✅ Experiment created successfully")
        
        print("\n" + "="*50)
        print("Starting Training...")
        print("="*50)
        
        exp.train()
        
        print("\n" + "="*50) 
        print("Starting Testing...")
        print("="*50)
        
        exp.test()
        
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\nDebug info:")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()