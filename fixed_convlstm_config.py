#!/usr/bin/env python

import os
import torch

def check_convlstm_parameters():
    """Check what parameters ConvLSTM actually needs"""
    try:
        from openstl.models import ConvLSTM_Model
        import inspect
        
        # Get the signature of ConvLSTM_Model
        sig = inspect.signature(ConvLSTM_Model.__init__)
        print("ConvLSTM_Model parameters:")
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                print(f"  {param_name}: {param}")
        
        return True
    except Exception as e:
        print(f"Error checking ConvLSTM: {e}")
        return False

def create_working_convlstm_config():
    """Create a working ConvLSTM config based on OpenSTL parameters"""
    
    # Check existing ConvLSTM configs in OpenSTL
    convlstm_config = '''method = 'ConvLSTM'

# dataset parameters
dataname = 'coco_search'
data_root = './data/coco_search'
in_shape = (20, 1, 64, 64)
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

# ConvLSTM model parameters - based on OpenSTL defaults
num_layers = 4
num_hidden = [64, 64, 64, 64]
filter_size = 5
stride = 1
layer_norm = True

# training parameters
lr = 1e-3
batch_size = 4
val_batch_size = 4
epoch = 5
sched = 'onecycle'

# evaluation metrics
metrics = ['mse', 'mae']
'''
    
    config_dir = 'configs/coco_search/convlstm'
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'ConvLSTM_fixed.py')
    
    with open(config_path, 'w') as f:
        f.write(convlstm_config)
    
    print(f"✅ Created fixed ConvLSTM config: {config_path}")
    return config_path

def test_convlstm_model():
    """Test ConvLSTM model creation"""
    try:
        from openstl.models import ConvLSTM_Model
        from openstl.datasets.dataloader_coco_search import COCOSearchDataset
        
        # Test dataset
        dataset = COCOSearchDataset(
            data_root='./data/coco_search',
            is_training=True,
            in_shape=(20, 1, 64, 64),
            pre_seq_length=10,
            aft_seq_length=10
        )
        print(f"✅ Dataset: {len(dataset)} samples")
        
        # Test model with correct parameters
        model = ConvLSTM_Model(
            num_layers=4,
            num_hidden=[64, 64, 64, 64],
            height=64,
            width=64,
            channels=1,
            filter_size=5,
            stride=1,
            layer_norm=True
        )
        print("✅ ConvLSTM model created")
        
        # Test forward pass
        x, y = dataset[0]
        x = x.unsqueeze(0)  # Add batch dimension
        print(f"Input shape: {x.shape}")
        
        with torch.no_grad():
            output = model(x)
        
        print(f"✅ Forward pass successful: {x.shape} -> {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ ConvLSTM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_baseline():
    """Create a very simple baseline model"""
    
    simple_config = '''method = 'SimVP'

# dataset parameters
dataname = 'coco_search'
data_root = './data/coco_search'
in_shape = (20, 3, 64, 64)  # Try 3 channels like RGB
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

# model parameters - minimal
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'  # Try IncepU instead of gSTA
hid_S = 64
hid_T = 256
N_T = 4
N_S = 4

# training parameters
lr = 1e-3
batch_size = 2
val_batch_size = 2
epoch = 3
drop_path = 0.0
sched = 'onecycle'

# evaluation metrics
metrics = ['mse', 'mae']
'''
    
    config_path = 'configs/coco_search/simvp/SimVP_IncepU_simple.py'
    with open(config_path, 'w') as f:
        f.write(simple_config)
    
    print(f"✅ Created simple baseline config: {config_path}")
    return config_path

def update_dataloader_for_rgb():
    """Update dataloader to support 3-channel output"""
    
    # This is a quick patch to make coordinates work with 3-channel models
    patch_content = '''
    def _coords_to_spatial_rgb(self, coords, H, W):
        """Convert coordinate sequence to 3-channel spatial representation"""
        T = coords.shape[0]
        spatial = torch.zeros(T, 3, H, W)  # RGB channels
        
        for t in range(T):
            x, y = coords[t]
            h_idx = max(0, min(int(y * (H - 1)), H-1))
            w_idx = max(0, min(int(x * (W - 1)), W-1))
            
            # Use all 3 channels for robustness
            spatial[t, 0, h_idx, w_idx] = 1.0  # R
            spatial[t, 1, h_idx, w_idx] = 1.0  # G  
            spatial[t, 2, h_idx, w_idx] = 1.0  # B
            
        return spatial
'''
    
    print("💡 Consider updating dataloader for 3-channel support if needed")

def main():
    print("🔧 Fixing ConvLSTM Configuration\n")
    
    # Check ConvLSTM parameters
    print("=== Checking ConvLSTM Parameters ===")
    check_convlstm_parameters()
    
    print("\n=== Creating Fixed ConvLSTM Config ===")
    config_path = create_working_convlstm_config()
    
    print("\n=== Testing ConvLSTM ===")
    convlstm_works = test_convlstm_model()
    
    if convlstm_works:
        print("\n🎉 ConvLSTM is working!")
        
        # Create training script
        train_script = f'''#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import torch
from openstl.api import BaseExperiment
from openstl.utils import create_parser

def main():
    parser = create_parser()
    args = parser.parse_args([
        '-d', 'coco_search',
        '-c', '{config_path}',
        '--ex_name', 'coco_search_convlstm_working',
        '--device', 'mps' if torch.backends.mps.is_available() else 'cpu'
    ])
    
    print("🚀 Starting ConvLSTM training for eye tracking...")
    
    try:
        exp = BaseExperiment(args)
        exp.train()
        exp.test()
        print("🎉 Training completed!")
    except Exception as e:
        print(f"❌ Error: {{e}}")

if __name__ == '__main__':
    main()
'''
        
        with open('train_convlstm.py', 'w') as f:
            f.write(train_script)
        
        print("✅ Created training script: train_convlstm.py")
        print("\n🚀 Now run: python train_convlstm.py")
        
    else:
        print("\n❌ ConvLSTM still not working")
        print("Creating alternative approaches...")
        
        # Try alternative configs
        create_simple_baseline()
        update_dataloader_for_rgb()
        
        print("\nTry these alternatives:")
        print("1. Fix the ConvLSTM import/parameter issues") 
        print("2. Use the IncepU variant of SimVP")
        print("3. Update dataloader for 3-channel support")

if __name__ == '__main__':
    main()