#!/usr/bin/env python
"""
Integration patch to add COCO-Search support to OpenSTL
Run this script to automatically add the necessary code to existing files
"""

import os

def patch_dataloader():
    """Add COCO-Search support to dataloader.py"""
    file_path = 'openstl/datasets/dataloader.py'
    
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'coco_search' in content:
        print(f"✅ {file_path} already patched")
        return True
    
    # Find the elif chain and add our case
    insertion_point = content.find("else:\n        raise ValueError(f'Dataname {dataname} is unsupported')")
    
    if insertion_point == -1:
        print(f"❌ Could not find insertion point in {file_path}")
        return False
    
    new_code = """    elif dataname == 'coco_search':
        from .dataloader_coco_search import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    """
    
    new_content = content[:insertion_point] + new_code + content[insertion_point:]
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Patched {file_path}")
    return True

def patch_parser():
    """Add COCO-Search to parser choices"""
    file_path = 'openstl/utils/parser.py'
    
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    if 'coco_search' in content:
        print(f"✅ {file_path} already patched")
        return True
    
    # Find the choices list and add coco_search
    old_choices = "'sevir_vis', 'sevir_ir069', 'sevir_ir107', 'sevir_vil']"
    new_choices = "'sevir_vis', 'sevir_ir069', 'sevir_ir107', 'sevir_vil', 'coco_search']"
    
    new_content = content.replace(old_choices, new_choices)
    
    if new_content == content:
        print(f"❌ Could not find choices list in {file_path}")
        return False
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Patched {file_path}")
    return True

def create_dataset_constants():
    """Create dataset constants file"""
    file_path = 'openstl/datasets/dataset_constant.py'
    
    content = '''# Dataset parameters for different datasets
dataset_parameters = {
    'coco_search': {
        'in_shape': (20, 1, 32, 32),
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'metrics': ['mse', 'mae']
    }
}
'''
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created {file_path}")
    return True

def patch_datasets_init():
    """Add import to datasets __init__.py"""
    file_path = 'openstl/datasets/__init__.py'
    
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    if 'dataloader_coco_search' in content:
        print(f"✅ {file_path} already patched")
        return True
    
    # Add import after existing imports
    import_line = "from .dataloader_coco_search import COCOSearchDataset\n"
    
    # Find first from import
    insertion_point = content.find("from .dataloader_human")
    if insertion_point != -1:
        new_content = content[:insertion_point] + import_line + content[insertion_point:]
        
        # Also add to __all__
        old_all = "'WeatherBenchDataset', 'SEVIRDataset'"
        new_all = "'WeatherBenchDataset', 'SEVIRDataset', 'COCOSearchDataset'"
        new_content = new_content.replace(old_all, new_all)
        
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Patched {file_path}")
        return True
    
    print(f"❌ Could not find insertion point in {file_path}")
    return False

def main():
    print("🔧 Applying OpenSTL Integration Patches for COCO-Search\n")
    
    patches = [
        ("Dataset constants", create_dataset_constants),
        ("Datasets __init__", patch_datasets_init),
        ("Dataloader", patch_dataloader),
        ("Parser", patch_parser),
    ]
    
    results = []
    for name, patch_func in patches:
        print(f"Applying {name} patch...")
        success = patch_func()
        results.append(success)
        print()
    
    print("="*50)
    print(f"Patches applied: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✅ All patches applied successfully!")
        print("\nNow run:")
        print("1. python debug_simvp_model.py")
        print("2. python train_simple_m2.py")
    else:
        print("❌ Some patches failed. Check the errors above.")

if __name__ == '__main__':
    main()