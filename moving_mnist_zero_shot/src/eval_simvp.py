import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('/Users/apple/git/neuro/OpenSTL')
from openstl.models import SimVP_Model

# Add current dir for config
from pathlib import Path as _Path
CURRENT_DIR = _Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import config as cfg


def load_sequences(variant_name="baseline", max_seqs=3):
    variant_dir = Path(cfg.VARIANTS_ROOT) / variant_name
    sequences = []
    for seq_file in sorted(variant_dir.glob("seq*.npy"))[:max_seqs]:
        seq = np.load(seq_file)  # (T, H, W)
        sequences.append(seq)
    return sequences


def evaluate_model(model, sequences, device, context_frames=10):
    model.eval()
    results = []
    
    with torch.no_grad():
        for idx, seq in enumerate(sequences):
            T, H, W = seq.shape
            if T < context_frames + 1:
                continue
            
            # Prepare input: (1, T_in, 1, H, W) 
            input_frames = seq[:context_frames]
            input_tensor = torch.from_numpy(input_frames).float().unsqueeze(0).unsqueeze(2)
            input_tensor = input_tensor.to(device) / 255.0  # Normalize
            
            # Model prediction
            pred = model(input_tensor)  # Should output (1, T_out, 1, H, W)
            pred = pred.squeeze().cpu().numpy() * 255.0  # Denormalize
            
            # Compare with ground truth
            gt_frames = seq[context_frames:context_frames+pred.shape[0]]
            mse = np.mean((pred - gt_frames) ** 2)
            
            results.append({
                "seq_idx": idx,
                "mse": float(mse),
                "pred_frames": pred.shape[0],
            })
    
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint_path = "models/simvp_method_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model (simplified, may need adjustment based on checkpoint structure)
    model = SimVP_Model(
        in_shape=(10, 1, 64, 64),  # (T_in, C, H, W)
        hid_S=64,
        hid_T=256,
        N_S=4,
        N_T=8,
        model_type='gSTA',
    )
    
    # Load weights (handle 'model.' prefix in checkpoint)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    
    # Load test sequences
    sequences = load_sequences("baseline", max_seqs=3)
    
    # Evaluate
    results = evaluate_model(model, sequences, device)
    
    # Save results
    out_path = Path(cfg.OUTPUT_ROOT) / "eval" / "simvp_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump({"model": "SimVP", "results": results}, f, indent=2)
    
    print(str(out_path))
    for r in results:
        print(f"Seq {r['seq_idx']}: MSE = {r['mse']:.2f}")


if __name__ == "__main__":
    main()
