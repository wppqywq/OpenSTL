# Hybrid SimVP Encoder + MLP Regression Experiment

## Overview

This experiment implements a **hybrid architecture** that combines SimVP's powerful spatiotemporal encoder with a simple MLP head for direct coordinate regression. This approach fundamentally changes the task from sparse heatmap prediction to direct (x, y) coordinate regression, which is more efficient and potentially more accurate.

## Architecture

### Key Innovation: Task Transformation
- **Original SimVP**: Predicts entire video frames (32x32 heatmaps with sparse white dots)
- **Hybrid Model**: Directly regresses (x, y) coordinates of next fixation point

### Model Components

1. **SimVP Encoder**: 
   - Uses SimVP's gSTA modules for spatiotemporal feature extraction
   - Processes input video sequence: `(B, T=4, C=1, H=32, W=32)`
   - Outputs latent features: `(B, T=4, C_latent=64, H_latent=8, W_latent=8)`

2. **MLP Regression Head**:
   - Takes final timestep's latent state: `(B, 64*8*8=4096)`
   - 3-layer MLP: `4096 → 512 → 256 → 2`
   - Outputs: `(B, 2)` coordinates for next fixation

### Advantages
- **Direct**: No complex loss functions or sparse target handling
- **Efficient**: Single forward pass produces coordinate prediction
- **Simple**: L1 loss instead of focal/sparsity/concentration losses
- **Robust**: Regression task is more stable than sparse classification

## Files

- `model.py`: Hybrid model implementation (`SimVP_RegressionHead` class)
- `main.py`: Training script with coordinate regression pipeline
- `config.py`: Configuration parameters
- `test_model.py`: Model validation script
- `README.md`: This documentation

## Data Pipeline

### Input-Output Format
- **Input**: 4 consecutive frames showing fixation dots `(B, 4, 1, 32, 32)`
- **Target**: Coordinates of 5th frame's fixation `(B, 2)`

### Data Reuse
- Reuses data from `single_fixation_experiment`
- Converts sparse heatmaps to coordinates using `argmax`
- No new data generation required

## Training

### Loss Function
- **L1 Loss (Mean Absolute Error)**: `loss = |predicted_coords - target_coords|`
- More robust to outliers than MSE
- Direct optimization of pixel error

### Evaluation Metric
- **Mean Pixel Error**: Average L2 distance between predicted and target coordinates
- Directly comparable to previous experiments' results

### Expected Performance
- Target: Improve upon 5.12px mean error from previous approaches
- Hypothesis: Direct regression should achieve <3px error

## Usage

### Test Model
```bash
cd experiments/hybrid_encoder_mlp_experiment
python test_model.py
```

### Train Model
```bash
cd experiments/hybrid_encoder_mlp_experiment
python main.py
```

### Monitor Training
```bash
# View logs
tail -f logs/training_log_hybrid_regression.txt

# Check results
cat logs/final_results.txt
```

## Model Statistics

- **Total Parameters**: ~41.7M
- **Architecture**: SimVP encoder (40M+) + MLP head (~1.7M)
- **Input Shape**: `(B, 4, 1, 32, 32)`
- **Output Shape**: `(B, 2)`
- **Latent Dimensions**: `64 × 8 × 8 = 4096` features

## Key Improvements Over Previous Approaches

1. **Task Simplification**: Coordinate regression vs. sparse heatmap prediction
2. **Loss Simplification**: Single L1 loss vs. multi-component losses
3. **Direct Optimization**: Optimizes pixel error directly
4. **Architecture Efficiency**: Reuses proven SimVP encoder, minimal MLP head
5. **Training Stability**: Regression is more stable than sparse classification

## Expected Results

Based on the architecture design, we expect:
- **Faster Convergence**: Simpler task should converge faster
- **Better Accuracy**: Direct optimization of pixel error
- **More Stable Training**: No loss component balancing issues
- **Lower Memory Usage**: No need to store/predict full heatmaps

## Comparison to Baseline

| Approach | Task | Loss Function | Parameters | Expected Error |
|----------|------|---------------|------------|----------------|
| Original SimVP | Heatmap Prediction | Focal + Sparsity + Concentration | ~41M | 5.12px |
| **Hybrid Model** | **Coordinate Regression** | **L1 Loss** | **~41.7M** | **<3px** |

The hybrid approach represents a more direct and potentially more effective solution to the eye movement prediction problem. 