# Gaussian Enhanced Eye Movement Prediction

## Overview

This experiment optimizes input representation for eye movement prediction using SimVP by replacing sparse single-pixel dots with **Gaussian blobs** and **history trails**. This approach helps CNN layers "understand" the data better without modifying the core model architecture.

## Key Improvements

### 1. Gaussian Blurring
- Replace single pixel dots with 2D Gaussian distributions
- Configurable standard deviation (σ = 2.0 by default)
- Provides gradient information for CNN kernels to learn from

### 2. History Trails
- Encode previous fixation locations with exponential decay
- Shows last 3 frames with γ = 0.75 decay factor
- Gives explicit motion direction and velocity information

## Quick Start

```bash
# Navigate to experiment directory
cd experiments/gaussian_enhanced_experiment

# Generate enhanced data
python generate_data.py

# Train model
python train_gaussian_simvp.py

# Evaluate and compare with baseline
python evaluate_gaussian.py
```

## Configuration

Key parameters in `config.py`:

```python
# Gaussian blob parameters
gaussian_sigma = 2.0        # Standard deviation for blobs
gaussian_normalize = True   # Normalize to [0,1] range

# History trail parameters  
enable_history_trails = True    # Enable/disable history encoding
history_length = 3             # Number of previous frames (T=3)
history_decay_gamma = 0.75     # Exponential decay factor
history_min_intensity = 0.1    # Minimum intensity threshold
```

## Expected Results

Based on the design improvements:

- **Better CNN feature extraction**: Gaussian blobs provide learnable gradients
- **Motion awareness**: History trails give explicit velocity information  
- **Faster convergence**: Richer input representation aids training
- **Improved accuracy**: Better coordinate prediction than sparse baselines

## File Structure

```
gaussian_enhanced_experiment/
├── config.py                 # Configuration parameters
├── generate_data.py          # Enhanced data generation
├── train_gaussian_simvp.py   # Training script
├── evaluate_gaussian.py      # Evaluation and comparison
├── data/                     # Generated datasets
│   ├── train_data.pt
│   ├── val_data.pt
│   └── test_data.pt
├── models/                   # Saved model checkpoints
├── logs/                     # Training logs  
├── results/                  # Evaluation outputs
└── README.md                 # This file
```

## Implementation Details

### Gaussian Blob Generation

```python
def generate_gaussian_blob(center_pos, img_size, sigma=2.0):
    # Create coordinate grids
    y_coords = torch.arange(img_size, dtype=torch.float32)
    x_coords = torch.arange(img_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Calculate Gaussian: G(x,y) = exp(-((x-μx)² + (y-μy)²) / (2σ²))
    distance_sq = (xx - x_center)**2 + (yy - y_center)**2
    gaussian = torch.exp(-distance_sq / (2 * sigma**2))
    
    return gaussian / gaussian.max()  # Normalize to [0,1]
```

### History Trail Encoding

```python
def render_gaussian_frame_with_history(coords_sequence, current_t, img_size):
    frame = torch.zeros(img_size, img_size)
    
    # Add history trail with exponential decay
    history_start = max(0, current_t - config.history_length)
    for t in range(history_start, current_t):
        history_age = current_t - t
        intensity = config.history_decay_gamma ** history_age
        if intensity >= config.history_min_intensity:
            # Generate blob with reduced intensity
            blob = generate_gaussian_blob(coords_sequence[t], img_size, config.gaussian_sigma)
            frame = torch.maximum(frame, blob * intensity)
    
    # Add current fixation with full intensity
    current_blob = generate_gaussian_blob(coords_sequence[current_t], img_size, config.gaussian_sigma)
    frame = torch.maximum(frame, current_blob)
    
    return frame
```

### Enhanced Loss Function

The training uses a modified loss function optimized for Gaussian inputs:

- **Focal loss**: Adapted for continuous Gaussian targets
- **Sparsity loss**: Modified to allow local concentration
- **MSE loss**: Direct similarity between Gaussian distributions  
- **Coordinate loss**: Soft-argmax (center of mass) guidance
- **Concentration loss**: Encourages sharp peaks

## Comparison with Baseline

The evaluation script compares against:

1. **Single fixation baseline**: Original sparse pixel approach
2. **Repeat last frame**: Simple baseline prediction
3. **Multiple coordinate extraction methods**: Argmax vs center of mass

Metrics include:
- Pixel accuracy at multiple thresholds (1px, 2px, 3px, 5px)
- Temporal error evolution across prediction frames
- Error distribution analysis
- Direct frame-by-frame comparison plots

## Hyperparameter Selection

### Gaussian Sigma (σ)
- **σ = 1.5-2.0**: Good starting range
- **Principle**: Large enough for CNN kernel receptive field (3x3, 5x5)
- **Constraint**: Small enough to avoid confusion between nearby fixations

### History Length (T)  
- **T = 3-4**: Optimal for velocity/acceleration inference
- **Too small**: Insufficient motion context
- **Too large**: Noisy long-term dependencies

### Decay Factor (γ)
- **γ = 0.75**: Exponential memory decay 
- **Alternative**: Linear decay for different memory models
- **Principle**: Recent frames more important than distant ones

## Code Standards Compliance

This experiment follows the established patterns from `single_fixation_experiment`:

-  Reuses existing `single_fixation` coordinate sampling logic
-  Minimal changes to core architecture  
-  Same evaluation metrics for fair comparison
-  English-only code and comments
-  Clean file organization under `experiments/`
-  Configuration-driven parameters
-  Comprehensive logging and evaluation

## Future Extensions

Potential improvements:
- **Multi-scale Gaussians**: Variable σ based on fixation importance
- **Attention-weighted history**: Learn optimal trail weighting  
- **Adaptive trail length**: Dynamic history based on motion patterns
- **Multi-channel inputs**: Separate channels for current vs history 