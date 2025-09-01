
## Directory Structure

```
experiments_refactored/
├── datasets/                # Data loading and preprocessing
│   ├── gauss.py             # Position-dependent Gaussian data
│   └── geom_unified.py      # Unified geometric patterns
├── models/                  # Neural network models
│   └── simvp_base.py        # SimVP model with task-specific heads
├── losses/                  # Loss functions
│   ├── focal_bce.py         # Pixel-based losses (FocalBCE, WeightedBCE, FocalTversky)
│   ├── heatmap.py           # Heatmap-based losses (MSE, KL, EMD)
│   ├── vector.py            # Vector-based losses (Huber, PolarDecoupled, UncertaintyWeighted)
│   └── factory.py           # Loss function factory
├── results/                 # Training results and visualizations
├── train.py                 # Unified training script
├── eval_unified.py          # Unified evaluation script
├── eval_all.py              # Batch evaluation script
├── plot_utils.py            # Representation-aware plotting utilities
├── generate_datasets.py     # Dataset generation and freezing
├── sweep.yaml               # Experiment definitions and hyperparameter search spaces
└── README.md                # This file
```

## Key Components

### Datasets

- **PositionDependentGaussianDataset**: Loads real eye movement data from the original experiments.
- **UnifiedGeometricDataset**: Generates synthetic geometric patterns (line, arc, bounce) for controlled experiments.

### Representations

- **Pixel**: Binary pixel representation (sparse, one-hot encoding).
- **Heat**: Dense Gaussian heatmap representation.
- **Coord**: Coordinate displacement vector representation.

### Models

- **SimVPWithTaskHead**: SimVP model with task-specific heads for different representations.

### Loss Functions

- **Pixel-based**: FocalBCELoss, WeightedBCELoss, FocalTverskyLoss, DiceBCELoss
- **Heatmap-based**: MSELoss, WeightedMSELoss, KLDivergenceLoss, EarthMoverDistanceLoss
- **Vector-based**: HuberLoss, PolarDecoupledLoss, UncertaintyWeightedLoss, L1CosineLoss

## Core Experiments

The `sweep.yaml` file defines 18 core experiments that systematically investigate the impact of different representations and loss functions:

### Gauss Dataset Experiments (lr=0.001)
1. **Full_gauss_pixel_WeightedBCE**: Weighted BCE for sparse pixel representation
2. **Full_gauss_pixel_FocalBCE**: Focal BCE for class imbalance
3. **Full_gauss_pixel_DiceBCE**: Dice BCE combination
4. **Full_gauss_heat_KL**: KL divergence for Gaussian heatmaps
5. **Full_gauss_heat_WeightedMSE**: Weighted MSE for heatmap regression
6. **Full_gauss_heat_EMD**: Earth Mover Distance for spatial distribution matching
7. **Full_gauss_coord_Huber**: Huber loss for robust coordinate regression
8. **Full_gauss_coord_Polar**: Polar decoupled loss for direction/magnitude separation
9. **Full_gauss_coord_L1Cosine**: L1 + Cosine similarity for coordinate prediction

### Geom_simple Dataset Experiments (lr=0.0005)
10. **Full_geom_pixel_WeightedBCE**: Weighted BCE for geometric data
11. **Full_geom_pixel_FocalBCE**: Focal BCE for geometric data
12. **Full_geom_pixel_DiceBCE**: Dice BCE for geometric data
13. **Full_geom_heat_KL**: KL divergence for geometric heatmaps
14. **Full_geom_heat_WeightedMSE**: Weighted MSE for geometric heatmaps
15. **Full_geom_heat_EMD**: EMD for geometric heatmaps
16. **Full_geom_coord_Huber**: Huber loss for geometric coordinates
17. **Full_geom_coord_Polar**: Polar decoupled for geometric coordinates
18. **Full_geom_coord_L1Cosine**: L1+Cosine for geometric coordinates

## Usage

### Dataset Generation

```bash
# Generate and freeze all datasets for reproducible experiments
python generate_datasets.py
```

### Training

```bash
# Train with gauss dataset, pixel representation, and focal_bce loss
python train.py --data gauss --repr pixel --loss focal_bce

# Train with geom_simple dataset, heat representation, and kl loss
python train.py --data geom_simple --repr heat --loss kl --sigma 1.5

# Train with gauss dataset, coord representation, and polar_decoupled loss
python train.py --data gauss --repr coord --loss polar_decoupled --w_dir 1.0 --w_mag 1.0
```

### Evaluation

```bash
# Evaluate a single model checkpoint
python eval_unified.py --checkpoint results/P1_Gauss_Pixel_FocalBCE/best_checkpoint.pth

# Evaluate all experiments
python eval_all.py
```

### Running Experiments

```bash
# Run all experiments
python run_sweep.py --all

# Run specific experiment by name
python run_sweep.py --exp P1_Gauss_Pixel_FocalBCE

# Run specific experiment by index
python run_sweep.py --idx 0

# Run hyperparameter sweep for specific experiment
python run_sweep.py --exp P4_Gauss_Coord_PolarDecoupled --sweep --param w_dir
```

## Key Metrics

For coordinate representation, the following metrics are calculated:

- **displacement_error_full**: L2 distance between predicted and target coordinates (full sequence)
- **displacement_error_3**: L2 distance for first 3 prediction steps
- **displacement_error_6**: L2 distance for first 6 prediction steps
- **avg_cosine_similarity**: Cosine similarity between predicted and target direction vectors
- **avg_magnitude_ratio**: Ratio of predicted to target velocity magnitude

For pixel representation, the following metrics are calculated:

- **precision**: Precision of binary pixel prediction
- **recall**: Recall of binary pixel prediction
- **f1**: F1 score of binary pixel prediction

## Results

Training and evaluation results are saved in the `results/` directory, organized by experiment name. Each experiment directory contains:

- **checkpoints**: Model checkpoints
- **logs**: Training logs
