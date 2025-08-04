

## Directory Structure

```
experiments_refactored/
├── datasets/                # Data loading and preprocessing
│   ├── eye_gauss.py         # Real eye movement data
│   └── geom_simple.py       # Synthetic geometric patterns
├── models/                  # Neural network models
│   └── simvp_base.py        # SimVP model with task-specific heads
├── losses/                  # Loss functions
│   ├── focal_bce.py         # Pixel-based losses (FocalBCE, WeightedBCE, FocalTversky)
│   ├── heatmap.py           # Heatmap-based losses (MSE, KL, EMD)
│   ├── vector.py            # Vector-based losses (Huber, PolarDecoupled, UncertaintyWeighted)
│   └── factory.py           # Loss function factory
├── results/                 # Training results and visualizations
├── train.py                 # Unified training script
├── eval.py                  # Evaluation script
├── run_sweep.py             # Script to run experiments defined in sweep.yaml
├── sweep.yaml               # Experiment definitions and hyperparameter search spaces
└── README.md                # This file
```

## Key Components

### Datasets

- **EyeGaussDataset**: Loads real eye movement data from the original experiments.
- **GeometricDataset**: Generates synthetic geometric patterns (line, arc, bounce) for controlled experiments.

### Representations

- **Pixel**: Binary pixel representation (sparse, one-hot encoding).
- **Heat**: Dense Gaussian heatmap representation.
- **Coord**: Coordinate displacement vector representation.

### Models

- **SimVPWithTaskHead**: SimVP model with task-specific heads for different representations.

### Loss Functions

- **Pixel-based**: FocalBCELoss, WeightedBCELoss, FocalTverskyLoss
- **Heatmap-based**: MSELoss, KLDivergenceLoss, EarthMoverDistanceLoss
- **Vector-based**: HuberLoss, PolarDecoupledLoss, UncertaintyWeightedLoss

## Core Experiments

The `sweep.yaml` file defines 9 core experiments that systematically investigate the impact of different representations and loss functions:

1. **P1_Gauss_Pixel_FocalBCE**: Replicate Phase 1 extreme imbalance baseline
2. **P2_Gauss_Heat_KL**: Verify 'heatmap cheating' reproduction
3. **P3_Gauss_Coord_Huber**: Expose velocity bias
4. **P4_Gauss_Coord_PolarDecoupled**: Random direction & velocity fix
5. **P5_Geom_Pixel_FocalBCE**: Exclude real data complexity impact
6. **P6_Geom_Heat_KL**: Check if still cheating
7. **P7_Geom_Coord_Huber**: Basic shrinkage vs. predictability
8. **P8_Geom_Coord_PolarDecoupled**: Verify gradient conflict again
9. **P9_Gauss_Pixel_WeightedBCE**: Alternative to focal loss for extreme imbalance

## Usage

### Training

```bash
# Train with eye_gauss dataset, pixel representation, and focal_bce loss
python train.py --data eye_gauss --repr pixel --loss focal_bce

# Train with geom_simple dataset, heat representation, and kl loss
python train.py --data geom_simple --repr heat --loss kl --sigma 1.5

# Train with eye_gauss dataset, coord representation, and polar_decoupled loss
python train.py --data eye_gauss --repr coord --loss polar_decoupled --w_dir 1.0 --w_mag 1.0
```

### Evaluation

```bash
# Evaluate a model checkpoint
python eval.py --checkpoint results/P1_Gauss_Pixel_FocalBCE/best_checkpoint.pth
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

## Hyperparameter Sweeps

The `sweep.yaml` file also defines hyperparameter search spaces for:

- **sigma**: Standard deviation of Gaussian heatmap [1.0, 1.5, 2.0]
- **gamma**: Focusing parameter for focal losses [2.0, 3.0, 4.0]
- **w_dir**: Direction weight for polar decoupled loss [0.1, 0.5, 1.0, 2.0]
- **w_mag**: Magnitude weight for polar decoupled loss [0.1, 0.5, 1.0, 2.0]

## Key Metrics

For coordinate representation, the following metrics are calculated:

- **pixel_error**: L2 distance between predicted and target coordinates
- **velocity_ratio**: Ratio of predicted to target velocity magnitude
- **direction_cos**: Cosine similarity between predicted and target direction vectors

For pixel representation, the following metrics are calculated:

- **precision**: Precision of binary pixel prediction
- **recall**: Recall of binary pixel prediction
- **f1**: F1 score of binary pixel prediction

## Results

Training and evaluation results are saved in the `results/` directory, organized by experiment name. Each experiment directory contains:

- **checkpoints**: Model checkpoints
- **logs**: Training logs
- **visualizations**: Visualizations of predictions
- **metrics.json**: Evaluation metrics 
