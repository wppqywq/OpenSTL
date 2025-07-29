# Eye Movement Simulation with SimVP

This project implements eye movement prediction using the SimVP model for spatiotemporal sequence prediction.

## Project Structure

```
baseline_simulation/
├── train.py              # Main training script (clean, robust implementation)
├── config.py             # Configuration parameters
├── simulation.py         # Data generation script
├── analysis_notebook.py  # Complete analysis and visualization
├── CODE_LOG.md           # Development history and iterations
├── models/               # Trained model files
├── results/              # Generated outputs (GIFs, plots, comparisons)
├── data/                 # Training and test datasets
├── legacy/               # Historical implementations and experiments
└── README.md            # This file
```

## Quick Start

### 1. Generate Data
```bash
python simulation.py
```

### 2. Train Model
```bash
# Train with default settings
python train.py

# Train with custom name and epochs
python train.py --model_name my_model --epochs 100

# Resume from checkpoint
python train.py --resume models/my_model_checkpoint.pth
```

### 3. Analyze Results
```bash
python analysis_notebook.py
```

## Configuration

All parameters are in `config.py`:

- **Data Generation**: Sample sizes, sequence length, spatial parameters
- **Training**: Batch size, learning rate, epochs, early stopping
- **Model**: SimVP architecture configuration
- **Loss Weights**: Multi-component robust loss balancing

## Key Features

### Robust Loss Framework
- **Focal Loss**: Direct heatmap supervision
- **Sparsity Loss**: Prevents uniform outputs
- **Concentration Loss**: Ensures peak activation at targets
- **KL Divergence**: Maintains distribution similarity

### Training Features
- Automatic checkpoint saving
- Early stopping with patience
- Resume from checkpoint capability
- Model path management

### Analysis Tools
- Comprehensive model evaluation
- Training curve visualization
- Prediction GIF generation
- Performance comparison tables

## Best Model Performance

Current best: **5.894px** average pixel error (Robust Sharp configuration)

## Development History

See `CODE_LOG.md` for complete development iterations and technical discoveries.

## Legacy Experiments

Historical implementations available in `legacy/`:
- Loss function iterations
- Training script variants
- Analysis and inspection tools

## Results

Generated outputs stored in `results/`:
- Model comparison GIFs
- Training curves
- Performance tables
- Prediction visualizations

## Dependencies

- PyTorch
- OpenSTL framework
- matplotlib
- pandas
- numpy

## Usage Examples

### Custom Loss Weights
```python
# Edit config.py
LOSS_WEIGHTS = {
    'focal': 1.0,
    'sparsity': 0.3,
    'concentration': 1.5,
    'kl': 0.1
}
```

### Model Evaluation
```python
# Use analysis_notebook.py functions
results = evaluate_all_models()
create_comparison_plots()
generate_prediction_gifs()
```

### Training Monitoring
```bash
# Check training progress
tail -f models/*_training_log.txt

# Resume interrupted training
python train.py --resume models/checkpoint.pth
```

This implementation provides a clean, reproducible framework for eye movement prediction research with comprehensive analysis tools and robust training procedures. 