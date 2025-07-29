# Phase 1: Loss Function Analysis

## Summary
This phase tests whether standard loss functions can handle extreme sparsity (1000:1 class imbalance) in video sequence prediction. The experiment compares different loss functions for predicting sparse events in video where only 0.1% of pixels are active.

## How to Run
```bash
cd experiments/phase1_loss_function_analysis
python train_loss_comparison.py
```

### Input Data
- Real eye movement coordinate data from `data/train_data.pt`, `data/val_data.pt`, `data/test_data.pt`
- Data format: sparse binary frames (32x32 pixels) with single white pixels marking fixation points
- Input: 10 consecutive frames as model input
- Target: Next 10 frames as prediction target

### Main Configuration and Parameters
- **Model**: SimVP with hidden dimensions hid_S=64, hid_T=256, N_S=4, N_T=8
- **Training**: 25 epochs, batch_size=8, lr=1e-3, weight_decay=1e-5
- **Loss Function Weights**: 
  - Focal Loss: alpha=0.25, gamma=2.0
  - Composite Loss: sparsity_weight=0.8, concentration_weight=1.5
  - Weighted MSE: pos_weight=100.0

## Code Files and Functions

### enhanced_loss_functions.py
This file implements specialized loss functions designed for extreme sparsity problems.

**FocalLoss(nn.Module)**
- **Purpose**: Address class imbalance using focal loss from Lin et al. (2017)
- **Principle**: Down-weights easy examples and focuses on hard negatives
- **Method**: Uses alpha-balanced focal loss with gamma parameter to modulate focusing strength
- **Formula**: FL(pt) = -α(1-pt)^γ log(pt)
- **Assumption**: Hard examples (misclassified) are more informative than easy examples

**SparsityLoss(nn.Module)**
- **Purpose**: Encourage sparse predictions through L1 regularization
- **Principle**: Combines reconstruction loss with sparsity penalty
- **Method**: MSE reconstruction + L1 penalty on predictions
- **Assumption**: Natural sparsity in predictions improves generalization

**ConcentrationLoss(nn.Module)**
- **Purpose**: Encourage focused predictions near target locations
- **Principle**: Penalizes predictions far from target center of mass
- **Method**: Calculates center of mass for targets, penalizes predictions proportional to distance squared
- **Algorithm**: Uses 2D grid coordinates to compute spatial distances
- **Assumption**: Predictions should be spatially concentrated around true target locations

**CompositeLoss(nn.Module)**
- **Purpose**: Combine focal, sparsity, and concentration losses based on successful training logs
- **Principle**: Multi-objective optimization balancing different aspects of sparse prediction
- **Method**: Weighted combination of three loss components
- **Weights**: focal_weight=1.0, sparsity_weight=0.8, concentration_weight=1.5 (from empirical results)
- **Assumption**: Different loss components address different failure modes in sparse prediction

**DiceLoss(nn.Module)**
- **Purpose**: Optimize overlap between prediction and target
- **Principle**: Based on Dice coefficient for segmentation tasks
- **Method**: 2*intersection/(union) with smooth factor for numerical stability
- **Assumption**: Overlap-based metrics better capture sparse region matching

**WeightedMSE(nn.Module)**
- **Purpose**: Simple reweighting approach for class imbalance
- **Principle**: Assign higher weights to positive pixels
- **Method**: Element-wise weighting of MSE loss
- **Assumption**: Higher penalty on missing positive pixels improves recall

**get_loss_function()**
- **Purpose**: Factory function for loss function selection
- **Method**: Dictionary-based selection with default fallback to MSE

### train_loss_comparison.py
This file conducts real training experiments comparing different loss functions.

**load_real_training_data()**
- **Purpose**: Load actual training data from archived experiments
- **Data Source**: Loads .pt files containing coordinates and fixation masks
- **Assumption**: Real data provides authentic evaluation of loss function performance

**create_sparse_frames()**
- **Purpose**: Convert coordinate data to sparse binary frames
- **Method**: Places single white pixels at coordinate locations
- **Algorithm**: Iterates through batch and time dimensions, sets pixel=1.0 at valid coordinates
- **Assumption**: Single pixel representation captures essential spatial information

**create_model()**
- **Purpose**: Initialize SimVP model for video prediction
- **Architecture**: Spatiotemporal prediction with hierarchical hidden states
- **Parameters**: in_shape=(10,1,32,32) for 10-frame input sequences
- **Assumption**: SimVP architecture suitable for sparse temporal pattern learning

**train_model_with_loss()**
- **Purpose**: Train model with specific loss function using real training loop
- **Algorithm**: Standard PyTorch training with gradient clipping and learning rate scheduling
- **Optimization**: Adam optimizer with ReduceLROnPlateau scheduler
- **Validation**: Tracks best validation loss for model selection
- **Assumption**: Standard deep learning training procedures apply to sparse prediction

**evaluate_model()**
- **Purpose**: Comprehensive evaluation of trained models
- **Metrics**: White pixel recall, precision, F1-score, test loss
- **Method**: Converts continuous predictions to binary using 0.5 threshold
- **Key Metric**: White pixel recall measures ability to detect sparse events
- **Assumption**: Binary classification metrics meaningful for sparse prediction evaluation

**run_real_loss_function_analysis()**
- **Purpose**: Main experimental function comparing all loss functions
- **Method**: Trains separate models for each loss function with identical conditions
- **Evaluation**: Systematic comparison on held-out test set
- **Output**: JSON results file with comprehensive metrics
- **Scientific Integrity**: All results from actual model training, no simulation 