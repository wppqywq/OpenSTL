# Phase 2: Representation Study

## Summary
This phase tests whether dense representation (Gaussian heatmaps) enables better learning compared to sparse binary representation. The experiment investigates whether the representation format affects coordinate prediction accuracy in video sequences.

## How to Run
```bash
cd experiments/phase2_representation_study
python train_representation_comparison.py
```

### Input Data
- Same real eye movement coordinate data as Phase 1
- Converted to two different representation formats:
  - Sparse: Single binary pixel per frame
  - Dense: Gaussian heatmaps with varying sigma values

### Main Configuration and Parameters
- **Model**: SimVP with hid_S=64, hid_T=256, N_S=4, N_T=8
- **Training**: 20 epochs, batch_size=8, lr=1e-3, weight_decay=1e-5
- **Gaussian Parameters**: sigma values tested = [1.0, 1.5, 2.0, 2.5]
- **Coordinate Extraction**: Soft-argmax with temperature=1.0
- **Data Split**: 300 train, 50 val, 50 test samples

## Code Files and Functions

### train_representation_comparison.py
This file compares sparse vs dense representations through controlled experiments.

**load_real_data()**
- **Purpose**: Load coordinate data for representation comparison
- **Data Source**: Same .pt files as Phase 1 containing coordinates and fixation masks
- **Assumption**: Identical data ensures fair representation comparison

**create_sparse_representation()**
- **Purpose**: Create traditional sparse binary representation
- **Method**: Single white pixel (value=1.0) at exact coordinate location
- **Algorithm**: Direct coordinate-to-pixel mapping with bounds checking
- **Assumption**: Point-wise representation sufficient for location encoding

**create_dense_gaussian_representation()**
- **Purpose**: Create dense Gaussian heatmap representation
- **Principle**: Smooth spatial distribution around coordinate locations
- **Method**: 2D Gaussian kernel centered at coordinates
- **Formula**: gaussian = exp(-distance²/(2*sigma²))
- **Algorithm**: Vectorized computation using meshgrid for efficiency
- **Parameters**: sigma controls Gaussian width/smoothness
- **Assumption**: Smooth gradients provide better learning signal than sparse binary

**create_model()**
- **Purpose**: Initialize SimVP model identical to Phase 1
- **Architecture**: Same spatiotemporal encoder-decoder structure
- **Assumption**: Architecture differences should not confound representation comparison

**extract_coordinates_from_prediction()**
- **Purpose**: Convert model predictions back to coordinate space
- **Method**: Soft-argmax for differentiable coordinate extraction
- **Algorithm**: 
  1. Flatten prediction to probability distribution using softmax
  2. Compute weighted average of coordinate positions
  3. Temperature parameter controls sharpness of distribution
- **Principle**: Soft-argmax maintains differentiability while extracting spatial location
- **Assumption**: Probabilistic coordinate extraction more robust than hard maximum

**train_model_on_representation()**
- **Purpose**: Train model on specific representation format
- **Algorithm**: Standard PyTorch training loop with MSE loss
- **Optimization**: Adam optimizer with ReduceLROnPlateau scheduler
- **Validation**: Early stopping based on validation loss
- **Assumption**: MSE loss appropriate for both sparse and dense representations

**evaluate_representation()**
- **Purpose**: Evaluate coordinate prediction accuracy across representations
- **Metrics**: 
  - Mean coordinate error (Euclidean distance)
  - Standard deviation of errors
  - Accuracy at different pixel thresholds (1px, 3px, 5px)
  - Frame-level MSE
- **Method**: Compares predicted vs true coordinates in pixel space
- **Key Insight**: Coordinate accuracy more meaningful than pixel-level metrics for this task
- **Assumption**: Coordinate distance reflects prediction quality better than pixel classification

**run_real_representation_comparison()**
- **Purpose**: Main experimental function comparing all representations
- **Experimental Design**: 
  - Tests sparse representation vs multiple Gaussian sigma values
  - Uses identical training conditions across representations
  - Controlled comparison with same train/val/test splits
- **Analysis**: Compares coordinate accuracy improvement over sparse baseline
- **Output**: JSON results with accuracy metrics for each representation
- **Scientific Method**: Isolates representation as only variable, controls all other factors 