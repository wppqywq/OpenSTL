# Phase 5: Direction-Magnitude Decoupling

## Summary
This phase resolves velocity undershooting through explicit separation of direction and magnitude learning. The experiment implements enhanced loss functions that decompose displacement prediction into independent direction and magnitude components, tested on both real coordinate data (Phase 3) and geometric patterns (Phase 4).

## How to Run
```bash
cd experiments/phase5_direction_magnitude_decoupling
python train_direction_magnitude_decoupling.py  # Phase 3 data
python train_with_phase4_data.py               # Geometric data  
python complete_method_comparison.py           # All methods comparison
python correct_model_visualization.py          # Model predictions
```

### Input Data
- **Phase 3 data**: Real eye movement coordinates (complex patterns)
- **Phase 4 data**: Synthetic geometric patterns (predictable motions)
- **Format**: Frame sequences converted to displacement vectors
- **Input**: 5-10 frames → Target: displacement vector (Δx, Δy)

### Main Configuration and Parameters
- **Model**: 3D CNN encoder + MLP displacement head
- **Training**: 25-30 epochs, batch_size=16, lr=1e-3, weight_decay=1e-5
- **Loss Methods**: L1 baseline, Direction-Magnitude Combined, Direction Emphasis
- **Direction-Magnitude Weights**: 
  - Combined: direction_weight=1.0, magnitude_weight=1.0
  - Emphasis: direction_weight=2.0, magnitude_weight=1.0
- **Architecture**: Conv3D(32,64) → AdaptiveAvgPool3D → Linear(256,128,64,2)

## Code Files and Functions

### train_direction_magnitude_decoupling.py
This file implements direction-magnitude decoupling on real coordinate data (Phase 3).

**DirectionMagnitudeLoss(nn.Module)**
- **Purpose**: Enhanced loss function with explicit direction-magnitude decomposition
- **Mathematical Principle**: 
  - Displacement vector d = |d| * (d/|d|) = magnitude * direction
  - Direction loss: 1 - cosine_similarity(pred_dir, true_dir)
  - Magnitude loss: smooth_l1_loss(pred_mag, true_mag)
- **Loss Modes**:
  - **l1_baseline**: Standard L1 loss on displacement vectors
  - **combined**: Weighted sum of direction and magnitude losses
  - **direction_only**: Focus only on direction learning
  - **magnitude_only**: Focus only on magnitude learning
- **Key Innovation**: Separates gradient flow for direction vs magnitude learning
- **Assumption**: Independent direction/magnitude learning reduces velocity undershooting

**get_components()**
- **Purpose**: Extract direction and magnitude components for analysis
- **Direction Calculation**: Unit vector = displacement / ||displacement||
- **Magnitude Calculation**: Euclidean norm ||displacement||
- **Metrics**: Direction similarity (cosine), magnitude ratio (predicted/true)
- **Numerical Stability**: Uses epsilon=1e-6 to handle zero displacements

**DirectionMagnitudeModel(nn.Module)**
- **Purpose**: Neural network for displacement prediction with 3D spatiotemporal processing
- **Architecture**:
  - **3D CNN Encoder**: Processes spatiotemporal input [B,T,C,H,W] → [B,C,T,H,W]
  - **Spatial-Temporal Convolution**: Conv3D with (3,3,3) kernels
  - **Feature Extraction**: 32→64 channels with ReLU activation
  - **Global Pooling**: AdaptiveAvgPool3D reduces to (1,4,4) spatial
  - **MLP Head**: 256→128→64→2 for displacement prediction
- **Assumption**: 3D convolution captures spatiotemporal dependencies for displacement

**load_real_data()**
- **Purpose**: Load Phase 3 coordinate data for direction-magnitude training
- **Data Source**: Same coordinate sequences used in Phase 3
- **Processing**: Converts to displacement vectors for enhanced training

**coordinates_to_frames()**
- **Purpose**: Convert coordinate sequences to binary frame sequences
- **Method**: Standard sparse representation with single white pixels
- **Assumption**: Binary frames provide sufficient spatial information

**calculate_displacement_vectors()**
- **Purpose**: Convert coordinate sequences to displacement vectors
- **Algorithm**: Δx(t) = x(t+1) - x(t) for each time step
- **Output**: Displacement targets for direction-magnitude training
- **Mathematical**: Provides velocity information for enhanced loss functions

**train_with_direction_magnitude_loss()**
- **Purpose**: Training loop with direction-magnitude loss functions
- **Training Strategy**: Standard PyTorch training with enhanced loss functions
- **Optimization**: Adam optimizer with ReduceLROnPlateau scheduling
- **Gradient Analysis**: Tracks direction similarity and magnitude ratios during training
- **Model Selection**: Saves best model based on validation loss
- **Scientific Rigor**: Real training without simulation or hardcoded results

**analyze_gradient_independence()**
- **Purpose**: Theoretical analysis of gradient flow independence
- **Mathematical Analysis**:
  - Computes gradients for L1 vs Direction-Magnitude losses
  - Measures gradient angle using cosine similarity
  - Independence criterion: gradient_angle > 45°
- **Key Insight**: Independent gradients enable separate learning of direction/magnitude
- **Scientific Value**: Provides theoretical foundation for enhanced loss function

**run_real_direction_magnitude_analysis()**
- **Purpose**: Main experimental function comparing all three methods
- **Methods Tested**: L1 baseline, Direction-Magnitude Combined, Direction Emphasis
- **Experimental Design**: Controlled comparison with identical data and training conditions
- **Evaluation**: Direction similarity, magnitude ratios, validation loss
- **Output**: Comprehensive JSON results with gradient independence analysis

### train_with_phase4_data.py
This file tests direction-magnitude decoupling on geometric patterns (Phase 4 data).

**DirectionMagnitudeLoss(nn.Module)**
- **Purpose**: Identical enhanced loss function as Phase 3 script
- **Consistency**: Same mathematical formulation ensures fair comparison

**DirectionMagnitudeModel(nn.Module)**
- **Purpose**: Same neural architecture for displacement prediction
- **Assumption**: Identical model ensures geometric vs real data comparison isolates data complexity

**generate_line_pattern(), generate_arc_pattern(), generate_bounce_pattern()**
- **Purpose**: Generate synthetic geometric patterns identical to Phase 4
- **Pattern Parameters**: Same ranges and physics as geometric validation
- **Assumption**: Predictable patterns should show clearer direction-magnitude benefits

**coordinates_to_frames(), calculate_displacement_vectors()**
- **Purpose**: Same data processing as Phase 3 for consistency
- **Method**: Identical frame and displacement conversion

**generate_geometric_datasets()**
- **Purpose**: Create large-scale geometric datasets for robust training
- **Dataset Size**: 600 samples (200 per pattern type)
- **Parameter Sampling**: Random parameter values within realistic ranges
- **Quality Control**: Boundary constraints and validity checking
- **Assumption**: Diverse geometric data enables robust evaluation

**train_with_geometric_data()**
- **Purpose**: Main experimental function for geometric pattern training
- **Comparison**: Tests all three methods on predictable geometric patterns
- **Hypothesis**: Geometric patterns should show clearer direction-magnitude benefits
- **Expected Results**: Better performance than real data due to predictability
- **Scientific Value**: Isolates method effectiveness from data complexity

### complete_method_comparison.py
This file creates comprehensive visualizations comparing all three methods.

**DirectionMagnitudeModel(nn.Module)**
- **Purpose**: Same model architecture for consistent visualization
- **Model Loading**: Loads pre-trained models for all three methods

**load_phase3_data(), generate_geometric_test_data()**
- **Purpose**: Load both real and geometric data for comprehensive comparison
- **Data Sources**: Phase 3 real coordinates and Phase 4 synthetic patterns

**predict_trajectory_iterative()**
- **Purpose**: Generate multi-step trajectory predictions using trained models
- **Algorithm**: 
  1. Predict single displacement step
  2. Update position: position += displacement
  3. Create new frame with updated position
  4. Shift frame window and repeat
- **Method**: Iterative prediction enables multi-step trajectory generation
- **Assumption**: Single-step displacement model can generate multi-step trajectories

**load_all_models()**
- **Purpose**: Load all three trained models for comparison
- **Methods**: L1 baseline, Direction-Magnitude, Direction Emphasis
- **Data Types**: Separate models for Phase 3 and Phase 4 data
- **Model Management**: Handles missing models gracefully

**visualize_phase3_all_methods(), visualize_phase4_all_methods()**
- **Purpose**: Create side-by-side comparison visualizations
- **Visualization Design**:
  - 3×3 grid showing all methods on multiple test samples
  - Color coding: true input (black), true future (gray), predictions (colored)
  - Error metrics: Mean coordinate error for each prediction
- **Analysis**: Visual comparison reveals method differences
- **Documentation**: Provides evidence of method performance differences

**create_method_performance_summary()**
- **Purpose**: Quantitative performance comparison across methods and datasets
- **Metrics**: Validation loss and direction similarity
- **Visualization**: Bar charts comparing method performance
- **Analysis**: Shows relative method effectiveness on different data types

### correct_model_visualization.py
This file creates proper model visualizations following experimental design.

**DirectionMagnitudeModel(nn.Module)**
- **Purpose**: Same model architecture ensuring visualization consistency

**visualize_phase3_predictions(), visualize_phase4_predictions()**
- **Purpose**: Create detailed prediction visualizations for each phase
- **Design Principle**: Phase 3 model on Phase 3 data, Phase 4 model on Phase 4 data
- **Visualization Format**: 3×3 grids showing trajectory predictions with error analysis
- **Scientific Rigor**: Proper experimental design avoiding cross-phase contamination

**predict_trajectory_iterative()**
- **Purpose**: Same iterative trajectory prediction as comparison script
- **Consistency**: Identical prediction method ensures comparable results

**visualize_frame_predictions_separate()**
- **Purpose**: Frame-by-frame visualization of model predictions
- **Analysis Components**:
  - True frame sequences showing ground truth
  - Predicted frame sequences with model outputs
  - Error overlays showing prediction accuracy
  - Frame indices and error metrics
- **Educational Value**: Shows how models perform at frame level
- **Documentation**: Provides detailed evidence of model prediction quality

**main()**
- **Purpose**: Generate complete visualization suite
- **Output Files**: 
  - Trajectory predictions for both phases
  - Frame-by-frame predictions
  - Comprehensive model evaluation visualizations
- **Scientific Documentation**: Provides visual evidence supporting direction-magnitude decoupling effectiveness 