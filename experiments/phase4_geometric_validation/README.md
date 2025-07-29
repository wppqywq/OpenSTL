# Phase 4: Geometric Validation

## Summary
This phase tests models on predictable geometric patterns (lines, arcs, bounces) to isolate systematic biases from data complexity. The experiment uses synthetic geometric data to validate whether velocity undershooting is model-inherent rather than due to data complexity.

## How to Run
```bash
cd experiments/phase4_geometric_validation
python test_geometric_patterns.py
python plot_trajectory_examples.py
python visualize_geometric_predictions.py
```

### Input Data
- **Synthetic geometric patterns**: Line, Arc, and Bounce motions
- **Pattern Parameters**:
  - Lines: constant velocity movement
  - Arcs: circular motion with angular velocity
  - Bounces: velocity reversals at boundaries
- **Input**: 10 frames → Target: next 10 frames
- **Coordinate space**: 32x32 pixel grid with boundary constraints

### Main Configuration and Parameters
- **Model**: Enhanced coordinate regression with temporal strategies
- **Training**: 25 epochs, batch_size=16, lr=1e-3, weight_decay=1e-5
- **Temporal Strategies**: last_frame, velocity_propagation, learned_dynamics
- **Geometric Parameters**:
  - Line velocities: [-1.5, 1.5] pixels/frame
  - Arc radii: [3, 8] pixels, angular velocity: [-0.3, 0.3] rad/frame
  - Bounce velocities: [-1.8, 1.8] pixels/frame
- **Boundary Constraints**: [2, 30] pixel boundaries with bouncing

## Code Files and Functions

### test_geometric_patterns.py
This file implements improved coordinate regression and tests on geometric patterns.

**ImprovedCoordinateRegressionModel(nn.Module)**
- **Purpose**: Enhanced model with temporal prediction strategies
- **Architecture**:
  - CNN encoder for spatial feature extraction
  - Temporal processing with LSTM layers
  - Strategy-specific heads for different prediction approaches
- **Temporal Strategies**:
  - **last_frame**: Use only final frame for prediction
  - **velocity_propagation**: Estimate velocity and propagate forward
  - **learned_dynamics**: Learn temporal dynamics through LSTM
- **Assumption**: Better temporal modeling reduces velocity undershooting

**generate_line_pattern()**
- **Purpose**: Create linear motion trajectories
- **Algorithm**: Constant velocity integration with boundary clamping
- **Physics**: x(t) = x₀ + v*t, simple kinematic motion
- **Boundary Handling**: Clamp coordinates to valid range [2, 30]
- **Assumption**: Linear motion provides simplest test case for temporal prediction

**generate_arc_pattern()**
- **Purpose**: Create circular/arc motion trajectories  
- **Algorithm**: Parametric circle with time-varying angle
- **Physics**: x(t) = cx + r*cos(θ₀ + ω*t), y(t) = cy + r*sin(θ₀ + ω*t)
- **Parameters**: center (cx,cy), radius r, initial angle θ₀, angular velocity ω
- **Boundary Handling**: Clamp computed positions to valid range
- **Assumption**: Curved motion tests model's ability to learn non-linear dynamics

**generate_bounce_pattern()**
- **Purpose**: Create bouncing motion with velocity reversals
- **Algorithm**: Linear motion with elastic boundary collisions
- **Physics**: Velocity reverses when hitting boundaries (v → -v)
- **Boundary Logic**: Check position limits and reverse appropriate velocity component
- **Complexity**: Combines linear motion with discrete state changes
- **Assumption**: Discontinuous dynamics test model's handling of sudden changes

**coordinates_to_frames()**
- **Purpose**: Convert geometric coordinates to binary frame representation
- **Method**: Place single white pixel at coordinate location
- **Discretization**: Round continuous coordinates to integer pixel positions
- **Assumption**: Binary frame representation sufficient for geometric pattern encoding

**calculate_displacement_vectors()**
- **Purpose**: Compute frame-to-frame displacement vectors
- **Algorithm**: Δx(t) = x(t+1) - x(t), provides velocity information
- **Use Case**: Input for velocity-based prediction methods
- **Assumption**: Displacement vectors capture essential motion information

**create_enhanced_dataset()**
- **Purpose**: Prepare geometric data for enhanced temporal prediction
- **Data Structure**: Input frames, displacement vectors, future frame indices
- **Multi-step Setup**: Supports prediction at different temporal horizons
- **Assumption**: Multi-step prediction setup reveals temporal biases

**generate_geometric_datasets()**
- **Purpose**: Generate large-scale synthetic datasets for each pattern type
- **Sampling Strategy**: Random parameter sampling for pattern diversity
- **Data Augmentation**: Multiple samples per pattern type with parameter variation
- **Quality Control**: Boundary constraint enforcement and validity checking
- **Assumption**: Diverse synthetic data enables robust pattern learning

**train_enhanced_coordinate_model()**
- **Purpose**: Train improved coordinate regression with enhanced loss
- **Loss Function**: MSE for coordinate prediction accuracy
- **Optimization**: Adam with learning rate scheduling and gradient clipping
- **Enhanced Features**: Better handling of temporal dependencies
- **Assumption**: Enhanced architecture reduces systematic biases

**analyze_enhanced_velocity_patterns()**
- **Purpose**: Comprehensive analysis of velocity undershooting on geometric data
- **Velocity Analysis**:
  - **Magnitude Ratios**: predicted_magnitude / true_magnitude
  - **Direction Accuracy**: Cosine similarity between predicted and true directions
  - **Temporal Consistency**: Frame-by-frame velocity analysis
- **Bias Detection**: Systematic undershooting measurement
- **Pattern-Specific Analysis**: Separate analysis for each geometric pattern type
- **Key Metrics**: Undershooting ratios, direction accuracy, coordinate errors

**run_geometric_pattern_validation()**
- **Purpose**: Main experimental function for geometric validation
- **Experimental Design**: Train and test on synthetic geometric patterns
- **Validation Purpose**: Isolate model biases from data complexity
- **Expected Results**: Confirm velocity undershooting on predictable patterns
- **Scientific Value**: Demonstrates bias is model-inherent, not data-dependent

### plot_trajectory_examples.py
This file creates visualizations of geometric pattern examples.

**generate_line_pattern(), generate_arc_pattern(), generate_bounce_pattern()**
- **Purpose**: Same geometric pattern generation as main script
- **Visualization Focus**: Create representative examples for documentation
- **Parameter Selection**: Choose visually clear parameter values

**plot_phase4_trajectories()**
- **Purpose**: Create comprehensive visualization of all three pattern types
- **Visualization Design**:
  - Three-panel layout showing different pattern types
  - Color coding: input (black), future (red), start/end markers
  - Frame annotations: Show temporal progression
- **Educational Value**: Illustrates the types of patterns used in validation
- **Documentation**: Provides visual reference for geometric test cases

### visualize_geometric_predictions.py
This file creates detailed visualizations of model predictions on geometric patterns.

**ImprovedCoordinateRegressionModel(nn.Module)**
- **Purpose**: Same enhanced model architecture as main script
- **Consistency**: Ensures identical model for prediction visualization

**generate_line_pattern(), generate_arc_pattern(), generate_bounce_pattern()**
- **Purpose**: Identical pattern generation for consistent testing
- **Assumption**: Same patterns used for training and visualization

**coordinates_to_frames()**
- **Purpose**: Convert coordinates for model input
- **Method**: Standard binary frame representation

**predict_parallel_trajectory()**
- **Purpose**: Generate model predictions for visualization
- **Prediction Strategy**: Parallel prediction of multiple future frames
- **Method**: Model predicts all future coordinates simultaneously
- **Output**: Predicted coordinate sequence for comparison with ground truth

**reconstruct_trajectory_from_displacements()**
- **Purpose**: Convert displacement predictions back to coordinate trajectories
- **Algorithm**: Integrate displacements starting from known initial position
- **Mathematical**: x(t+1) = x(t) + Δx(t), cumulative integration
- **Use Case**: For models that predict displacements rather than absolute coordinates

**generate_parallel_predictions_visualization()**
- **Purpose**: Create comprehensive visualization comparing predictions vs ground truth
- **Visualization Components**:
  - Side-by-side comparison of true vs predicted trajectories
  - Error analysis and statistical summaries
  - Pattern-specific results for lines, arcs, and bounces
- **Analysis**: Quantifies prediction accuracy and identifies systematic biases
- **Documentation**: Provides visual evidence of model performance on geometric patterns 