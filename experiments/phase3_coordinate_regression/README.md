# Phase 3: Direct Coordinate Regression

## Summary
This phase implements direct coordinate prediction to demonstrate expected failure with clustering behavior. The experiment tests whether neural networks can learn multi-step trajectory prediction through direct (x,y) coordinate regression using different prediction strategies.

## How to Run
```bash
cd experiments/phase3_coordinate_regression
python train_coordinate_regression.py
python visualize_predictions.py
```

### Input Data
- Same eye movement coordinate data from previous phases
- Converted to frame sequences with direct coordinate targets
- Input: 5 frames → Output: 5 future coordinates
- Three prediction strategies: Parallel, Sequential, Seq2Seq

### Main Configuration and Parameters
- **Model**: SimVP encoder + MLP head for coordinate prediction
- **Training**: 30 epochs, batch_size=8, lr=1e-3, weight_decay=1e-5
- **Architecture**: SimVP encoder → 512D features → MLP → 2D coordinates
- **Strategies**: Parallel (all at once), Sequential (one-by-one), Seq2Seq (recurrent)
- **Expected Outcome**: All strategies should fail with severe clustering

## Code Files and Functions

### train_coordinate_regression.py
This file implements direct coordinate regression to demonstrate clustering failure.

**SimVPCoordinateRegressor(nn.Module)**
- **Purpose**: Neural network for direct coordinate prediction from video frames
- **Architecture**: 
  - SimVP encoder for spatiotemporal feature extraction
  - MLP decoder for coordinate regression
- **Strategies**:
  - **Parallel**: Predict all future coordinates simultaneously
  - **Sequential**: Predict coordinates one-by-one in temporal order
  - **Seq2Seq**: Recurrent prediction with hidden state propagation
- **Assumption**: Different prediction strategies might have different failure modes

**load_real_data()**
- **Purpose**: Load coordinate data for regression experiments
- **Data Processing**: Extracts coordinate sequences for direct supervision
- **Assumption**: Direct coordinate supervision provides strongest training signal

**create_sparse_frames()**
- **Purpose**: Convert coordinates to binary frames for model input
- **Method**: Standard sparse representation used in previous phases
- **Assumption**: Frame representation necessary for video model input

**create_coordinate_dataset()**
- **Purpose**: Prepare coordinate sequences for direct regression training
- **Format**: Input frames paired with target coordinate sequences
- **Temporal Structure**: Maintains sequential ordering for multi-step prediction
- **Assumption**: Sequential structure important for trajectory learning

**train_coordinate_model()**
- **Purpose**: Train coordinate regression model with MSE loss
- **Loss Function**: L2 distance between predicted and true coordinates
- **Algorithm**: Standard PyTorch training with gradient clipping
- **Optimization**: Adam optimizer with learning rate scheduling
- **Validation**: Early stopping based on coordinate prediction accuracy
- **Assumption**: MSE loss appropriate for coordinate regression

**evaluate_coordinate_model()**
- **Purpose**: Comprehensive evaluation revealing clustering behavior
- **Metrics**:
  - **Mean coordinate error**: Average Euclidean distance
  - **Movement analysis**: Ratio of predicted vs true movement
  - **Clustering detection**: Standard deviation analysis
  - **Spatial distribution**: Center bias analysis
- **Key Insights**: Designed to detect clustering and center bias failures
- **Method**: Compares statistical properties of predictions vs ground truth

**visualize_prediction_failure()**
- **Purpose**: Create visualizations demonstrating clustering behavior
- **Plots**:
  - Full trajectory comparison (predicted vs true)
  - Focus on predicted region showing clustering
  - Statistical summaries of movement ratios
- **Analysis**: Quantifies clustering through movement ratio and standard deviation
- **Assumption**: Visual analysis reveals failure modes not captured by mean error alone

**run_coordinate_regression_experiment()**
- **Purpose**: Main experimental function demonstrating failure across all strategies
- **Experimental Design**: Tests all three prediction strategies under identical conditions
- **Expected Results**: All strategies should show clustering and center bias
- **Scientific Purpose**: Demonstrates insufficiency of direct coordinate prediction

### visualize_predictions.py
This file creates detailed visualizations of clustering failures.

**load_trained_model()**
- **Purpose**: Load pre-trained coordinate regression models
- **Model Loading**: Supports all three prediction strategies
- **Assumption**: Trained models exhibit characteristic clustering behavior

**coordinates_to_frames()**
- **Purpose**: Convert coordinate sequences back to frame representation
- **Method**: Creates binary frames for visualization purposes
- **Use Case**: Enables frame-by-frame analysis of predictions

**predict_sample_trajectories()**
- **Purpose**: Generate predictions on test samples for visualization
- **Method**: Applies trained model to multiple test sequences
- **Output**: Predicted coordinate sequences for clustering analysis
- **Sampling**: Selects diverse test cases to demonstrate general failure pattern

**visualize_clustering_failure()**
- **Purpose**: Create comprehensive visualization of clustering behavior
- **Visualization Components**:
  - **Trajectory plots**: Show full predicted vs true trajectories
  - **Clustering analysis**: Focus on predicted regions with density plots
  - **Statistical overlays**: Movement ratios and standard deviations
  - **Failure indicators**: Highlight characteristic clustering patterns
- **Analysis Method**: 
  - Computes movement ratios (predicted/true movement magnitude)
  - Measures spatial concentration through standard deviation
  - Identifies center bias through mean position analysis
- **Key Metrics**: Movement ratios <0.3 indicate severe clustering

**analyze_clustering_behavior()**
- **Purpose**: Quantitative analysis of clustering characteristics
- **Statistical Analysis**:
  - **Movement ratio**: Measures relative movement magnitude
  - **Spatial spread**: Standard deviation of coordinate distributions
  - **Center bias**: Distance from predicted center to true center
  - **Temporal consistency**: Frame-to-frame movement analysis
- **Clustering Indicators**:
  - Low movement ratios (expected <0.3)
  - Reduced spatial standard deviation
  - Bias toward image center (16, 16)
- **Scientific Value**: Provides quantitative evidence of prediction clustering

**main()**
- **Purpose**: Generate all visualization outputs demonstrating failure
- **Output Files**: Creates clustering failure visualizations for all strategies
- **Documentation**: Provides visual evidence supporting hypothesis that direct coordinate regression fails 