## Contributions Overview

For the original OpenSTL documentation, see the OpenSTL README: [./README.md](./README.md).

### gauss_geom_simvp_exp

- Experiments for sparse trajectory modeling with SimVP-gSTA on synthetic data.
- Datasets: A white dot moving with `gauss` (position-dependent Gaussian sampling) and `geom_simple` (regular line/arc/bounce patterns) trajectory.
- Representations of white dot: single pixel, heatmap, and coordinate (displacement vectors directly feed into model).
- Loss families explored: pixel (WeightedBCE, FocalBCE, DiceBCE), heat (MSE/WeightedMSE, KL, EMD), coord (Huber, L1+Cosine).
- Metrics recorded per representation:
  - pixel: precision, recall, F1
  - heat: MSE/KL/EMD
  - coord: displacement errors (at full/3/6 frames), cosine similarity, magnitude ratio
- The PDF summarizes 18 experiments crossing the two datasets, three representations, and multiple losses, with training curves and qualitative frame grids:
  [Sparse_Trajectory_Modeling_Experiments_with_SimVPgSTA.pdf](./Sparse_Trajectory_Modeling_Experiments_with_SimVPgSTA.pdf).

### moving_mnist_zero_shot

- Zero-shot stress tests on Moving MNIST variants (1.5x/2x/3x speed-ups and 3 digits) to visualize model behavior.

- Preview: Direct movie grids rendered from raw outputs:

![Sample Movies from SimVP-gSTA](./moving_MNIST_openSTL\(1\).gif)

![Sample Movies from ConvLSTM](./moving_MNIST_openSTL\(2\).gif)


