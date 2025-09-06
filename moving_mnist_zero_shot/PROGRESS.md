## Moving MNIST Zero-shot Benchmark — Progress Log

- What: Set up a clean workspace and zero-shot pipeline for data audit, patch detaching, motion-controlled variants, and evaluation.
- Why: Ensure reproducible, controlled stress tests (speed/field) and verifiable preprocessing before model benchmarking.

### Done
- Pixel-value audit
  - Data: `data/moving_mnist/mnist_cifar_test_seq.npy` (shape: 20x10000x64x64x3)
  - Finding: `any_gray_detected = true` (anti-aliased edges present)
  - Outputs: `moving_mnist_zero_shot/results/audit/unique_values.json` + histogram PNGs
- Patch detaching
  - Thresholding (T=200, Otsu optional) + morphological opening
  - Keep two largest connected components; extract tight patches with black background
  - Outputs: masks/patches/overlays + `manifest.json` under `results/patch_detach/`
- Motion controller (re-render)
  - Modes: baseline, fast (s=1.5), center_speed (alpha in {0.0,0.3,0.6,0.9}), center_direction
  - Saved: sequences per mode under `moving_mnist_zero_shot/data_variants/<variant>/seq*.npy`
  - Logged: per-frame digit centers in `seq*_centers.json`
- Variant analysis (trajectory-level)
  - Metrics: avg speed error (baseline/center_speed), avg direction error (center_direction)
  - Result: near-zero errors, consistent with controller definitions
  - Output: `moving_mnist_zero_shot/results/eval/variants_summary.json`
- Frame-level metrics (baseline vs original)
  - Computed MSE/PSNR for first 5 sequences
  - Result: avg PSNR ~4-7 dB (expected difference due to re-rendering from patches)
  - Output: `moving_mnist_zero_shot/results/eval/frame_metrics.json`

### Next
- Add frame-level metrics (MSE/PSNR) vs. original frames for baseline variant
- Plot error-vs-time summaries; then integrate model baselines (SimVP, PredRNN++) zero-shot
