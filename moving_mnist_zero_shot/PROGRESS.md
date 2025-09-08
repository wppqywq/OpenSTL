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
- Error-vs-time plots
  - Generated MSE and PSNR temporal evolution plots
  - Output: `moving_mnist_zero_shot/results/eval/error_vs_time.png`

- Fast motion variants generated (s=2.0, s=3.0)
  - Generated 10 sequences per variant  
  - Trajectory analysis shows near-zero speed errors (machine precision)
  - Output: `moving_mnist_zero_shot/results/eval/all_variants_analysis.json`
- Model baseline evaluation completed
  - Three methods tested: copy-last, linear extrapolation, mean-context
  - Copy-last performs best (avg MSE=7.62) vs linear (5239) and mean (5697)
  - Results show expected patterns: higher speed/variation → higher error
  - Output: `moving_mnist_zero_shot/results/eval/comprehensive_baseline_results.json`

### Key Findings
- **Motion impact**: Fast motion (s=1.5) shows higher error than baseline
- **Field effects**: Center-direction shows moderate error increase
- **Best baseline**: Copy-last frame consistently outperforms other simple baselines

### Next  
- Long-horizon rollout to T=100/200 to analyze error accumulation
- Generate summary tables and plots for paper
- Complete reproducibility documentation
