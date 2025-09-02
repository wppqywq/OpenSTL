# Paths
DATA_ROOT = "data/moving_mnist"
OUTPUT_ROOT = "experiments/moving_mnist_zero_shot/results"
VARIANTS_ROOT = "experiments/moving_mnist_zero_shot/data_variants"

# Audit parameters
AUDIT_SAMPLE_SEQUENCES = 64
AUDIT_SAMPLE_FRAMES_PER_SEQUENCE = 5
AUDIT_SAVE_PLOTS = True

# Patch detaching parameters
PATCH_THRESHOLD = 200
PATCH_USE_OTSU = False
PATCH_MORPH_OPEN = True
PATCH_MIN_AREA = 10

# Motion controller parameters
MOTION_MODE = "baseline"  # baseline | fast | center_speed | center_direction
MOTION_BASE_SPEED = 1.0
MOTION_STEP_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]
CENTER_SPEED_ALPHAS = [0.0, 0.3, 0.6, 0.9]
CANVAS_SIZE = (64, 64)
NUM_DIGITS = 2
USE_MIRROR_REFLECTION = True

# Evaluation parameters
EVAL_ROLLOUT_STEPS = [20, 100, 200]
EVAL_COMPUTE_LPIPS = False

# Baselines
BASELINES = [
    {
        "name": "SimVP",
        "checkpoint": "models/simvp_method_best.pth",
        "expects_channels": 1,
    },
    {
        "name": "PredRNNpp",
        "checkpoint": None,
        "expects_channels": 1,
    },
]
