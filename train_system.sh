#!/bin/bash
#SBATCH --job-name=simvp_fixed
#SBATCH --account=def-skrishna
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# 加载正确的模块版本
module load StdEnv/2023 python/3.11 scipy-stack

# 检查可用CUDA版本并加载
echo "Available CUDA versions:"
module avail cuda 2>&1 | grep cuda
# 通常cedar有这些版本
if module avail cuda 2>&1 | grep -q "cuda/12"; then
    module load cuda/12
elif module avail cuda 2>&1 | grep -q "cuda/11.7"; then
    module load cuda/11.7
else
    echo "Loading default CUDA"
    module load cuda
fi

# 加载OpenCV
module load opencv

echo "Loaded modules:"
module list

cd $SLURM_SUBMIT_DIR

# 创建venv并安装OpenSTL
if [ ! -d "openstl_env" ]; then
    python -m venv openstl_env --system-site-packages
fi

source openstl_env/bin/activate

# 安装必要依赖
pip install --no-cache-dir lightning timm einops fvcore lpips tqdm packaging pyyaml omegaconf

# 关键：安装OpenSTL包
cd OpenSTL
pip install -e .

# 验证安装
python -c "
import openstl
import torch
import cv2
print('✅ All modules imported successfully')
print(f'OpenSTL version: {openstl.__version__ if hasattr(openstl, \"__version__\") else \"dev\"}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo "=== Starting Training ==="

python tools/train.py \
    -d mmnist \
    --lr 1e-3 \
    -c configs/mmnist/simvp/SimVP_gSTA.py \
    --ex_name system_fixed \
    --batch_size 4

echo "=== Training Completed ==="
