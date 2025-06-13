#!/bin/bash
#SBATCH --job-name=test_modules
#SBATCH --account=def-skrishna
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# 加载Compute Canada预装模块
module load StdEnv/2020 python/3.9 scipy-stack cuda/11.8
module load opencv/4.5.5

# 创建轻量级虚拟环境
python -m venv ~/venvs/openstl_light --system-site-packages
source ~/venvs/openstl_light/bin/activate

# 只安装缺少的包
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install lightning==2.2.1 timm einops fvcore lpips
pip install hickle omegaconf

# 测试
python -c "
import cv2
import torch
import lightning
import timm
import einops
print('All modules loaded successfully!')
print(f'OpenCV: {cv2.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
