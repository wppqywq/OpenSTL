#!/bin/bash
#SBATCH --job-name=train_fixed
#SBATCH --account=def-skrishna
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load StdEnv/2023 python/3.11 scipy-stack
module load cuda/12
module load opencv

source $SLURM_SUBMIT_DIR/openstl_env/bin/activate

cd $SLURM_SUBMIT_DIR/OpenSTL

# 确保数据路径正确 - 创建软链接
if [ ! -L "./data" ]; then
    ln -sf ../data ./data
    echo "Created symlink to data directory"
fi

echo "=== Checking data path ==="
ls -la ./data/moving_mnist/ | head -3

# 检查config和支持的参数
echo "=== Available configs ==="
ls configs/mmnist/simvp/

echo "=== Starting Training ==="
# 一行写完所有参数，避免参数分离问题
python tools/train.py -d mmnist --lr 1e-3 -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name fixed_test --batch_size 4

echo "=== Training Completed ==="
