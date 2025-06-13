#!/bin/bash
#SBATCH --job-name=train_with_ownload
#SBATCH --account=def-skrishna
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load StdEnv/2023 python/3.11 scipy-stack
module load cuda/12
module load opencv

source $SLURM_SUBMIT_DIR/openstl_env/bin/activate

cd $SLURM_SUBMIT_DIR/

echo "Current directory: $(pwd)"
echo "Environment check:"
which python
python -c "import sys; print('Python path:', sys.path[0])"



echo "=== Checking data path ==="
ls -la ../data/moving_mnist/ | head -3
ls -la ./data/moving_mnist/ | head -3

echo "=== Testing openstl import ==="
python -c "import openstl; print('✅ OpenSTL imported successfully')"

echo "=== Available configs ==="
ls OpenSTL/configs/mmnist/simvp/ | head -5

echo "=== Starting Training ==="
python OpenSTL/tools/train.py \
  -d mmnist \
  -c OpenSTL/configs/mmnist/simvp/SimVP_gSTA.py \
  --ex_name path_fixed

echo "=== Training Completed ==="
