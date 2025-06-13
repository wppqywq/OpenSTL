#!/bin/bash
#SBATCH --job-name=simvp_training
#SBATCH --account=def-skrishna
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# 加载必要模块
module load apptainer

# 设置工作目录
cd $SLURM_SUBMIT_DIR/OpenSTL

# 打印环境信息
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# 运行训练
echo "=== Starting Training ==="
apptainer exec --nv --bind $PWD:/workspace ../pytorch_openstl_fixed.sif python tools/train.py \
    -d mmnist \
    --lr 1e-3 \
    -c configs/mmnist/simvp/SimVP_gSTA.py \
    --ex_name mmnist_simvp_gsta_$(date +%Y%m%d_%H%M) \
    --epochs 200 \
    --batch_size 16 \
    --log_step 50 \
    --save_checkpoint

echo "=== Training Completed ==="
