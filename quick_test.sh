#!/bin/bash
#SBATCH --job-name=simvp_test
#SBATCH --account=def-skrishna
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G

module load apptainer
cd $SLURM_SUBMIT_DIR

apptainer exec --nv --bind $SLURM_SUBMIT_DIR:/workspace ./pytorch_openstl_fixed.sif python /workspace/OpenSTL/tools/train.py \
    -d mmnist \
    --lr 1e-3 \
    -c /workspace/OpenSTL/configs/mmnist/simvp/SimVP_gSTA.py \
    --ex_name quick_test \
    --epochs 1 \
    --batch_size 4
