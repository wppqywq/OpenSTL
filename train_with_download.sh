#!/bin/bash

#SBATCH --job-name=train_download

#SBATCH --account=def-skrishna

#SBATCH --time=00:30:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

#SBATCH --mem=16G

module load StdEnv/2023 python/3.11 scipy-stack

module load cuda/12

module load opencv

source $SLURM_SUBMIT_DIR/openstl_env/bin/activate

cd $SLURM_SUBMIT_DIR

# 下载MNIST数据（如果不存在）

if [ ! -f "data/moving_mnist/train-images-idx3-ubyte.gz" ]; then

    echo "=== Downloading MNIST data ==="

    mkdir -p data/moving_mnist

    wget -q -O data/moving_mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

    wget -q -O data/moving_mnist/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

    wget -q -O data/moving_mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

    wget -q -O data/moving_mnist/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

    echo "MNIST data downloaded"

fi

cd OpenSTL

echo "=== Starting Training ==="

python tools/train.py \

    -d mmnist \

    --lr 1e-3 \

    -c configs/mmnist/simvp/SimVP_gSTA.py \

    --ex_name with_download \

    --batch_size 4

echo "=== Training Completed ==="

