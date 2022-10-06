#!/bin/bash

#SBATCH --job-name=train_unet_1
#SBATCH --output=train_unet_1.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=11000M

pwd; hostname; date

python3 ../models/train_multi_gpu.py unet 128
python3 ../models/train_multi_gpu.py unet 258