#!/bin/bash

#SBATCH --job-name=train_transformer_1
#SBATCH --output=train_transformer_1.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=20000M

pwd; hostname; date

python3 ../models/train_multi_gpu.py transformer 128 2