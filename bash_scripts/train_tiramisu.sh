#!/bin/bash

#SBATCH --job-name=train_tiramisu_1
#SBATCH --output=train_tiramisu_1.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2000M

pwd; hostname; date

python3 ../models/train.py tiramisu