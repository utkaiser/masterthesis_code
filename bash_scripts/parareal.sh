#!/bin/bash

#SBATCH --job-name=parareal_1
#SBATCH --output=parareal_1.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=1M

pwd; hostname; date

python3 ../parareal/models/train_end_to_end.py