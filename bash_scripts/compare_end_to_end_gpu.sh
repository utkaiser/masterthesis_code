#!/bin/bash

#SBATCH --job-name=compare_gpu_speed
#SBATCH --output=compare_gpu_speed.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1000M

pwd; hostname; date

python3 end_to_end_compare_gpu.py