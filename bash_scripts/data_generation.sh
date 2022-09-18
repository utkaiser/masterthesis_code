#!/bin/bash

#SBATCH --job-name=data_generation_1
#SBATCH --output=data_generation_1.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15000M

pwd; hostname; date

python3 ../mounted/parareal_datagen.py