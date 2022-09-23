#!/bin/bash

#SBATCH --job-name=data_generation
#SBATCH --output=data_generation.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000M

pwd; hostname; date

python3 ../generate_data/parareal_datagen.py