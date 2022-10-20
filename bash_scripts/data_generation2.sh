#!/bin/bash

#SBATCH --job-name=data_generation_2
#SBATCH --output=data_generation_2.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000M

pwd; hostname; date

python3 ../generate_data/datagen_Dtp.py 64 256