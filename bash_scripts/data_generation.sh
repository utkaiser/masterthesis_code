#!/bin/bash

#SBATCH --job-name=data_generation_1
#SBATCH --output=data_generation_1.out
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M

pwd; hostname; date

python3 ../mounted/generatecroppedVmodel.py
python3 ../mounted/parareal_datagen.py