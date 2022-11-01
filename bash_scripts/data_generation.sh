#!/bin/bash

#SBATCH --job-name=data_gen_end_to_end
#SBATCH --output=data_gen_end_to_end.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000M

pwd; hostname; date

python3 ../generate_data/datagen_end_to_end.py