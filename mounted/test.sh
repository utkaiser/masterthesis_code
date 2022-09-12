#!/bin/bash

#SBATCH --job-name=TestJob_1
#SBATCH --output=TestJob_1.out
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M
pwd; hostname; date

python3 generatecroppedVmodel.py