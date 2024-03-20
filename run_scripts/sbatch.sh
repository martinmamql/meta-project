#!/bin/bash

#SBATCH -c8
#SBATCH -p high 
#SBATCH --mem=16g
#SBATCH --gres=gpu:2080Ti:1

# Add commands to run below here, e.g., 
# bash run.sh
