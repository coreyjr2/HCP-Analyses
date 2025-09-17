#!/bin/bash
#SBATCH --job-name=ucla_predgen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --array=235-819%100
#SBATCH --mem=32G
#SBATCH --output=/raid-18/LS/medinak/kbaacke/dr-fs/slurm/dr-fs-ucla_Random%A_%a.out 
#SBATCH --error=/raid-18/LS/medinak/kbaacke/dr-fs/slurm/dr-fs-ucla__random%A_%a.err

. /sharedapps/LS/conda/miniconda/etc/profile.d/conda.sh
conda activate
conda activate ~/Data/kbaacke/conda_envs/HCPAnalyses_v3

python -u ~/Data/kbaacke/dr-fs/scripts/gridsearch-predict_ucla.py 
exit