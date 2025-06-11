#!/bin/bash
#SBATCH --job-name=ucla_predgen_test1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --array=0
#SBATCH --mem=32G
#SBATCH --output=/raid-18/LS/medinak/kbaacke/dr-fs/slurm/dr-fs-ucla-test1_%A_%a.out 
#SBATCH --error=/raid-18/LS/medinak/kbaacke/dr-fs/slurm/dr-fs-ucla-test1_%A_%a.err

. /sharedapps/LS/conda/miniconda/etc/profile.d/conda.sh
conda activate
conda activate ~/Data/kbaacke/conda_envs/HCPAnalyses_v3

/raid-18/LS/medinak/kbaacke/dr-fs/scripts/compute_monitor.sh -i 5 -d 259200 -n dr-fs-ucla_linear -o /raid-18/LS/medinak/kbaacke/dr-fs/ucla/ &
python ~/Data/kbaacke/dr-fs/scripts/gridsearch-predict_ucla.py 
exit