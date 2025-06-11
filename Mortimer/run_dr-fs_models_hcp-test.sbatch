#!/bin/bash
#SBATCH --job-name=hcp_predgen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --array=0
#SBATCH --mem=64G
#SBATCH --output=/raid-18/LS/medinak/kbaacke/dr-fs/slurm/dr-fs-hcp_%A_%a.out 
#SBATCH --error=/raid-18/LS/medinak/kbaacke/dr-fs/slurm/dr-fs-hcp_%A_%a.err

. /sharedapps/LS/conda/miniconda/etc/profile.d/conda.sh
conda activate
conda activate ~/Data/kbaacke/conda_envs/HCPAnalyses_v3


/raid-18/LS/medinak/kbaacke/dr-fs/scripts/compute_monitor.sh -i 5 -d 259200 -n dr-fs-hcp_0 -o /raid-18/LS/medinak/kbaacke/dr-fs/hcp/ &
python ~/Data/kbaacke/dr-fs/scripts/gridsearch-predict_hcp.py
exit