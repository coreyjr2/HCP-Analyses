#!/bin/bash
#SBATCH --job-name=hcp_predgen_rerun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0,1,7,8,9,10,11,12,16,18,19
#SBATCH --mem=123G
#SBATCH --output=/raid-18/LS/medinak/kbaacke/dr-fs/slurm/__dr-fs-hcp_reruns_%A_%a.out 
#SBATCH --error=/raid-18/LS/medinak/kbaacke/dr-fs/slurm/__dr-fs-hcp_reruns_%A_%a.err

. /sharedapps/LS/conda/miniconda/etc/profile.d/conda.sh
conda activate
conda activate ~/Data/kbaacke/conda_envs/HCPAnalyses_v3

python -u ~/Data/kbaacke/dr-fs/scripts/gridsearch-predict_hcp_reruns.py
exit