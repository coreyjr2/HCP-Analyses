#!/bin/bash
#SBATCH --job-name=permutation_importance_abcd_sip
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19740%100
#SBATCH --mem=3G
#SBATCH --output=/home/kbaacke/Data/kbaacke/dr-fs/slurm/dr-fs_%A_%a.out 
#SBATCH --error=/home/kbaacke/Data/kbaacke/dr-fs/slurm/dr-fs_%A_%a.err

. /sharedapps/LS/conda/miniconda/etc/profile.d/conda.sh
conda activate
conda activate ~/Data/kbaacke/conda_envs/HCPAnalyses

python ~/Data/kbaacke/dr-fs/BIGCLUSTERMODELRUNNINGSCRIPT.py