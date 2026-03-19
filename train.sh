#!/bin/bash
#SBATCH --job-name=max
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            
#SBATCH --mem=8G                     
#SBATCH --gres=gpu:1
#SBATCH --time=65:00:00
#SBATCH --partition=gpu

# --- 1. Initialize the Module System ---
# This line is critical to make the 'module' command work in SLURM
#source /etc/profile.d/modules.sh || source /usr/share/modules/init/bash


# --- 2. Initialize and Activate Conda ---
# Use the full path to your conda initialization script
#CONDA_PATH=$(which conda | sed 's|/bin/conda||')
source "/home/super/miniconda3/condabin/conda.sh"
conda activate maxtorch

# --- 3. Run Training ---
# Note: The command is 'accelerate launch', not 'accelerator launch'
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/run_all.py