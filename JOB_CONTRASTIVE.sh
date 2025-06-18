#!/bin/bash
# SCRIPT NAME: job.sh

# Partition type
#SBATCH --partition=high

# Number of nodes
#SBATCH --nodes=1

# Number of tasks
#SBATCH --ntasks=1

# Number of tasks per node
#SBATCH --tasks-per-node=1

# Memory per node. 5 GB (In total, 10 GB)
#SBATCH --mem=10g

# Number of GPUs per node
#SBATCH --gres=gpu:1

# Select Intel nodes (with Infiniband)
#SBATCH --constraint=intel

# Modules
module load CUDA

cd /home/tnuttall/CMMR2025/DeepGRU

singularity run --nv tn-deepgru_dtw.sif python contrastive_main.py --run-name 'saraga' --save-model
