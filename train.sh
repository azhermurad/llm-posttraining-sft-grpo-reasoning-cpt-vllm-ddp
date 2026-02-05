#!/bin/bash
#SBATCH --job-name=cpt_llm_mistral_1b
#SBATCH --output=outputs/pt_llm_mistral_training_%j.out
#SBATCH --error=outputs/pt_llm_mistral_training_%j.err
#SBATCH --nodelist=compute1          # Force execution on compute1
#SBATCH --nodes=1                     # Single node
#SBATCH --ntasks=1                    # Single task
#SBATCH --cpus-per-task=4             # CPU cores
#SBATCH --gres=gpu:1                  # Single GPU
# MAX allowed by cluster
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G                     # Memory

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of Nodes: $SLURM_NNODES"
echo "Node List: $SLURM_NODELIST"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate venv
python -V

# Set Hugging Face cache variable
export HF_HUB_DISABLE_XET=1

# Start training


# Start training
echo "Starting single GPU training on compute1..."
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Simple direct execution - no distributed setup needed
python cpt.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "End Time: $(date)"
echo "=========================================="


