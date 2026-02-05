#!/bin/bash
#SBATCH --job-name=llm_training
#SBATCH --gres=gpu:1
#SBATCH --nodelist=compute1
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.log
#SBATCH --partition=gpu

srun --pty tmux
