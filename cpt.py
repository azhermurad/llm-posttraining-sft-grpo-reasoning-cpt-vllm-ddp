import os
import argparse
import torch
# from unsloth import FastLanguageModel
# from transformers import TrainingArguments
# from unsloth import UnslothTrainer, UnslothTrainingArguments,is_bfloat16_supported
# from datasets import load_dataset
from dotenv import load_dotenv
from utils.auth import login_wandb,login_huggingface
from utils.config_loader import load_config

# Load environment variables from .env file
load_dotenv()






def main():
    parser = argparse.ArgumentParser(description='Model Training')
    # Training parameters
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--model', type=str, default='resnet', help='Model name')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    
    args = parser.parse_args()
    
    # Your training code
    print(f"Training with:")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Model: {args.model}")
    print(f"  GPU: {args.gpu}")
    
    
    # login wandb
    login_wandb()
    login_huggingface()
    load_config_path = 'cpt_config'
    
    confg = load_config(load_config_path)
    print(confg)

    
        
    
    
if __name__ == '__main__':
    main()
    
    