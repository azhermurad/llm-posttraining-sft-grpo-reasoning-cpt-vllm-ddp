import os
import argparse
from data.dataset_loader import load_dataset_by_name
from models.model_loader import load_model
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
    
    
    # login wandb and huggingface
    login_wandb()
    login_huggingface()
    
    # load model
    model, tokenizer = load_model("unsloth/Llama-3.2-1B")
    
    
    # load dataset
    dataset_name = "roneneldan/TinyStories"
    dataset = load_dataset_by_name(dataset_name, tokenizer)
    
    print(f"Loaded dataset: {dataset['text'][0]}")
    # add lora adapters
    from models.model_loader import add_lora_adapters
    model = add_lora_adapters(model)
    
    #  taining model 
    
    from training.cpt_trainer import cpt_trainer
    trainer_stats = cpt_trainer(model, tokenizer, dataset).train(resume_from_checkpoint = True)
    
    
    
    # # # init_wandb_run
    # from utils.auth import init_wandb_run
    # run, artifact_dir = init_wandb_run()
    # trainer_stats = cpt_trainer(model, tokenizer, dataset).train(resume_from_checkpoint = artifact_dir)
    
    
    
    
 
    
        
    
    
if __name__ == '__main__':
    main()
    
    