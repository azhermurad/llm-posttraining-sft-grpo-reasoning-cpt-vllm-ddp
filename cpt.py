import os
from dotenv import load_dotenv
from huggingface_hub import notebook_login,login


# Load environment variables from .env file
load_dotenv()

WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_LOG_MODEL = os.getenv("WANDB_LOG_MODEL")
WANDB_WATCH = os.getenv("WANDB_WATCH")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
Huggingface_API_TOKEN = os.getenv("Huggingface_API_TOKEN")



import argparse

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

    
    def hf_login():
        """Login to Huggingface Hub using the provided API token."""
        login(token=Huggingface_API_TOKEN)
        print("Logged in to Huggingface Hub.")
        
    
    
if __name__ == '__main__':
    main()
    