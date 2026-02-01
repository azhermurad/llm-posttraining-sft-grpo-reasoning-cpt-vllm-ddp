"""
Authentication utilities for wandb, HuggingFace, etc.
"""
import os
from typing import Optional
import wandb
from huggingface_hub import login as hf_login
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()



"""
Authentication utilities for wandb, HuggingFace, etc.
"""
import os
from typing import Optional
import wandb
from huggingface_hub import login
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    return {
        'wandb_api_key': os.getenv('WANDB_API_KEY'),
        'hf_token': os.getenv('Huggingface_API_TOKEN'),
        'wandb_project': os.getenv('WANDB_PROJECT'),
    }


# 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
def login_wandb(
    relogin: bool = False
):
    """
    Login to Weights & Biases
    
    Args:
        relogin: Force relogin even if already logged in
    """
    env_vars = load_environment()
    api_key =  env_vars['wandb_api_key']
    
    if not api_key:
        raise ValueError("WANDB_API_KEY not found in environment or .env file")
    
    if relogin:
        wandb.login(relogin=True, key=api_key)
    else:
        wandb.login(key=api_key)
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"]="cpt-mistral-7b-model"
    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="true"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"
    
    print(f"✓ Logged in to wandb (project: {env_vars['wandb_project']})")
    return True

def login_huggingface():
    """
    Login to HuggingFace Hub
    
    Args:
        token: HF token (if None, reads from env)
        add_to_git_credential: Add token to git credential store
    """
    env_vars = load_environment()
    token = env_vars['hf_token']
    if not token:
        raise ValueError("HF_TOKEN not found in environment .env file")
    
    login(token=token)
    print("✓ Logged in to HuggingFace Hub")
    return True

def setup_all_logins():
    """Setup all required logins (wandb, HF, etc.)"""
    print("Setting up authentication...")
    
    try:
        login_wandb()
        login_huggingface()
        print("\n✓ All authentication successful!")
        return True
    except Exception as e:
        print(f"\n✗ Authentication failed: {e}")
        return False

def init_wandb_run():
    """
    Initialize wandb run for a training stage

    """
    
    env_vars = load_environment()
  
    project = env_vars['wandb_project']
    
    run = wandb.init()
    artifact = run.use_artifact('azheraly009-nust/<Wandb-project-name>/<run-id>', type='model')
    artifact_dir = artifact.download()
    # trainer.train(resume_from_checkpoint=artifact_dir)
        
    print(f"✓ wandb run initialized: {run.name}")
    return run, artifact_dir