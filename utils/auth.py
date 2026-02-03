
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
        "run_id": os.getenv('RUN_ID'),
    }


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



def init_wandb_run():
    """
    Initialize wandb run for a training stage

    """
    
    env_vars = load_environment()
    project_name = env_vars['wandb_project']
    run_id = env_vars['run_id']

    api = wandb.Api()

# List all projects under your account
    for p in api.projects("azheraly009-nust"):
        print(p.name)
        
    print("Entity:", api.viewer.entity)
    print("Projects:")
    for p in api.projects(api.viewer.entity):
        print("-", p.name)
    # run = wandb.init()
    # artifact = run.use_artifact(f'azheraly009-nust/{project_name}:latest', type='model')
    # artifact_dir = artifact.download()
    # trainer.train(resume_from_checkpoint=artifact_dir)
        
    # print(f"✓ wandb run initialized: {run.name}")
    # return run, artifact_dir