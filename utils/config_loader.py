"""
Configuration loading utilities
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import os

def load_config(filename: str) -> Dict[str, Any]:
    """
    Load YAML configuration file
    Args:
        filename: Name of YAML config file (without .yaml extension)
        e.g., 'cpt_config', 'grpo_config', 'sft_config'
        
    Returns:
        Configuration dictionary
    """
    ALLOWED_CONFIGS = {'cpt_config', 'grpo_config','sft_config'}
    if filename not in ALLOWED_CONFIGS:
        raise ValueError(f"Config file {filename} not allowed. Allowed files: {ALLOWED_CONFIGS}")
    
    BASE_DIR = Path(__file__).resolve().parents[1]
    config_path = BASE_DIR / "configs" / f"{filename}.yaml"
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config








