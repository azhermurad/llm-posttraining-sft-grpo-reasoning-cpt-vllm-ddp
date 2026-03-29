from unsloth import FastLanguageModel
import torch
from typing import Dict


def load_model(config:Dict) -> FastLanguageModel:
    """
    Load a language model with specified configurations.
    Args:
        config (dict): config of the pre-trained model to load.
    Returns:
        model (FastLanguageModel): The loaded language model instance.
        tokenizer (Tokenizer): The tokenizer associated with the model.
        
    """
    # try and catch errors during model loading
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = config['model_name'], # "unsloth/mistral-7b" for 16bit loading
            max_seq_length = config['max_seq_length'], # Choose any! We auto support RoPE Scaling internally!
            dtype = None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit = config['load_in_4bit'], # Set to False to disable 4bit quantization and use full precision 16bit
            load_in_8bit = config['load_in_8bit']
        )
        
    except Exception as e:
        print(f"Error loading model {config['model_name']}: {e}")
        raise RuntimeError(f"Failed to load model {config['model_name']}")
    
    return model, tokenizer


    
# Add LoRA adapters

def add_lora_adapters(model: FastLanguageModel,lora_config: Dict) -> FastLanguageModel:
    """
    Add LoRA adapters to a language model with custom configuration.
    Args:
        model:FastLanguageModel model to load.
        lora_config (dict): LoRA configuration dictionary. Example:
            {
                "r": 128,
                "lora_alpha": 32,
                "lora_dropout": 0,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"
                ]
            }
    Returns:
        FastLanguageModel: The language model with LoRA adapters added.
    """
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.get("r", 128), # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=lora_config.get("target_modules", []),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0),
            bias = "none",    # Supports any, but = "none" is optimized
            #  uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context 
            random_state = 3407,
            use_rslora = True,   # We support rank stabilized LoRA
            loftq_config = None,
        )
    except Exception as e:
        print(f"Error adding LoRA adapters: {e}")
        raise RuntimeError("Failed to add LoRA adapters")
    
    return model
