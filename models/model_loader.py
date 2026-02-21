from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.



def load_model(model_name: str) -> FastLanguageModel:
    """
    Load a language model with specified configurations.
    Args:
        model_name (str): The name or path of the pre-trained model to load.
    Returns:
        model (FastLanguageModel): The loaded language model instance.
        tokenizer (Tokenizer): The tokenizer associated with the model.
        
    """
    # try and catch errors during model loading
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, # "unsloth/mistral-7b" for 16bit loading
            max_seq_length = max_seq_length, # Choose any! We auto support RoPE Scaling internally!
            dtype = dtype,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit = load_in_4bit # Set to False to disable 4bit quantization and use full precision 16bit
        )
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise RuntimeError(f"Failed to load model {model_name}")
    
    return model, tokenizer


# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=config["model"]["base_model"],
#     max_seq_length=config["model"]["max_seq_length"],
#     dtype=config["model"]["dtype"],
#     load_in_4bit=config["model"]["load_in_4bit"],
#     load_in_8bit=config["model"]["load_in_8bit"],
# )
    
    
# Add LoRA adapters

def add_lora_adapters(model: FastLanguageModel) -> FastLanguageModel:
    """
    Add LoRA adapters to the model for parameter-efficient fine-tuning.
    Args:
        model (FastLanguageModel): The language model to which LoRA adapters will be added.
    Returns:
        model (FastLanguageModel): The language model with LoRA adapters added.
        
    """
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                            "embed_tokens", "lm_head",], # Add for continual pretraining
            lora_alpha = 32,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            #  uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = True,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    except Exception as e:
        print(f"Error adding LoRA adapters: {e}")
        raise RuntimeError("Failed to add LoRA adapters")
    
    return model

# model = FastLanguageModel.get_peft_model(
#     model,
#     r=config["lora"]["r"],
#     lora_alpha=config["lora"]["lora_alpha"],
#     lora_dropout=config["lora"]["lora_dropout"],
#     bias=config["lora"]["bias"],
#     target_modules=config["lora"]["target_modules"],
#     use_gradient_checkpointing=config["lora"]["use_gradient_checkpointing"],
#     random_state=config["lora"]["random_state"],
#     use_rslora=config["lora"]["use_rslora"],
#     loftq_config=config["lora"]["loftq_config"],
# )