from datasets import load_dataset




def load_dataset_by_name(dataset_name: str,tokenizer, split: str = 'train'):
    """
    Load a dataset by its name using the Hugging Face datasets library.
    Args:
        dataset_name (str): The name of the dataset to load.
        tokenizer (Tokenizer): The tokenizer to use for processing the dataset.
        split (str): The split of the dataset to load (default is 'train').
    Returns:
        dataset: The loaded dataset.
    """
    
    try:
        dataset = load_dataset("roneneldan/TinyStories", split = "validation[:2500]")
        EOS_TOKEN = tokenizer.eos_token
        def formatting_prompts_func(examples):
            return { "text" : [example + EOS_TOKEN for example in examples["text"]] }
        dataset = dataset.map(formatting_prompts_func, batched = True,)
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        raise RuntimeError(f"Failed to load dataset {dataset_name}")
    
    return dataset



