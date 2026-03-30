from datasets import load_dataset



 # English Prompt
    # _wikipedia_prompt = """Wikipedia Article
    # ### Title: {}

    # ### Article:
    # {}"""
    
ur_wiki_prompt = """ویکیپیڈیا آرٹیکل
### عنوان: {}

### مضمون:
{}"""


def load_dataset_by_name(dataset_name: str,language: str,tokenizer, split: str = 'train'):
    """
    Load a dataset by its name using the Hugging Face datasets library.
    Args:
    Returns:
        dataset_name (str): The name of the dataset to load.
        language: wikipedia dataset language 
        tokenizer (Tokenizer): The tokenizer to use for processing the dataset.
        split (str): The split of the dataset to load (default is 'train').
        dataset: The loaded dataset.
    """
    
    try:
        dataset = load_dataset(dataset_name, language, split = split)
        
        # We select 1% of the data to make training faster!
        # dataset = dataset.train_test_split(train_size = 0.01)["train"]
        
        EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
        
        def formatting_prompts_func(examples):
            titles = examples["title"]
            texts  = examples["text"]
            outputs = []
            for title, text in zip(titles, texts):
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                text = ur_wiki_prompt.format(title, text) + EOS_TOKEN
                outputs.append(text)
            return { "text" : outputs, }
            
        
        dataset = dataset.map(formatting_prompts_func, batched = True,)
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        raise RuntimeError(f"Failed to load dataset {dataset_name}")
    
    return dataset

    
  

