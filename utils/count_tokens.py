from datasets import load_dataset
from transformers import AutoTokenizer

# Load Wikipedia dataset (or your local copy)

dataset = load_dataset("wikimedia/wikipedia", "20231101.ur", split="train")  # small sample

# Load CPT tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")  # replace with CPT model name

# Function to count tokens
def count_tokens(batch):
    return {"num_tokens": len(tokenizer(batch["text"])["input_ids"])}

# Map the function over the dataset
dataset_with_tokens = dataset.map(count_tokens)

total_tokens = sum(dataset_with_tokens["num_tokens"])
print(f"Total tokens in dataset according to CPT tokenizer: {total_tokens}")