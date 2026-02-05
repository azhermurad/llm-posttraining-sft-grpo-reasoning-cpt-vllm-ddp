import argparse
from data.dataset_loader import load_dataset_by_name
from models.model_loader import load_model
from dotenv import load_dotenv
from utils.auth import login_wandb,login_huggingface
from utils.config_loader import load_config
from utils.get_checkpoint import get_checkpoint

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
    from utils.get_checkpoint import get_checkpoint
    
    # trainer_stats = cpt_trainer(model, tokenizer, dataset).train(resume_from_checkpoint = True)
    
    # Use it
   
    checkpoint_path = get_checkpoint("llama3_cpt_tinystories")
    if checkpoint_path:
        print(f"Resuming from: {checkpoint_path}")
        trainer_stats = cpt_trainer(model, tokenizer, dataset).train(resume_from_checkpoint=checkpoint_path)
    else:
        print("Starting from scratch")
        trainer_stats = cpt_trainer(model, tokenizer, dataset).train()
    
    # save the model
    model.save_pretrained("llama3_cpt_tinystories_final")
    tokenizer.save_pretrained("llama3_cpt_tinystories_final")
    
    
    from transformers import TextIteratorStreamer
    from threading import Thread
    text_streamer = TextIteratorStreamer(tokenizer)
    import textwrap
    max_print_width = 100

    # Before running inference, call `FastLanguageModel.for_inference` first

# fast inference
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
    [
        "Once upon a time, in a galaxy, far far away,"
    ]*1, return_tensors = "pt").to("cuda")

    generation_kwargs = dict(
        inputs,
        streamer = text_streamer,
        max_new_tokens = 256,
        use_cache = True,
    )
    thread = Thread(target = model.generate, kwargs = generation_kwargs)
    thread.start()

    length = 0
    for j, new_text in enumerate(text_streamer):
        if j == 0:
            wrapped_text = textwrap.wrap(new_text, width = max_print_width)
            length = len(wrapped_text[-1])
            wrapped_text = "\n".join(wrapped_text)
            print(wrapped_text, end = "")
        else:
            length += len(new_text)
            if length >= max_print_width:
                length = 0
                print()
            print(new_text, end = "")
        pass
    pass
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    