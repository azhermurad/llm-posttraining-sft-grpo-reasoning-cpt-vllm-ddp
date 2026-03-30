import argparse
from data.dataset_loader import load_dataset_by_name
from dotenv import load_dotenv
from models.model_loader import load_model, add_lora_adapters
from utils.auth import login_wandb, login_huggingface
from utils.config_loader import load_config
from training.cpt_trainer import cpt_trainer
from training.sft_trainer import sft_trainer
from utils.get_checkpoint import get_checkpoint

# Load environment variables from .env file
load_dotenv()


def main():
    # parser = argparse.ArgumentParser(description='Model Training')
    # # Training parameters
    # parser.add_argument('--batch', type=int, default=32, help='Batch size')
    # parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    # parser.add_argument('--model', type=str, default='resnet', help='Model name')
    # parser.add_argument('--gpu', action='store_true', help='Use GPU')
    # args = parser.parse_args()
    # # Your training code
    # print(f"Training with:")
    # print(f"  Batch size: {args.batch}")
    # print(f"  Learning rate: {args.lr}")
    # print(f"  Epochs: {args.epochs}")
    # print(f"  Model: {args.model}")
    # print(f"  GPU: {args.gpu}")

    # login wandb and huggingface
    login_wandb()
    login_huggingface()

    # load configuration from  .yaml for cpt intead of ArugumentParser
    cpt_config = load_config("cpt_config")
    sft_config = load_config("sft_config")

    # load model
    model, tokenizer = load_model(cpt_config["model_config"])
    # add lora adapters
    model = add_lora_adapters(model, cpt_config["lora"])

    # # load dataset

    # dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split = "train",)
    dataset = load_dataset_by_name(
        cpt_config["dataset"]["name"], cpt_config["dataset"]["language"], tokenizer
    )
    print(f"Loaded dataset: {dataset}")

    # Continued Pretraining

    checkpoint_path = get_checkpoint(cpt_config["paths"]["output_dir"])

    if checkpoint_path:
        print(f"Resuming from: {checkpoint_path}")
        trainer_stats = cpt_trainer(model, tokenizer, dataset, cpt_config).train(
            resume_from_checkpoint=checkpoint_path
        )
    else:
        print("Starting from scratch")
        trainer_stats = cpt_trainer(model, tokenizer, dataset, cpt_config).train()

    # # save the model lora adapters and tokenizer
    # Saving, loading finetuned models
    # To save the final model as LoRA adapters, either use Hugging Face's push_to_hub for an online save or save_pretrained for a local save.
    # Just LoRA adapters
    if True:
        model.save_pretrained(cpt_config["paths"]["final_model_local_save_dir_name"])
        tokenizer.save_pretrained(
            cpt_config["paths"]["final_model_local_save_dir_name"]
        )
    if True:
        model.push_to_hub(cpt_config["paths"]["final_model_hub_save_dir_name"])
        tokenizer.push_to_hub(cpt_config["paths"]["final_model_hub_save_dir_name"])

    # # [NOTE] This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF

    #     # Merge to 16bit
    # if False: model.save_pretrained_merged("mistral_v0_finetune_16bit", tokenizer, save_method = "merged_16bit",)
    # if False: model.push_to_hub_merged("HF_USERNAME/mistral_v0_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

    # # Merge to 4bit
    # if False: model.save_pretrained_merged("mistral_v0_finetune_4bit", tokenizer, save_method = "merged_4bit",)
    # if False: model.push_to_hub_merged("HF_USERNAME/mistral_v0_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

    print("Config loader loaded successfully.", load_config("cpt_config"))

    # Instruction Finetuning

    from datasets import load_dataset

    alpaca_dataset = load_dataset(
        "ravithejads/alpaca_urdu_cleaned_output", split="train"
    )

    # alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
    # ### Instruction:
    # {}

    # ### Response:
    # {}"""
    # Becomes:
    alpaca_prompt_urdu = """ذیل میں ایک ہدایت ہے جو ایک کام کی وضاحت کرتی ہے۔ ایک جواب لکھیں جو مناسب طریقے سے درخواست کو مکمل کرے۔ 
#### ہدایات: 
# {} 

#### جواب: 
# {}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        instruction = examples["urdu_instruction"]
        urdu_output = examples["urdu_output"]
        outputs = []
        for title, text in zip(instruction, urdu_output):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt_urdu.format(title, text) + EOS_TOKEN
            outputs.append(text)
        return {
            "text": outputs,
        }

    alpaca_dataset = alpaca_dataset.map(
        formatting_prompts_func,
        batched=True,
    )

 # Continued Pretraining

    checkpoint_path = get_checkpoint(sft_config["paths"]["output_dir"])

    if checkpoint_path:
        print(f"Resuming from: {checkpoint_path}")
        trainer_stats = sft_trainer(model, tokenizer, dataset, sft_config).train(
            resume_from_checkpoint=checkpoint_path
        )
    else:
        print("Starting from scratch")
        trainer_stats = sft_trainer(model, tokenizer, dataset, sft_config).train()
        
        
    if True:
        model.save_pretrained(sft_config["paths"]["final_model_local_save_dir_name"])
        tokenizer.save_pretrained(
            sft_config["paths"]["final_model_local_save_dir_name"]
        )
    if True:
        model.push_to_hub(sft_config["paths"]["final_model_hub_save_dir_name"])
        tokenizer.push_to_hub(sft_config["paths"]["final_model_hub_save_dir_name"])

if __name__ == "__main__":
    main()
