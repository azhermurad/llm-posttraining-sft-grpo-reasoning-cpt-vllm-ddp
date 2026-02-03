from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import UnslothTrainer, UnslothTrainingArguments


max_seq_length = 2048


def cpt_trainer(model, tokenizer, dataset):
    """
    Create and return an UnslothTrainer for training the model.
    Args:
        model (FastLanguageModel): The language model to be trained.
        tokenizer (Tokenizer): The tokenizer associated with the model.
        dataset: The dataset to be used for training.
    Returns:
        trainer (UnslothTrainer): The configured UnslothTrainer instance.  
        
    """
    
    trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,

        warmup_ratio = 0.1,
        # num_train_epochs = 1,
        max_steps = 50,
        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6,

        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "llama3_cpt_tinystories",
        save_strategy = "steps",
        save_steps = 10,
        save_total_limit = 2,
        # wanddb
        report_to = "wandb", # Use TrackIO/WandB etc
    ),
)
    
    # from transformers import EarlyStoppingCallback
    # early_stopping_callback = EarlyStoppingCallback(
    #     early_stopping_patience = 3,     # How many steps we will wait if the eval loss doesn't decrease
    #                                     # For example the loss might increase, but decrease after 3 steps
    #     early_stopping_threshold = 0.0,  # Can set higher - sets how much loss should decrease by until
    #                                     # we consider early stopping. For eg 0.01 means if loss was
    #                                     # 0.02 then 0.01, we consider to early stop the run.
    # )
    # trainer.add_callback(early_stopping_callback)
    return trainer