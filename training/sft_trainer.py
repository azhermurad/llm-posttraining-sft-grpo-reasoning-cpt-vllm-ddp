from unsloth import UnslothTrainer, UnslothTrainingArguments,is_bfloat16_supported



def sft_trainer(model, tokenizer, dataset,config):
    """
    Create and return an UnslothTrainer for training the model.
    Args:
        model (FastLanguageModel): The language model to be trained.
        tokenizer (Tokenizer): The tokenizer associated with the model.
        dataset: The dataset to be used for training.
        config: cpt configuration
        
    Returns:
        trainer (UnslothTrainer): The configured UnslothTrainer instance.  
        
    """
    
    trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = int(config['model_config']['max_seq_length']),
    dataset_num_proc =  8,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = int(config['training']['per_device_train_batch_size']),
        gradient_accumulation_steps = int(config['training']['gradient_accumulation_steps'] ),
        
        
        # Use warmup_ratio and num_train_epochs for longer runs!
        max_steps = int(config['training']['max_steps']),
        warmup_steps = int(config['training']['max_steps']),
        # warmup_ratio = config['training']["warmup_ratio"],
        # num_train_epochs = config['training']['num_train_epochs'],


        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate =  float(config['training']["learning_rate"]),
        embedding_learning_rate =  float(config['training']["embedding_learning_rate"]),

        logging_steps = 1,
        optim =  config['training']['optim'],
        weight_decay = 0.001,
        lr_scheduler_type = config["training"]['lr_scheduler_type'],
        seed = 3407,
        
        output_dir = config["paths"]["output_dir"],
        save_strategy = "steps",
        save_steps =int(config["training"]["save_steps"] ),
        save_total_limit = int(config["training"]["save_total_limit"]),
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        # load_best_model_at_end = True, # MUST USE for early stopping
        # wanddb
        report_to = config["training"]["report_to"], # Use TrackIO/WandB etc
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