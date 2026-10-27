"""Training pipeline for TacticaLM using QLoRA."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_training(
    train_data_path: Path | str,
    output_dir: Path | str,
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    num_epochs: int = 2,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
):
    """Run QLoRA instruction fine-tuning using TRL's SFTTrainer.
    
    Args:
        train_data_path: Path to the training dataset JSON.
        output_dir: Directory to save the fine-tuned adapter.
        model_name: Base model huggingface ID.
        num_epochs: Number of training epochs.
        batch_size: Per device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Peak learning rate.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        seed: Random seed.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        logger.error(f"Missing required ML dependencies for training: {e}")
        logger.error("Make sure torch, transformers, peft, trl, datasets, and bitsandbytes are installed.")
        raise
        
    logger.info(f"Starting QLoRA training for {model_name}")
    
    # 1. Load Data
    with open(train_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Format for ChatML/Instruct (TRL expects a text column or messages column)
    # Since Qwen-Instruct uses chat format, we construct a conversational dataset.
    formatted_data = []
    for item in data:
        instruction = item["instruction"]
        response = item["response"]
        
        # We can pass raw text if we define our formatting function, but TRL prefers the standard text column
        # Let's format it directly as standard Qwen chatml
        prompt = (
            f"<|im_start|>system\n{instruction.split('Match Context:')[0].strip()}<|im_end|>\n"
            f"<|im_start|>user\nMatch Context:{instruction.split('Match Context:')[1].strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )
        formatted_data.append({"text": prompt})
        
    dataset = Dataset.from_list(formatted_data)
    logger.info(f"Loaded {len(dataset)} training examples.")
    
    # 2. BitsAndBytes Config for 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 4. Load Model
    logger.info("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # 5. LoRA Config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    from transformers import TrainingArguments
    
    # 6. Training Arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        seed=seed,
        max_length=2048,
        dataset_text_field="text",
    )
    
    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    
    # 8. Train
    logger.info("Starting training...")
    trainer.train()
    
    # 9. Save
    logger.info(f"Saving final adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    
    return str(output_dir)
