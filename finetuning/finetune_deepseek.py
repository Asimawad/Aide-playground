# START OF NEW FILE aide-ds/finetuning/finetune_deepseek.py
import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-7B model using QLoRA.")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/deepseek-coder-7b-instruct-v1.5", help="Hugging Face model ID.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the prepared JSONL dataset file.")
    parser.add_argument("--output_dir", type=str, default="./deepseek-7b-finetuned-adapters", help="Directory to save the fine-tuned adapters.")

    # --- Training Arguments ---
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log training status every N steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", help="Optimizer type.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision (if available).")
    parser.add_argument("--report_to", type=str, default="none", help="Where to report metrics (e.g., 'wandb', 'tensorboard', 'none').")

    # --- LoRA Arguments ---
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    # Common target modules for DeepSeek Coder v1.5 (verify if needed)
    parser.add_argument("--target_modules", nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help="Modules to apply LoRA to.")

    args = parser.parse_args()

    logger.info(f"Starting fine-tuning with arguments: {args}")

    # 1. Load Dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    try:
        # Assuming the dataset was prepared with a 'text' field containing the formatted prompt
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
        # Optional: split into train/eval
        # dataset = dataset.train_test_split(test_size=0.05)
        # train_dataset = dataset["train"]
        # eval_dataset = dataset["test"]
        logger.info(f"Dataset loaded: {dataset}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        return

    # 2. Load Tokenizer
    logger.info(f"Loading tokenizer for {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Set pad token if missing
        logger.info(f"Set tokenizer pad_token to eos_token: {tokenizer.eos_token}")
    tokenizer.padding_side = "right" # Default, important for some models

    # 3. Configure Quantization (QLoRA)
    # Check CUDA availability for BitsAndBytes
    use_4bit = torch.cuda.is_available()
    if use_4bit:
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True, # Nested quantization
        )
        logger.info(f"Using 4-bit QLoRA with compute_dtype={compute_dtype}")
    else:
        bnb_config = None
        logger.warning("CUDA not available. Loading model in full precision (no QLoRA).")

    # 4. Load Base Model
    logger.info(f"Loading base model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config if use_4bit else None,
        device_map="auto", # Automatically distribute across GPUs if available
        trust_remote_code=True,
        torch_dtype=compute_dtype if use_4bit else None # Set dtype if quantized
    )
    model.config.use_cache = False # Disable cache for training stability with gradient checkpointing
    if args.gradient_checkpointing:
        model.config.pretraining_tp = 1 # May be needed for gradient checkpointing compatibility

    # 5. Configure LoRA
    logger.info(f"Configuring LoRA with r={args.lora_r}, alpha={args.lora_alpha}, modules={args.target_modules}")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare model for k-bit training if using quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # Apply PEFT config - this modifies the model in place
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. Configure Training Arguments
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=not args.bf16 and use_4bit, # Use fp16 if not bf16 and quantized
        bf16=args.bf16 and use_4bit, # Use bf16 if specified and quantized/GPU supports
        max_grad_norm=0.3,
        max_steps=-1, # Override with num_train_epochs
        warmup_ratio=args.warmup_ratio,
        group_by_length=True, # Group sequences of similar length for efficiency
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        save_strategy="steps", # Save checkpoints based on steps
        save_total_limit=3, # Keep only the last few checkpoints
        # evaluation_strategy="steps", # If using eval dataset
        # eval_steps=args.save_steps, # Evaluate at same frequency as saving
    )

    # 7. Initialize Trainer
    # SFTTrainer handles packing sequences by default if max_seq_length is set
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset, # Use the full loaded dataset for training for now
        # eval_dataset=eval_dataset, # Uncomment if using eval split
        peft_config=peft_config,
        dataset_text_field="text", # Field containing the formatted prompt+completion
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False, # Set to True if dataset is large and sequences are short
                       # Set to False if using full prompts from prepare_data.py as 'text'
        # data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), # Use if packing=True
    )

    # 8. Train
    logger.info("Starting training...")
    trainer.train()

    # 9. Save Final Adapters
    logger.info(f"Training complete. Saving final adapters to {args.output_dir}")
    trainer.save_model(args.output_dir) # Saves only the LoRA adapters

    # Optional: Save tokenizer too
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Finetuning finished successfully.")

if __name__ == "__main__":
    main()
# END OF NEW FILE aide-ds/finetuning/finetune_deepseek.py