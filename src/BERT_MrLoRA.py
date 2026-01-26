import argparse
import logging
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from mrlora import MrLoraConfig
from utils import *

# Suppress tokenizer warning about overflowing tokens not returned for 'longest_first' truncation strategy
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def main(args):
    # Determine labels
    num_labels = get_num_labels(args)

    # Load dataset and tokenizer
    # dataset = load_dataset('glue', args.task, cache_dir=args.dataset_path)
    dataset = load_glue_dataset(args.dataset_path, args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_datasets = dataset.map(tokenize_function(args, tokenizer), batched=True)
    print(tokenized_datasets)

    # Prepare model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # 1. Define Mr. LoRA Config instead of standard LoraConfig
    mr_lora_config = MrLoraConfig(
        ranks=args.ranks,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=args.lora_dropout,
        task_type="SEQ_CLS"
    )

    # 2. Apply Mr. LoRA
    # Using the custom tuner:
    model_lora = get_peft_model(model, mr_lora_config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=32,
        num_train_epochs=args.num_train_epochs,
        logging_strategy="epoch",  # Changed from "no" to see progress
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    print('training_args', training_args)

    # Initialize Trainer
    trainer = Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics(args),
    )

    trainer.train()
    print(trainer.evaluate())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./dataset")
    parser.add_argument("--model_name", type=str, default="./models/bert-base-uncased")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument('--task', type=str, default="wnli")
    parser.add_argument('--ranks', type=int, nargs='+', default=[32, 16, 8, 4, 2])

    main(parser.parse_args())
