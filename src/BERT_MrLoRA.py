import argparse
from datasets import load_dataset
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from mrlora import MrLoraConfig
from utils import *


def main(args):
    # Determine labels
    num_labels = get_num_labels(args)

    # Load dataset and tokenizer
    dataset = load_dataset('glue', args.task, cache_dir=args.dataset_path)
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
    # model_lora = MrLoraModel(model, mr_lora_config, "default")
    model_lora = get_peft_model(model, mr_lora_config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=32,
        num_train_epochs=args.num_train_epochs,
        logging_strategy="steps",  # Changed from "no" to see progress
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics(args),
    )

    trainer.train()


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
