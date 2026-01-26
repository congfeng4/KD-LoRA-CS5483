import argparse
import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import *

# Suppress tokenizer warning about overflowing tokens not returned for 'longest_first' truncation strategy
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def main(args):
    # Print input arguments for clarity
    print(f"Dataset path: {args.task}")
    print(f"Model name: {args.model_name}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Training batch size: {args.train_batch_size}")
    print(f"Evaluation batch size: {args.eval_batch_size}")
    print(f"Number of training epochs: {args.num_train_epochs}")
    print(f"Weight decay: {args.weight_decay}")

    # Set the number of labels based on the dataset type
    num_labels = get_num_labels(args)

    # Load dataset from disk and display its features
    dataset = load_dataset('glue', args.task, cache_dir=args.dataset_path)
    print(dataset)
    print(dataset["train"].features)

    # Load the tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Apply the tokenization function to the dataset
    tokenized_datasets = dataset.map(tokenize_function(args, tokenizer), batched=True)

    # Split tokenized datasets for training and evaluation
    if args.task == "mnli":
        # MNLI requires evaluation on both matched and mismatched datasets
        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_matched_dataset = tokenized_datasets["validation_matched"].shuffle(seed=42)
        eval_mismatched_dataset = tokenized_datasets["validation_mismatched"].shuffle(seed=42)
    else:
        # Standard train-validation split for other datasets
        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

    # Load model for sequence classification with appropriate number of labels
    if args.task == "stsb" or args.task == "mnli":
        # Ignore mismatched sizes for STS-B and MNLI tasks
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels,
                                                                   ignore_mismatched_sizes=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # Set up training arguments for the model
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_steps=0,
        save_total_limit=0,
        logging_steps=0,
        logging_strategy="no",
    )

    # Set up Trainer with model, training arguments, datasets, and metrics
    if args.task == "mnli":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_matched_dataset,
            compute_metrics=compute_metrics(args),
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
        )

    # Train the model
    trainer.train()
    print("Trainable parameters:", get_trainable_param_count(model))

    # Evaluate model and print results, handling MNLI separately
    if args.task == "mnli":
        eval_results_matched = trainer.evaluate(eval_dataset=eval_matched_dataset)
        print(f"Evaluation results (matched): {eval_results_matched}")

        eval_results_mismatched = trainer.evaluate(eval_dataset=eval_mismatched_dataset)
        print(f"Evaluation results (mismatched): {eval_results_mismatched}")

        combined_results = {
            "matched_accuracy": eval_results_matched["eval_accuracy"],
            "mismatched_accuracy": eval_results_mismatched["eval_accuracy"]
        }

        print(f"Combined evaluation results: {combined_results}")
    else:
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Combined evaluation results: {eval_results}")

    # Uncomment to save fine-tuned model and tokenizer
    # output_dir = "./fine_tuned_model"
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune BERT model on GLUE benchmark")

    # Add required arguments with default values
    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to the dataset")
    parser.add_argument("--model_name", type=str, default="./models/bert-base-uncased", help="Model name or path")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument('--task', type=str, default="wnli", choices=tuple(GLUE_TASKS))

    args = parser.parse_args()
    main(args)
