import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

glue_tasks = [
    "cola", "sst2", "mrpc", "qqp", "stsb",
    "mnli", "qnli", "rte", "wnli",
]


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
    if args.task == "stsb":
        num_labels = 1  # Regression task for STS-B
    elif args.task == "mnli":
        num_labels = 3  # Multi-class classification for MNLI
    else:
        num_labels = 2  # Default binary classification

    # Load dataset from disk and display its features
    dataset = load_dataset('glue', args.task, cache_dir=args.dataset_path)
    print(dataset)
    print(dataset["train"].features)

    # Load the tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Function to tokenize dataset based on task-specific requirements
    def tokenize_function(examples):
        if args.task == "mrpc":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if args.task == "cola":
            return tokenizer(examples["sentence"], truncation=True, padding="max_length")
        if args.task == "sst2":
            return tokenizer(examples["sentence"], truncation=True, padding="max_length")
        if args.task == "wnli":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if args.task == "rte":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if args.task == "qqp":
            return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True)
        if args.task == "stsb":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if args.task == "qnli":
            return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True)
        if args.task == "mnli":
            return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True)

    # Apply the tokenization function to the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

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
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
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
        logging_strategy="no"
    )

    # Define function to compute evaluation metrics
    def compute_metrics(p):
        predictionss, labels = p
        predictions = np.argmax(predictionss, axis=1)

        # Choose metrics based on dataset type
        if args.task == "mrpc":
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="binary")
            return {"accuracy": accuracy, "f1": f1}
        if args.task == "cola":
            mcc = matthews_corrcoef(labels, predictions)
            return {"matthews_correlation": mcc}
        if args.task == "sst2":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if args.task == "wnli":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if args.task == "rte":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if args.task == "qqp":
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="binary")
            return {"accuracy": accuracy, "f1": f1}
        if args.task == "stsb":
            preds = predictionss.squeeze().tolist()
            labels = labels.tolist()
            pearson_corr, _ = pearsonr(preds, labels)
            spearman_corr, _ = spearmanr(preds, labels)
            return {
                'pearson': pearson_corr,
                'spearman': spearman_corr
            }
        if args.task == "qnli":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if args.task == "mnli":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}

    # Set up Trainer with model, training arguments, datasets, and metrics
    if args.task == "mnli":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_matched_dataset,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

    # Train the model
    trainer.train()

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
        trainer.evaluate()

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
    parser.add_argument('--task', type=str, default="wnli", choices=tuple(glue_tasks))

    args = parser.parse_args()
    main(args)
