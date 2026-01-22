import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType

from utils import glue_tasks, compute_metrics, tokenize_function, get_num_labels


def main(args):
    # Print out the configuration for tracking and debugging
    print(f"Dataset path: {args.dataset_path}")
    print(f"Dir Name: {args.dir_name}")
    print(f"Model name: {args.model_name}")
    print(f"Learning rate: 5e-4")
    print(f"Number of training epochs: {args.num_train_epochs}")
    print(f"Rank: {args.rank}")
    print(f"LoRA Alpha: {args.lora_alpha}")
    print(f"LoRA Dropout: {args.lora_dropout}")

    # Determine the number of labels based on the GLUE task
    num_labels = get_num_labels(args)

    # Load dataset from disk and display its features for reference
    dataset = load_dataset('glue', args.task, cache_dir=args.dataset_path)
    print(dataset)
    print(dataset["train"].features)

    # Load tokenizer for the specified pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Apply tokenization to the dataset
    tokenized_datasets = dataset.map(tokenize_function(args, tokenizer), batched=True)

    # Split dataset into training and evaluation sets
    if args.task == "mnli":
        # MNLI requires two separate validation sets
        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_matched_dataset = tokenized_datasets["validation_matched"].shuffle(seed=42)
        eval_mismatched_dataset = tokenized_datasets["validation_mismatched"].shuffle(seed=42)
    else:
        # Standard split for other tasks
        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

    # Load the model for sequence classification
    if args.task == "stsb" or args.task == "mnli":
        # Set 'ignore_mismatched_sizes' to True for STS-B and MNLI tasks
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # Configure LoRA parameters
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],  # Apply LoRA to these model modules
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"  # Task type for LoRA configuration
    )

    # Apply LoRA configuration to the model
    model_lora = get_peft_model(model, lora_config)

    # Define lists to track memory usage after each epoch
    epochs = []
    memory_allocated = []
    memory_reserved = []

    # Callback class to track and log GPU memory usage
    class MemoryTrackingCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            allocated_memory = torch.cuda.memory_allocated() / 1e6  # Convert to MB
            reserved_memory = torch.cuda.memory_reserved() / 1e6  # Convert to MB
            epochs.append(state.epoch)
            memory_allocated.append(allocated_memory)
            memory_reserved.append(reserved_memory)
            print(f"Epoch {state.epoch}: Allocated Memory: {allocated_memory:.2f} MB, Reserved Memory: {reserved_memory:.2f} MB")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_steps=0,
        save_total_limit=0,
        logging_steps=0,
        logging_strategy="no"
    )

    # Initialize Trainer with LoRA model, training arguments, datasets, and memory tracking
    if args.task == "mnli":
        trainer = Trainer(
            model=model_lora,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_matched_dataset,
            compute_metrics=compute_metrics(args),
            callbacks=[MemoryTrackingCallback()]
        )
    else:
        trainer = Trainer(
            model=model_lora,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
            callbacks=[MemoryTrackingCallback()]
        )

    # Train the model
    trainer.train()

    # Evaluate model on validation datasets for MNLI
    if args.task == "mnli":
        eval_results_matched = trainer.evaluate(eval_dataset=eval_matched_dataset)
        print(f"Evaluation results (matched): {eval_results_matched}")

        eval_results_mismatched = trainer.evaluate(eval_dataset=eval_mismatched_dataset)
        print(f"Evaluation results (mismatched): {eval_results_mismatched}")

        # Combine evaluation results for matched and mismatched datasets
        combined_results = {
            "matched_accuracy": eval_results_matched["eval_accuracy"],
            "mismatched_accuracy": eval_results_mismatched["eval_accuracy"]
        }
        print(f"Combined evaluation results: {combined_results}")
    else:    
        # Evaluate on single validation set for other tasks
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Combined evaluation results: {eval_results}")

    # Optionally save the fine-tuned model and tokenizer
    # output_dir = "./fine_tuned_model"
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Set up argument parsing for customizable inputs
    parser = argparse.ArgumentParser(description="Fine-tune BERT model on GLUE benchmark and recording memory usage")

    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to the dataset")
    parser.add_argument("--dir_name", type=str, default="./finetuned", help="Directory name for saving models")
    parser.add_argument("--model_name", type=str, default="./models/bert-base-uncased", help="Model name or path")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument('--task', type=str, default="wnli", choices=tuple(glue_tasks))

    args = parser.parse_args()
    main(args)
