import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from utils import GLUE_TASKS, compute_metrics, tokenize_function, get_num_labels


def main(args):
    # Print out the configuration for tracking and debugging
    print(f"Dataset path: {args.dataset_path}")
    print(f"Dir Name: {args.dir_name}")
    print(f"Model name: {args.teacher_model_name}")
    print(f"Model name: {args.student_model_name}")
    print(f"Learning rate: 5e-4")
    print(f"Number of training epochs: {args.num_train_epochs}")
    print(f"Rank: {args.rank}")
    print(f"LoRA Alpha: {args.lora_alpha}")
    print(f"LoRA Dropout: {args.lora_dropout}")

    # Determine the number of labels based on the GLUE task
    num_labels = get_num_labels(args)

    # Step 1: Fine-tune a Teacher Model
    print(f"Fine-tuning the teacher model: {args.teacher_model_name}")
    # teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_name, num_labels=num_labels)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)

    if args.task == "stsb" or args.task == "mnli":
        # Set 'ignore_mismatched_sizes' to True for STS-B and MNLI tasks
        teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_name,
                                                                           num_labels=num_labels,
                                                                           ignore_mismatched_sizes=True)
    else:
        teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_name,
                                                                           num_labels=num_labels)

    teacher_training_args = TrainingArguments(
        output_dir="./teacher_results",
        learning_rate=args.teacher_learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
    )

    teacher_dataset = load_dataset('glue', args.task, cache_dir=args.dataset_path)
    tokenized_teacher_dataset = teacher_dataset.map(tokenize_function(args, teacher_tokenizer), batched=True)

    if args.task == "mnli":
        # MNLI requires two separate validation sets
        train_dataset = tokenized_teacher_dataset["train"].shuffle(seed=42)
        eval_matched_dataset = tokenized_teacher_dataset["validation_matched"].shuffle(seed=42)
        eval_mismatched_dataset = tokenized_teacher_dataset["validation_mismatched"].shuffle(seed=42)
    else:
        # Standard split for other tasks
        train_dataset = tokenized_teacher_dataset["train"].shuffle(seed=42)
        eval_dataset = tokenized_teacher_dataset["validation"].shuffle(seed=42)

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
            print(
                f"Epoch {state.epoch}: Allocated Memory: {allocated_memory:.2f} MB, Reserved Memory: {reserved_memory:.2f} MB")

    if args.task == "mnli":
        teacher_trainer = Trainer(
            model=teacher_model,
            args=teacher_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_matched_dataset,
            compute_metrics=compute_metrics(args),
            callbacks=[MemoryTrackingCallback()]
        )
    else:
        teacher_trainer = Trainer(
            model=teacher_model,
            args=teacher_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
            callbacks=[MemoryTrackingCallback()]
        )

    teacher_trainer.train()
    print('Teacher training results:')
    if args.task == "mnli":
        eval_results_matched = teacher_trainer.evaluate(eval_dataset=eval_matched_dataset)
        print(f"Evaluation results (matched): {eval_results_matched}")

        eval_results_mismatched = teacher_trainer.evaluate(eval_dataset=eval_mismatched_dataset)
        print(f"Evaluation results (mismatched): {eval_results_mismatched}")

        combined_results = {
            "matched_accuracy": eval_results_matched["eval_accuracy"],
            "mismatched_accuracy": eval_results_mismatched["eval_accuracy"]
        }

        print(f"Combined evaluation results: {combined_results}")
    else:
        eval_results = teacher_trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Combined evaluation results: {eval_results}")

    # Save teacher model predictions (logits) as soft labels
    teacher_logits = teacher_trainer.predict(tokenized_teacher_dataset["train"]).predictions
    teacher_soft_labels = torch.tensor(teacher_logits)

    # Step 2: Initialize a Smaller Student Model with LoRA
    print(f"Initializing student model: {args.student_model_name} with LoRA")
    student_model = AutoModelForSequenceClassification.from_pretrained(args.student_model_name, num_labels=num_labels)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_lin", "v_lin"],
        # target_modules=["query", "value"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )

    # Apply LoRA configuration to the student model
    student_model = get_peft_model(student_model, lora_config)

    # Freeze all layers except LoRA parameters
    for param in student_model.parameters():
        param.requires_grad = False
    for name, param in student_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True  # Only LoRA weights are trainable

    # Step 3: Distillation from Teacher to Student
    print("Starting knowledge distillation from teacher to student")
    student_training_args = TrainingArguments(
        output_dir="./student_results",
        learning_rate=args.student_learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        remove_unused_columns=False,
        eval_strategy="epoch",
    )

    def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
        # Compute the distillation loss with temperature scaling
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean"
        ) * (temperature ** 2)
        hard_loss = F.cross_entropy(student_logits, labels)
        return alpha * soft_loss + (1 - alpha) * hard_loss

    # Define a custom training loop for distillation
    class DistillationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwds):
            labels = inputs.pop("labels")
            idx = inputs.pop('idx').long().cpu()
            outputs = model(**inputs)
            student_logits = outputs.logits
            teacher_logits = teacher_soft_labels[idx]  # Align teacher logits with batch size
            # teacher_logits = teacher_soft_labels[inputs["input_ids"].shape[0]]  # Align teacher logits with batch size
            teacher_logits = teacher_logits.to(student_logits.device)
            loss = distillation_loss(student_logits, teacher_logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Tokenize student dataset
    tokenized_student_dataset = teacher_dataset.map(tokenize_function(args, student_tokenizer, with_indices=True),
                                                    with_indices=True, batched=True)
    print(tokenized_student_dataset['train'][0]['idx'])

    # Initialize Distillation Trainer
    # student_trainer = DistillationTrainer(
    #     model=student_model,
    #     args=student_training_args,
    #     train_dataset=tokenized_student_dataset["train"],
    #     eval_dataset=tokenized_student_dataset["validation"]
    # )
    if args.task == "mnli":
        # MNLI requires two separate validation sets
        train_dataset = tokenized_student_dataset["train"].shuffle(seed=42)
        eval_matched_dataset = tokenized_student_dataset["validation_matched"].shuffle(seed=42)
        eval_mismatched_dataset = tokenized_student_dataset["validation_mismatched"].shuffle(seed=42)
    else:
        # Standard split for other tasks
        train_dataset = tokenized_student_dataset["train"].shuffle(seed=42)
        eval_dataset = tokenized_student_dataset["validation"].shuffle(seed=42)

    if args.task == "mnli":
        student_trainer = DistillationTrainer(
            model=student_model,
            args=student_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_matched_dataset,
            compute_metrics=compute_metrics(args),
            callbacks=[MemoryTrackingCallback()]
        )
    else:
        student_trainer = DistillationTrainer(
            model=student_model,
            args=student_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
            callbacks=[MemoryTrackingCallback()]
        )

    # print(tokenized_student_dataset['train'][0]['idx'])

    # Train student model with knowledge distillation
    student_trainer.train()

    # Evaluate student model
    if args.task == "mnli":
        eval_results_matched = student_trainer.evaluate(eval_dataset=eval_matched_dataset)
        print(f"Evaluation results (matched): {eval_results_matched}")

        eval_results_mismatched = student_trainer.evaluate(eval_dataset=eval_mismatched_dataset)
        print(f"Evaluation results (mismatched): {eval_results_mismatched}")

        # Combine evaluation results for matched and mismatched datasets
        combined_results = {
            "matched_accuracy": eval_results_matched["eval_accuracy"],
            "mismatched_accuracy": eval_results_mismatched["eval_accuracy"]
        }
        print(f"Combined evaluation results: {combined_results}")
    else:
        # Evaluate on single validation set for other tasks
        eval_results = student_trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Combined evaluation results: {eval_results}")

    # Save the fine-tuned LoRA student model
    output_dir = "./fine_tuned_student_model"
    student_model.save_pretrained(output_dir)
    student_tokenizer.save_pretrained(output_dir)
    print(f"Student model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation with LoRA-enhanced Student Model")

    # Model arguments
    parser.add_argument("--teacher_model_name", type=str, default="./models/bert-base-uncased",
                        help="Name of the teacher model")
    parser.add_argument("--student_model_name", type=str, default="./models/distilbert-base-uncased",
                        help="Name of the student model")
    output_dir = "./results",

    # Dataset and training parameters
    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to the dataset")
    # parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for the classification task")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--dir_name", type=str, default="./finetuned", help="Directory name for saving models")

    # LoRA parameters
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA layers")

    # Learning rates for teacher and student
    parser.add_argument("--teacher_learning_rate", type=float, default=5e-5, help="Learning rate for the teacher model")
    parser.add_argument("--student_learning_rate", type=float, default=5e-5, help="Learning rate for the student model")
    parser.add_argument('--task', type=str, default="wnli", choices=tuple(GLUE_TASKS))

    args = parser.parse_args()
    main(args)
