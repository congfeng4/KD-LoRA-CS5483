import argparse
from copy import deepcopy
from pathlib import Path
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

from utils import glue_tasks, compute_metrics, tokenize_function, get_num_labels


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
    teacher_soft_labels: torch.Tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwds):
        labels = inputs.pop("labels")
        idx = inputs.pop('idx').long().cpu()
        outputs = model(**inputs)
        student_logits = outputs.logits
        teacher_logits = self.teacher_soft_labels[idx]  # Align teacher logits with batch size
        # teacher_logits = teacher_soft_labels[inputs["input_ids"].shape[0]]  # Align teacher logits with batch size
        teacher_logits = teacher_logits.to(student_logits.device)
        loss = distillation_loss(student_logits, teacher_logits, labels)
        return (loss, outputs) if return_outputs else loss


# Callback class to track and log GPU memory usage
class MemoryTrackingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.memory_allocated = []
        self.memory_reserved = []

    def on_epoch_end(self, args, state, control, **kwargs):
        allocated_memory = torch.cuda.memory_allocated() / 1e6  # Convert to MB
        reserved_memory = torch.cuda.memory_reserved() / 1e6  # Convert to MB
        self.epochs.append(state.epoch)
        self.memory_allocated.append(allocated_memory)
        self.memory_reserved.append(reserved_memory)
        print(f"Epoch {state.epoch}: Allocated Memory: {allocated_memory:.2f} MB, Reserved Memory: {reserved_memory:.2f} MB")


class BertDistillPipeline:
    """
    BERT Distillation Pipeline.
    """
    def __init__(self, args):
        self.args = args
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
        self.num_labels = get_num_labels(args)
        self.dir = Path(args.dir_name)

    def get_args(self):
        return self.args.__dict__.copy()

    def load_dataset(self):
        teacher_dataset = load_dataset('glue', args.task, cache_dir=args.dataset_path)
        print('Loaded dataset', teacher_dataset, 'task', self.args.task)
        return teacher_dataset

    def load_pretrained_model(self, model_name):
        args = self.args
        num_labels = self.num_labels
        if args.task == "stsb" or args.task == "mnli":
            # Set 'ignore_mismatched_sizes' to True for STS-B and MNLI tasks
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                                       ignore_mismatched_sizes=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        print('Loaded pretrained model', model_name)
        return model

    def load_pretrained_model_lora(self, model_name, lora_config=None):
        student_model = self.load_pretrained_model(model_name)
        if lora_config is None:
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
        print("Loaded pretrained model with LoRA", student_model)
        return student_model

    def tokenize_teacher_dataset(self, teacher_dataset):
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)
        tokenized_teacher_dataset = teacher_dataset.map(tokenize_function(args, teacher_tokenizer), batched=True)
        return tokenized_teacher_dataset

    def tokenize_student_dataset(self, teacher_dataset):
        student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name)
        tokenized_student_dataset = teacher_dataset.map(tokenize_function(args, student_tokenizer, with_indices=True),
                                                        with_indices=True, batched=True)
        return tokenized_student_dataset

    def split_dataset(self, tokenized_datasets):
        """
        Prepare train/validation split
        :param tokenized_datasets:
        :return:
        """
        if args.task == "mnli":
            # MNLI requires evaluation on both matched and mismatched datasets
            train_dataset = tokenized_datasets["train"].shuffle(seed=42)
            eval_matched_dataset = tokenized_datasets["validation_matched"].shuffle(seed=42)
            eval_mismatched_dataset = tokenized_datasets["validation_mismatched"].shuffle(seed=42)
            return train_dataset, (eval_matched_dataset, eval_mismatched_dataset)
        else:
            # Standard train-validation split for other datasets
            train_dataset = tokenized_datasets["train"].shuffle(seed=42)
            eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)
            return train_dataset, eval_dataset

    def train_lora(self, model, train_dataset, eval_dataset):
        args = self.args
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(self.dir / "lora"),
            eval_strategy="epoch",
            learning_rate=args.teacher_learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            save_steps=0,
            save_total_limit=0,
            logging_steps=0,
            logging_strategy="no"
        )
        if args.task == "mnli":
            eval_dataset = eval_dataset[0]
        # callback = MemoryTrackingCallback()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
            # callbacks=[callback],
        )
        trainer.train()
        print("Finished Training")
        return trainer

    def train_distill_lora(self, student_model, train_dataset, eval_dataset, teacher_soft_labels):
        args = self.args
        student_training_args = TrainingArguments(
            output_dir=str(self.dir / "distill_lora"),
            learning_rate=args.student_learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            remove_unused_columns=False,
        )
        if args.task == "mnli":
            eval_dataset = eval_dataset[0]
        student_trainer = DistillationTrainer(
            model=student_model,
            args=student_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
            # callbacks=[MemoryTrackingCallback()]
        )
        student_trainer.teacher_soft_labels = teacher_soft_labels
        student_trainer.train()
        print("Finished Training")
        return student_trainer

    def train_fft(self, model, train_dataset, eval_dataset):
        args = self.args
        training_args = TrainingArguments(
            output_dir=str(self.dir / "fft"),
            eval_strategy="epoch",
            learning_rate=args.teacher_learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            save_steps=0,
            save_total_limit=0,
            logging_steps=0,
            logging_strategy="no"
        )
        if args.task == "mnli":
            eval_dataset = eval_dataset[0]
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
        )
        # Train the model
        trainer.train()
        print("Finished Training")
        return trainer

    def get_teacher_soft_labels(self, teacher_trainer, tokenized_teacher_dataset):
        teacher_logits = teacher_trainer.predict(tokenized_teacher_dataset["train"]).predictions
        teacher_soft_labels = torch.tensor(teacher_logits)
        return teacher_soft_labels

    def evaluate_model(self, trainer, eval_dataset):
        if args.task == "mnli":
            eval_matched_dataset, eval_mismatched_dataset = eval_dataset
            eval_results_matched = trainer.evaluate(eval_dataset=eval_matched_dataset)
            print(f"Evaluation results (matched): {eval_results_matched}")
            eval_results_mismatched = trainer.evaluate(eval_dataset=eval_mismatched_dataset)
            print(f"Evaluation results (mismatched): {eval_results_mismatched}")

            eval_results = {
                "matched_accuracy": eval_results_matched["eval_accuracy"],
                "mismatched_accuracy": eval_results_mismatched["eval_accuracy"]
            }
        else:
            eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Combined evaluation results: {eval_results}")
        eval_results['log_history'] = deepcopy(trainer.state.log_history)
        return eval_results


def main(args):
    model_family = Path(args.teacher_model_name).name
    config_name = f'{args.task}_{model_family}/{args.train_batch_size}_{args.teacher_learning_rate}/' + \
        f'{args.lora_alpha}_{args.lora_dropout}_{args.rank}.json'
    print(f"config_name: {config_name}")

    pipe = BertDistillPipeline(args)
    result_file = pipe.dir / config_name
    if result_file.exists():
        print(f"Result file already exists: {result_file}")
        return

    results = pipe.get_args()
    results['model_family'] = model_family
    teacher_dataset = pipe.load_dataset()

    # 1. Teacher FFT
    teacher_model = pipe.load_pretrained_model(args.teacher_model_name)
    tokenized_teacher_dataset = pipe.tokenize_teacher_dataset(teacher_dataset)
    teacher_train_dataset, teacher_eval_dataset = pipe.split_dataset(tokenized_teacher_dataset)

    teacher_trainer = pipe.train_fft(teacher_model, teacher_train_dataset, teacher_eval_dataset)
    teacher_fft_results = pipe.evaluate_model(teacher_trainer, teacher_eval_dataset)
    print(f"teacher fft results: {teacher_fft_results}")

    # 2. Student Distill + LoRA
    teacher_soft_labels = pipe.get_teacher_soft_labels(teacher_trainer, tokenized_teacher_dataset)
    tokenized_student_dataset = pipe.tokenize_student_dataset(teacher_dataset)
    student_model = pipe.load_pretrained_model_lora(args.student_model_name, lora_config=None)
    student_train_dataset, student_eval_dataset = pipe.split_dataset(tokenized_student_dataset)
    student_trainer = pipe.train_distill_lora(student_model, teacher_train_dataset, teacher_eval_dataset,
                                              teacher_soft_labels)
    student_lora_results = pipe.evaluate_model(student_trainer, student_eval_dataset)
    print(f"student lora results: {student_lora_results}")

    # 3. Teacher LoRA
    teacher_lora_model = pipe.load_pretrained_model_lora(args.teacher_model_name)
    teacher_lora_trainer = pipe.train_lora(teacher_lora_model, teacher_train_dataset, teacher_eval_dataset)
    teacher_lora_results = pipe.evaluate_model(teacher_lora_trainer, teacher_eval_dataset)
    print(f"teacher lora results: {teacher_lora_results}")

    results.update(teacher_fft_results=teacher_fft_results, teacher_lora_results=teacher_lora_results,
                   student_lora_results=student_lora_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation with LoRA-enhanced Student Model")

    # Model arguments
    parser.add_argument("--teacher_model_name", type=str, default="./models/bert-base-uncased", help="Name of the teacher model")
    parser.add_argument("--student_model_name", type=str, default="./models/distilbert-base-uncased", help="Name of the student model")

    # Dataset and training parameters
    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    parser.add_argument("--dir_name", type=str, default="./results", help="Directory name for saving models")

    # LoRA parameters
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA layers")

    # Learning rates for teacher and student
    parser.add_argument("--teacher_learning_rate", type=float, default=5e-5, help="Learning rate for the teacher model")
    parser.add_argument("--student_learning_rate", type=float, default=5e-5, help="Learning rate for the student model")
    parser.add_argument('--task', type=str, default="wnli", choices=tuple(glue_tasks))

    args = parser.parse_args()
    main(args)
