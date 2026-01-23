import argparse
import json
from addict import Addict
from copy import deepcopy
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from peft import get_peft_model
from utils import *


class BertDistillPipeline:
    """
    BERT Distillation Pipeline.
    """

    def __init__(self, **kwargs):
        args = Addict(kwargs)
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
        self.results = self.args.copy()
        self.training_params = dict(
            eval_strategy = "epoch",  # Enable evaluation every epoch
            logging_strategy = "steps",  # Enable logging
            logging_steps = 100,  # Log every 100 steps
            save_steps=0, # Don't save.
            save_total_limit=0, # Don't save.
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
        )

    def get_args(self):
        return self.args.copy()

    def save_results(self):
        if self.result_path.exists():
            print('Warning: Results overwritten')
        self.result_path.write_text(json.dumps(self.results, indent=4))
        print(f"Results written to {self.result_path}")

    @property
    def ckpt_dir(self):
        return self.dir / 'ckpt'
    
    @property
    def result_path(self):
        args = self.args
        config_name = f'{args.task}_{args.model_family}/' + \
                      f'{args.train_batch_size}_{args.teacher_learning_rate}_{args.weight_decay}/' + \
                      f'{args.peft}_{args.lora_alpha}_{args.lora_dropout}_{args.rank}.json'
        print(f"config_name: {config_name}")
        result_file = self.dir / 'metric' / config_name
        result_file.parent.mkdir(parents=True, exist_ok=True)
        return result_file

    def load_dataset(self):
        args = self.args
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
        args = self.args
        if lora_config is None:
            lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.lora_alpha,
                target_modules=get_target_modules(model_name),
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="SEQ_CLS"
            )
        # Apply LoRA configuration to the student model
        student_model = get_peft_model(student_model, lora_config)
        print("Loaded pretrained model with LoRA", model_name)
        return student_model

    def tokenize_teacher_dataset(self, teacher_dataset):
        args = self.args
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)
        tokenized_teacher_dataset = teacher_dataset.map(tokenize_function(args, teacher_tokenizer), batched=True)
        return tokenized_teacher_dataset

    def tokenize_student_dataset(self, teacher_dataset):
        args = self.args
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
        args = self.args
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
            output_dir=str(self.ckpt_dir / "lora"),
            learning_rate=args.student_learning_rate,
            **self.training_params,
        )
        if args.task == "mnli":
            eval_dataset = eval_dataset[0]
        callback = MemoryTrackingCallback()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
            callbacks=[callback],
        )
        train_output = trainer.train()
        print("Finished Training")
        return trainer, get_train_metrics(train_output, model, callback)

    def train_distill_lora(self, student_model, train_dataset, eval_dataset, teacher_soft_labels):
        args = self.args
        student_training_args = TrainingArguments(
            output_dir=str(self.ckpt_dir / "distill_lora"),
            learning_rate=args.student_learning_rate,
            remove_unused_columns=False,
            **self.training_params,
        )
        if args.task == "mnli":
            eval_dataset = eval_dataset[0]
        callback = MemoryTrackingCallback()
        student_trainer = DistillationTrainer(
            model=student_model,
            args=student_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
            callbacks=[callback],
        )
        student_trainer.teacher_soft_labels = teacher_soft_labels
        train_output = student_trainer.train()
        print("Finished Training")
        return student_trainer, get_train_metrics(train_output, student_model, callback)

    def train_fft(self, model, train_dataset, eval_dataset):
        args = self.args
        training_args = TrainingArguments(
            output_dir=str(self.ckpt_dir / "fft"),
            learning_rate=args.teacher_learning_rate,
            **self.training_params,
        )
        if args.task == "mnli":
            eval_dataset = eval_dataset[0]
        callback = MemoryTrackingCallback()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
            callbacks=[callback],
        )
        # Train the model
        train_output = trainer.train()
        print("Finished Training")
        return trainer, get_train_metrics(train_output, model, callback)

    def get_teacher_soft_labels(self, teacher_trainer, tokenized_teacher_dataset):
        teacher_logits = teacher_trainer.predict(tokenized_teacher_dataset["train"]).predictions
        teacher_soft_labels = torch.tensor(teacher_logits)
        return teacher_soft_labels

    def evaluate_model(self, trainer, eval_dataset):
        args = self.args
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

    def run(self):
        if self.result_path.exists():
            print(f"Result file already exists: {self.result_path}")
            return

        results = self.results
        teacher_dataset = self.load_dataset()
        args = self.args

        # 1. Teacher FFT
        print('Begin Teacher FFT...')
        teacher_model = self.load_pretrained_model(args.teacher_model_name)
        tokenized_teacher_dataset = self.tokenize_teacher_dataset(teacher_dataset)
        teacher_train_dataset, teacher_eval_dataset = self.split_dataset(tokenized_teacher_dataset)

        teacher_trainer, train_metrics = self.train_fft(teacher_model, teacher_train_dataset, teacher_eval_dataset)
        teacher_fft_results = self.evaluate_model(teacher_trainer, teacher_eval_dataset)
        teacher_fft_results['train'] = train_metrics
        print(f"teacher fft results: {teacher_fft_results}")
        teacher_soft_labels = self.get_teacher_soft_labels(teacher_trainer, tokenized_teacher_dataset)
        teacher_model.to('cpu')
        del teacher_model
        clear_gpu_memory()

        # 2. Student Distill + LoRA
        print("Begin Student Distill + LoRA...")
        peft_config = get_peft_config(args, args.student_model_name, args.peft)
        tokenized_student_dataset = self.tokenize_student_dataset(teacher_dataset)
        student_model = self.load_pretrained_model_lora(args.student_model_name, lora_config=peft_config)
        student_train_dataset, student_eval_dataset = self.split_dataset(tokenized_student_dataset)
        student_trainer, train_metrics = self.train_distill_lora(student_model, student_train_dataset,
                                                                 student_eval_dataset,
                                                                 teacher_soft_labels)
        student_lora_results = self.evaluate_model(student_trainer, student_eval_dataset)
        student_lora_results['train'] = train_metrics
        print(f"student lora results: {student_lora_results}")
        student_model.to('cpu')
        del student_model
        clear_gpu_memory()

        # 3. Teacher LoRA
        print("Begin Teacher LoRA...")
        peft_config = get_peft_config(args, args.teacher_model_name, args.peft)
        teacher_lora_model = self.load_pretrained_model_lora(args.teacher_model_name, lora_config=peft_config)
        teacher_lora_trainer, train_metrics = self.train_lora(teacher_lora_model, teacher_train_dataset,
                                                              teacher_eval_dataset)
        teacher_lora_results = self.evaluate_model(teacher_lora_trainer, teacher_eval_dataset)
        teacher_lora_results['train'] = train_metrics
        print(f"teacher lora results: {teacher_lora_results}")
        teacher_lora_model.to('cpu')
        del teacher_lora_model
        clear_gpu_memory()

        results.update(teacher_fft_results=teacher_fft_results, teacher_lora_results=teacher_lora_results,
                       student_lora_results=student_lora_results)
        self.save_results()
        print('Finish!!')


def main(args):
    # args serves as default.
    for model_family in MODEL_FAMILY.keys():
        for peft_method in PEFT_FAMILY:
            for task in GLUE_TASKS:
                config = args.__dict__.copy()
                config['model_family'] = model_family
                config['task'] = task
                config['peft'] = peft_method
                add_models(model_family, config)
                pipe = BertDistillPipeline(**config)
                pipe.run()


def main_single(args):
    pipe = BertDistillPipeline(**args.__dict__)
    pipe.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation with LoRA-enhanced Student Model")

    # Model arguments
    parser.add_argument("--teacher_model_name", type=str, default="./models/bert-base-uncased",
                        help="Name of the teacher model")
    parser.add_argument("--student_model_name", type=str, default="./models/distilbert-base-uncased",
                        help="Name of the student model")

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
    parser.add_argument("--student_learning_rate", type=float, default=5e-4, help="Learning rate for the student model")
    parser.add_argument('--task', type=str, default="wnli", choices=tuple(GLUE_TASKS), help="Name of the task")
    parser.add_argument('--peft', type=str, default="lora", choices=tuple(PEFT_FAMILY), help="PEFT method name")

    main(parser.parse_args())
