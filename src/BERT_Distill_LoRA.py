import argparse
import json
import shutil

from addict import Addict
from copy import deepcopy
from pathlib import Path

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
        print(f"Teacher learning rate: {args.teacher_learning_rate}")
        print(f"Student learning rate: {args.student_learning_rate}")
        print(f"Number of training epochs: {args.num_train_epochs}")
        print(f"Rank: {args.rank}")
        print(f"LoRA Alpha: {args.lora_alpha}")
        print(f"LoRA Dropout: {args.lora_dropout}")
        print(f'PEFT method: {args.peft}')
        print(f'Task: {args.task}')

        self.num_labels = get_num_labels(args)
        self.dir = Path(args.dir_name)
        self.results = self.args.copy()
        self.training_params = dict(
            eval_strategy="epoch",  # Enable evaluation every epoch
            logging_strategy="epoch",  # Enable logging
            save_strategy="epoch",
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
        )

    def get_args(self):
        return self.args.copy()

    def load_dataset(self):
        args = self.args
        teacher_dataset = load_glue_dataset(args.dataset_path, args.task, from_disk=bool(args.from_disk))
        print('Loaded dataset', teacher_dataset)
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
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name, use_fast=False)
        tokenized_teacher_dataset = teacher_dataset.map(tokenize_function(args, teacher_tokenizer),
                                                        batched=True, keep_in_memory=True)
        return tokenized_teacher_dataset

    def tokenize_student_dataset(self, teacher_dataset):
        args = self.args
        student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name, use_fast=False)
        tokenized_student_dataset = teacher_dataset.map(tokenize_function(args, student_tokenizer, with_indices=True),
                                                        with_indices=True, batched=True, keep_in_memory=True)
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

    def train_lora(self, model, train_dataset, eval_dataset, ckpt_dir):
        args = self.args
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
            learning_rate=args.student_learning_rate,
            **self.training_params,
        )
        print('training_args', training_args)
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

    def train_distill_lora(self, student_model, train_dataset, eval_dataset, teacher_soft_labels, ckpt_dir):
        args = self.args
        student_training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
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

    def train_fft(self, model, train_dataset, eval_dataset, ckpt_dir):
        args = self.args
        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
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

    @property
    def config_dir(self):
        args = self.args
        config_dir = f'task_{args.task}_{args.model_family}_{args.seed}/' + \
                     f'base_{args.train_batch_size}_{args.teacher_learning_rate}_{args.weight_decay}/' + \
                     f'peft_{args.peft}_{args.lora_alpha}_{args.lora_dropout}_{args.rank}'
        return config_dir

    @property
    def teacher_fft_dir(self):
        return self.dir / 'fft' / self.config_dir

    @property
    def teacher_lora_dir(self):
        return self.dir / 'lora' / self.config_dir

    @property
    def student_lora_dir(self):
        return self.dir / 'kd-lora' / self.config_dir

    @staticmethod
    def patch_results(results, args, train, variant):
        results.update(args=args, train=train, variant=variant)

    def run_teacher_fft(self):
        """
        Train, evaluate, and save teacher FFT model.
        Save teacher-soft-labels.
        :return:
        """
        args = self.args
        print('Begin Teacher FFT...')

        teacher_fft_dir = self.teacher_fft_dir
        teacher_fft_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = teacher_fft_dir / 'ckpt'
        metrics_file = teacher_fft_dir / 'metrics.json'
        if metrics_file.exists():
            return

        print('Preparing teacher FFT...')
        teacher_dataset = self.load_dataset()
        teacher_model = self.load_pretrained_model(args.teacher_model_name)
        tokenized_teacher_dataset = self.tokenize_teacher_dataset(teacher_dataset)
        teacher_train_dataset, teacher_eval_dataset = self.split_dataset(tokenized_teacher_dataset)
        print('Loaded dataset & model', '#train', len(teacher_train_dataset), '#eval', len(teacher_eval_dataset),
              '#param', get_trainable_param_count(teacher_model))

        teacher_trainer, train_metrics = self.train_fft(teacher_model, teacher_train_dataset, teacher_eval_dataset,
                                                        ckpt_dir)

        teacher_fft_results = self.evaluate_model(teacher_trainer, teacher_eval_dataset)
        self.patch_results(teacher_fft_results, args, train_metrics, 'fft')
        print(f"teacher fft results: {teacher_fft_results}")
        teacher_soft_labels = self.get_teacher_soft_labels(teacher_trainer, tokenized_teacher_dataset)

        if teacher_trainer.is_world_process_zero():
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(teacher_fft_results, f, indent=4, ensure_ascii=False)
            torch.save(teacher_soft_labels.cpu(), str(teacher_fft_dir / 'teacher_soft_labels.pth'))
            print('Saved teacher soft-labels.', teacher_soft_labels.shape)
            shutil.rmtree(ckpt_dir)
            teacher_dataset.cleanup_cache_files()

        if hasattr(teacher_trainer, "accelerator"):
            teacher_trainer.accelerator.wait_for_everyone()

        teacher_model.to('cpu')
        del teacher_model
        clear_gpu_memory()
        print('Teacher FFT is done.', args.task, args.teacher_model_name)

    def run_teacher_lora(self):
        teacher_lora_dir = self.teacher_lora_dir
        args = self.args
        ckpt_dir = teacher_lora_dir / 'ckpt'
        metrics_file = teacher_lora_dir / 'metrics.json'
        if metrics_file.exists():
            return
        # 3. Teacher LoRA
        print("Begin Teacher LoRA...")
        teacher_dataset = self.load_dataset()
        tokenized_teacher_dataset = self.tokenize_teacher_dataset(teacher_dataset)
        teacher_train_dataset, teacher_eval_dataset = self.split_dataset(tokenized_teacher_dataset)

        peft_config = get_peft_config(args, args.teacher_model_name, args.peft)
        print('target_modules', peft_config.target_modules)
        print('teacher', args.teacher_model_name)
        print('student', args.student_model_name)

        teacher_lora_model = self.load_pretrained_model_lora(args.teacher_model_name, lora_config=peft_config)
        print('Loaded dataset & model', '#train', len(teacher_train_dataset), '#eval', len(teacher_eval_dataset),
              '#param', get_trainable_param_count(teacher_lora_model))

        teacher_lora_trainer, train_metrics = self.train_lora(teacher_lora_model, teacher_train_dataset,
                                                              teacher_eval_dataset, ckpt_dir)

        teacher_lora_results = self.evaluate_model(teacher_lora_trainer, teacher_eval_dataset)
        self.patch_results(teacher_lora_results, args, train_metrics, 'lora')
        print(f"teacher lora results: {teacher_lora_results}")

        if teacher_lora_trainer.is_world_process_zero():
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(teacher_lora_results, f, indent=4, ensure_ascii=False)
            shutil.rmtree(ckpt_dir)
            teacher_dataset.cleanup_cache_files()

        if hasattr(teacher_lora_trainer, "accelerator"):
            teacher_lora_trainer.accelerator.wait_for_everyone()

        teacher_lora_model.to('cpu')
        del teacher_lora_model
        clear_gpu_memory()
        print('Teacher LoRA is done.', args.task, args.teacher_model_name)

    def run_student_lora(self):
        print("Begin Student Distill + LoRA...")
        student_lora_dir = self.student_lora_dir
        args = self.args
        ckpt_dir = student_lora_dir / 'ckpt'

        ori_peft = args.peft
        args.peft = 'lora'
        teacher_fft_dir = self.teacher_fft_dir
        args.peft = ori_peft
        metrics_file = student_lora_dir / 'metrics.json'
        if metrics_file.exists():
            return

        teacher_soft_labels = torch.load(str(teacher_fft_dir / 'teacher_soft_labels.pth'))
        print('Loaded teacher soft-labels.', args.taks, teacher_soft_labels.shape)

        teacher_dataset = self.load_dataset()
        peft_config = get_peft_config(args, args.student_model_name, args.peft)
        tokenized_student_dataset = self.tokenize_student_dataset(teacher_dataset)
        student_model = self.load_pretrained_model_lora(args.student_model_name, lora_config=peft_config)
        student_train_dataset, student_eval_dataset = self.split_dataset(tokenized_student_dataset)
        print('Loaded dataset & model', '#train', len(student_train_dataset), '#eval', len(student_eval_dataset),
              '#param', get_trainable_param_count(student_model))

        student_trainer, train_metrics = self.train_distill_lora(student_model, student_train_dataset,
                                                                 student_eval_dataset,
                                                                 teacher_soft_labels, ckpt_dir)

        student_lora_results = self.evaluate_model(student_trainer, student_eval_dataset)
        self.patch_results(student_lora_results, args, train_metrics, 'kd-lora')
        print(f"student lora results: {student_lora_results}")

        if student_trainer.is_world_process_zero():
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(student_lora_results, f, indent=4, ensure_ascii=False)
            shutil.rmtree(ckpt_dir)
            teacher_dataset.cleanup_cache_files()

        if hasattr(student_trainer, "accelerator"):
            student_trainer.accelerator.wait_for_everyone()

        student_model.to('cpu')
        del student_model
        clear_gpu_memory()
        print('Student LoRA is done.', args.task, args.student_model_name)


def main_teacher_fft(args):
    for seed in [42, 123, 2024]:
        for task in GLUE_TASKS:
            for model_family in MODEL_FAMILY.keys():
                set_seed(seed)
                config = args.__dict__.copy()
                config['model_family'] = model_family
                config['task'] = task
                config['seed'] = seed
                add_model_name_to_config(model_family, config)
                pipe = BertDistillPipeline(**config)
                try:
                    pipe.run_teacher_fft()
                except Exception as e:
                    print(e)
                    # raise e


def main_lora(args, is_student: bool):
    for seed in [42, 123, 2024]:
        for task in GLUE_TASKS:
            for model_family in MODEL_FAMILY.keys():
                for peft_method in PEFT_FAMILY:
                    set_seed(seed)
                    config = args.__dict__.copy()
                    config['model_family'] = model_family
                    config['task'] = task
                    config['peft'] = peft_method
                    config['seed'] = seed
                    add_model_name_to_config(model_family, config)
                    pipe = BertDistillPipeline(**config)
                    try:
                        if is_student:
                            pipe.run_student_lora()
                        else:
                            pipe.run_teacher_lora()
                    except FileNotFoundError as e:
                        print(e)
                    except Exception as e:
                        print(e)
                        # raise e


def main_mrlora(args):
    config = args.__dict__.copy()
    config['peft'] = 'mrlora'
    model_family = 'bert'
    add_model_name_to_config(model_family, config)
    pipe = BertDistillPipeline(**config)
    pipe.run_teacher_lora()
    # pipe.run_student_lora()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation with LoRA-enhanced Student Model")

    # Model arguments
    parser.add_argument("--teacher_model_name", type=str, default="./models/bert-base-uncased",
                        help="Name of the teacher model")
    parser.add_argument("--student_model_name", type=str, default="./models/distilbert-base-uncased",
                        help="Name of the student model")

    # Dataset and training parameters
    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    parser.add_argument("--dir_name", type=str, default="./results", help="Directory name for saving models")

    # LoRA parameters
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout rate for LoRA layers")
    parser.add_argument('--lora_ranks', type=int, default=(8, 4, 2, 1), nargs='+', help="MrLora ranks")

    # Learning rates for teacher and student
    parser.add_argument("--teacher_learning_rate", type=float, default=5e-5, help="Learning rate for the teacher model")
    parser.add_argument("--student_learning_rate", type=float, default=5e-4, help="Learning rate for the student model")
    parser.add_argument('--task', type=str, default="wnli", choices=tuple(GLUE_TASKS), help="Name of the task")
    parser.add_argument('--peft', type=str, default="lora", choices=tuple(PEFT_FAMILY), help="PEFT method name")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--type', '-t', type=int, choices=(0, 1, 2, 3),
                        help='0 => fft, 1 => student-lora, 2 => teacher-lora')
    parser.add_argument('--from_disk', type=int, default=1, help="If 1, use load_from_disk()")

    args_cmd = parser.parse_args()
    if args_cmd.type == 0:
        main_teacher_fft(args_cmd)
    elif args_cmd.type == 1:
        main_lora(args_cmd, is_student=True)
    elif args_cmd.type == 2:
        main_lora(args_cmd, is_student=False)
    elif args_cmd.type == 3:
        main_mrlora(args_cmd)
    else:
        raise ValueError(f"Unknown command {args_cmd.type}")
