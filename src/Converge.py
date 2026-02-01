import argparse
import json
import shutil

from addict import Addict
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model
from utils import *

# Suppress tokenizer warning about overflowing tokens not returned for 'longest_first' truncation strategy
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Hyperparameter search space (rank)
# KD-LoRA paper uses rank 8,16,32,64 with alpha = rank, but we fix alpha = 16
RANK_VALUES = [8]
# ALPHA_VALUES kept for reference (alpha is fixed at 16)
seed_list = [42]
EVAL_STEPS = 10
MAX_EPOCHS = 20
GLUE_TASKS = ['cola']
PEFT_FAMILY = ['mrlora-rs']

# TODO: merge this class
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
            eval_strategy="steps",  # Enable evaluation every epoch
            logging_strategy="steps",  # Enable logging
            save_strategy="steps",
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=MAX_EPOCHS,
            eval_steps=EVAL_STEPS,
            save_steps=EVAL_STEPS,
            logging_steps=EVAL_STEPS,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            save_total_limit=2,  # 只保留最近的两个模型，省空间
            warmup_ratio=0.1,
        )

    def get_args(self):
        return self.args.copy()

    def load_dataset(self):
        args = self.args
        teacher_dataset = load_glue_dataset(args.dataset_path, args.task)
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
        """Tokenize teacher dataset (teacher_dataset parameter ignored, cached dataset used)."""
        args = self.args
        tokenized_teacher_dataset = get_tokenized_dataset(
            task=args.task,
            tokenizer_name=args.teacher_model_name,
            with_indices=False,
            dataset_path=args.dataset_path,
        )
        return tokenized_teacher_dataset

    def tokenize_student_dataset(self, teacher_dataset):
        """Tokenize student dataset (teacher_dataset parameter ignored, cached dataset used)."""
        args = self.args
        tokenized_student_dataset = get_tokenized_dataset(
            task=args.task,
            tokenizer_name=args.student_model_name,
            with_indices=True,
            dataset_path=args.dataset_path,
        )
        return tokenized_student_dataset

    def split_dataset(self, tokenized_datasets):
        """
        Prepare train/validation split
        :param tokenized_datasets:
        :return:
        """
        args = self.args
        train_dataset = tokenized_datasets["train"].shuffle(args.seed)
        args.train_size = len(train_dataset)

        if args.task == "mnli":
            # MNLI requires evaluation on both matched and mismatched datasets
            eval_matched_dataset = tokenized_datasets["validation_matched"]
            eval_mismatched_dataset = tokenized_datasets["validation_mismatched"]
            return train_dataset, (eval_matched_dataset, eval_mismatched_dataset)
        else:
            # Standard train-validation split for other datasets
            eval_dataset = tokenized_datasets["validation"]
            return train_dataset, eval_dataset

    def train_model(self, model, train_dataset, eval_dataset, ckpt_dir, teacher_soft_labels=None):
        args = self.args
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
            learning_rate=args.student_learning_rate,
            **self.training_params,
        )
        if args.task == "mnli":
            eval_dataset = eval_dataset[0]
        callback = MemoryTrackingCallback()
        trainer = (Trainer if teacher_soft_labels is None else DistillationTrainer)(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics(args),
            callbacks=[callback, EarlyStoppingCallback(early_stopping_patience=5)],
        )
        if teacher_soft_labels is not None:
            trainer.teacher_soft_labels = teacher_soft_labels

        train_output = trainer.train()
        print("Finished Training")
        return trainer, get_train_metrics(train_output, model, callback)

    def get_teacher_soft_labels(self, teacher_trainer, tokenized_teacher_dataset):
        teacher_logits = teacher_trainer.predict(tokenized_teacher_dataset["train"]).predictions
        teacher_soft_labels = torch.tensor(teacher_logits)
        return teacher_soft_labels

    def evaluate_model(self, trainer, eval_dataset):
        args = self.args
        # set_seed(args.seed, deterministic=True) # Some kernel not supported.
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
            eval_results.update(eval_results_matched)
            del eval_results['eval_accuracy']
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

        teacher_trainer, train_metrics = self.train_model(teacher_model, teacher_train_dataset, teacher_eval_dataset,
                                                          ckpt_dir)

        teacher_fft_results = self.evaluate_model(teacher_trainer, teacher_eval_dataset)
        self.patch_results(teacher_fft_results, args, train_metrics, 'fft')
        # print(f"teacher fft results: {teacher_fft_results}")
        teacher_soft_labels = self.get_teacher_soft_labels(teacher_trainer, tokenized_teacher_dataset)

        if teacher_trainer.is_world_process_zero():
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(teacher_fft_results, f, indent=4, ensure_ascii=False)
            soft_labels_path = teacher_fft_dir / 'teacher_soft_labels.pth'
            print(f"Saving soft labels of shape {teacher_soft_labels.shape} to {soft_labels_path}")
            torch.save(teacher_soft_labels.cpu(), str(soft_labels_path))
            print('Saved teacher soft-labels.', teacher_soft_labels.shape)
            assert soft_labels_path.exists(), "Soft labels file not created!"
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
        teacher_lora_model = self.load_pretrained_model_lora(args.teacher_model_name, lora_config=peft_config)
        print('Loaded dataset & model', '#train', len(teacher_train_dataset), '#eval', len(teacher_eval_dataset),
              '#param', get_trainable_param_count(teacher_lora_model))

        teacher_lora_trainer, train_metrics = self.train_model(teacher_lora_model, teacher_train_dataset,
                                                               teacher_eval_dataset, ckpt_dir)

        teacher_lora_results = self.evaluate_model(teacher_lora_trainer, teacher_eval_dataset)
        self.patch_results(teacher_lora_results, args, train_metrics, 'lora')
        # print(f"teacher lora results: {teacher_lora_results}")

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

    def load_teacher_labels(self):
        args = self.args
        # Teacher labels are not related to lora settings.
        teacher_fft_dir = self.teacher_fft_dir.parent  # last level is lora config
        teacher_soft_labels_path = next(teacher_fft_dir.rglob('teacher_soft_labels.pth'))
        teacher_soft_labels = torch.load(teacher_soft_labels_path, weights_only=False)
        print('Loaded teacher soft-labels.', args.task, teacher_soft_labels_path, teacher_soft_labels.shape)
        return teacher_soft_labels

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

        teacher_soft_labels = self.load_teacher_labels()

        teacher_dataset = self.load_dataset()
        peft_config = get_peft_config(args, args.student_model_name, args.peft)

        tokenized_student_dataset = self.tokenize_student_dataset(teacher_dataset)
        student_model = self.load_pretrained_model_lora(args.student_model_name, lora_config=peft_config)
        student_train_dataset, student_eval_dataset = self.split_dataset(tokenized_student_dataset)
        print('Loaded dataset & model', '#train', len(student_train_dataset), '#eval', len(student_eval_dataset),
              '#param', get_trainable_param_count(student_model))

        student_trainer, train_metrics = self.train_model(student_model, student_train_dataset,
                                                          student_eval_dataset,
                                                          ckpt_dir, teacher_soft_labels)

        student_lora_results = self.evaluate_model(student_trainer, student_eval_dataset)
        self.patch_results(student_lora_results, args, train_metrics, 'kd-lora')
        # print(f"student lora results: {student_lora_results}")

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
    for seed in seed_list:
        for task in GLUE_TASKS:
            for model_family in MODEL_FAMILY.keys():
                set_seed(seed, deterministic=False)
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
                    raise e
    print('All finish')


def main_lora(args, is_student: bool):
    for rank in RANK_VALUES:
        for seed in seed_list:
            for task in GLUE_TASKS:
                for model_family in MODEL_FAMILY.keys():
                    for peft_method in PEFT_FAMILY:
                        # Set alpha = 16 (fixed) as per our experimental setup
                        set_seed(seed, deterministic=False)
                        config = args.__dict__.copy()
                        config['model_family'] = model_family
                        config['task'] = task
                        config['peft'] = peft_method
                        config['seed'] = seed
                        config['rank'] = rank
                        config['lora_alpha'] = 2 * rank

                        add_model_name_to_config(model_family, config)
                        pipe = BertDistillPipeline(**config)
                        try:
                            if is_student:
                                pipe.run_student_lora()
                            else:
                                pipe.run_teacher_lora()
                        except Exception as e:
                            print(e)
                            raise e
    print('All finish')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation with LoRA-enhanced Student Model")

    # Dataset and training parameters
    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    parser.add_argument("--dir_name", type=str, default="./results", help="Directory name for saving models")

    # LoRA parameters
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout rate for LoRA layers")
    parser.add_argument('--use_rslora', action='store_true',
                        help='Use rank-stabilized scaling for MrLoRA (lora_alpha/sqrt(r) instead of lora_alpha/max(ranks))')

    # Learning rates for teacher and student
    parser.add_argument("--teacher_learning_rate", type=float, default=2e-5, help="Learning rate for the teacher model")
    # TODO: lr.
    parser.add_argument("--student_learning_rate", type=float, default=1e-4, help="Learning rate for the student model")

    args_cmd = parser.parse_args()
    main_teacher_fft(args_cmd)
    main_lora(args_cmd, is_student=True)
    main_lora(args_cmd, is_student=False)
