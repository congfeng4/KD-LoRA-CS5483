import argparse
import json
import shutil
import logging
import sys
import numpy as np
import collections

from addict import Addict
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from peft import get_peft_model
from utils import *

# Suppress tokenizer warning about overflowing tokens not returned for 'longest_first' truncation strategy
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Hyperparameter search space (rank)
# KD-LoRA paper uses rank 8,16,32,64 with alpha = rank, but we fix alpha = 16
RANK_VALUES = [8, 16, 32, 64]
# ALPHA_VALUES kept for reference (alpha is fixed at 16)
seed_list = [42, 123, 2024]

# QA tasks
QA_TASKS = ["squad", "squad_v2"]
# Bench type hardcoded to QA
BENCH = "QA"

# Try to import evaluate for SQuAD metrics
try:
    import evaluate
    squad_metric = evaluate.load("squad")
except ImportError:
    squad_metric = None
    print("Warning: evaluate library not installed. SQuAD metrics will not be computed.")
    print("Install with: pip install evaluate")

# Caches for sharing datasets and tokenizers across runs in the same process
_RAW_DATASET_CACHE = {}
_TOKENIZER_CACHE = {}
_TOKENIZED_DATASET_CACHE = {}

def load_qa_dataset(dataset_path, task, from_disk=True):
    """Load SQuAD dataset from disk or Hugging Face Hub."""
    if from_disk:
        from datasets import load_from_disk
        return load_from_disk(Path(dataset_path) / task)
    else:
        from datasets import load_dataset
        return load_dataset(task, cache_dir=dataset_path)

def get_raw_dataset(dataset_path, task, from_disk=True):
    """Load raw QA dataset with caching."""
    key = (dataset_path, task, from_disk)
    if key not in _RAW_DATASET_CACHE:
        print(f"[CACHE MISS] Loading raw dataset: task={task}, from_disk={from_disk}")
        _RAW_DATASET_CACHE[key] = load_qa_dataset(dataset_path, task, from_disk)
    else:
        print(f"[CACHE HIT] Using cached raw dataset: task={task}, from_disk={from_disk}")
    return _RAW_DATASET_CACHE[key]

def get_tokenizer(model_name):
    """Get tokenizer with caching."""
    if model_name not in _TOKENIZER_CACHE:
        print(f"[CACHE MISS] Loading tokenizer: {model_name}")
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        print(f"[CACHE HIT] Using cached tokenizer: {model_name}")
    return _TOKENIZER_CACHE[model_name]

def get_tokenized_dataset_qa(task, tokenizer_name, with_indices=False, dataset_path=None, from_disk=True, max_length=384, doc_stride=128):
    """Get tokenized QA dataset with caching.
    
    Tokenized dataset is cached per (task, tokenizer_name, with_indices, max_length, doc_stride).
    dataset_path is required for first-time loading.
    """
    key = (task, tokenizer_name, with_indices, max_length, doc_stride)
    if key not in _TOKENIZED_DATASET_CACHE:
        print(f"[CACHE MISS] Tokenizing dataset: task={task}, tokenizer={tokenizer_name}, with_indices={with_indices}")
        if dataset_path is None:
            raise ValueError("dataset_path must be provided for first-time tokenization")
        raw_dataset = get_raw_dataset(dataset_path, task, from_disk)
        tokenizer = get_tokenizer(tokenizer_name)
        
        # Tokenization function
        def tokenize_function(examples):
            # Standard SQuAD preprocessing
            tokenized = tokenizer(
                examples["question"],
                examples["context"],
                truncation="only_second",
                max_length=max_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )
            # Since we return overflowing tokens, we need to map start/end positions
            # We'll handle this in a separate step
            return tokenized
        
        # Tokenize (keep original columns for mapping)
        # Determine columns to keep
        columns_to_keep = ["answers", "id", "context", "question", "title"]
        columns_to_remove = [col for col in raw_dataset["train"].column_names if col not in columns_to_keep]
        tokenized = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
            keep_in_memory=True
        )
        
        # Map start/end positions
        def add_token_positions(examples):
            # Get start/end character positions from answers
            start_positions = []
            end_positions = []
            
            for i, (answers, offset_mapping) in enumerate(zip(examples["answers"], examples["offset_mapping"])):
                # For squad_v2, answers may be empty (impossible question)
                if not answers["text"]:
                    # No answer
                    start_positions.append(0)
                    end_positions.append(0)
                    continue
                    
                # Start character position of answer
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # Find token indices
                sequence_ids = examples.sequence_ids(i)
                # Find the start and end of the context
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                token_end_index = len(sequence_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                
                # If the answer is not fully inside the context, label as impossible
                if offset_mapping[token_start_index][0] > start_char or offset_mapping[token_end_index][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise find start and end token indices
                    while token_start_index < len(offset_mapping) and offset_mapping[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)
                    while offset_mapping[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)
            
            examples["start_positions"] = start_positions
            examples["end_positions"] = end_positions
            return examples
        
        tokenized = tokenized.map(
            add_token_positions,
            batched=True,
            keep_in_memory=True
        )
        
        # Remove unnecessary columns (only title)
        if "title" in tokenized.column_names:
            tokenized = tokenized.remove_columns(["title"])
        
        if with_indices:
            # Add idx column for distillation alignment
            tokenized = tokenized.map(
                lambda example, idx: {"idx": idx},
                with_indices=True,
                keep_in_memory=True
            )
        
        _TOKENIZED_DATASET_CACHE[key] = tokenized
    else:
        print(f"[CACHE HIT] Using cached tokenized dataset: task={task}, tokenizer={tokenizer_name}, with_indices={with_indices}")
    return _TOKENIZED_DATASET_CACHE[key]

def generate_mrlora_ranks(highest_rank):
    """Generate MrLoRA ranks list from highest_rank down to 1 by halving."""
    ranks = []
    r = highest_rank
    while r >= 1:
        ranks.append(r)
        r = r // 2
    # Ensure at least two ranks
    if len(ranks) == 1:
        ranks.append(ranks[0] // 2)
    return ranks


def postprocess_qa_predictions(start_logits, end_logits, features, examples, version_2_with_negative=False, n_best_size=20, max_answer_length=30):
    """Postprocess predictions for SQuAD evaluation."""
    # Ensure we have numpy arrays
    start_logits = np.array(start_logits)
    end_logits = np.array(end_logits)
    
    # Build map from example index to its features (multiple features per example due to chunking)
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["id"]]].append(i)
    
    # Predictions dict
    predictions = collections.OrderedDict()
    
    # Loop over examples
    for example_index, example in enumerate(examples):
        example_id = example["id"]
        context = example["context"]
        answers = example["answers"]
        
        # Get features for this example
        feature_indices = features_per_example[example_index]
        
        min_null_score = None
        valid_answers = []
        
        # Loop over features
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            # Update minimum null prediction score (for impossible)
            feature_null_score = start_logit[0] + end_logit[0]
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score
            
            # Get top n_best predictions
            start_indexes = np.argsort(start_logit)[-n_best_size:]
            end_indexes = np.argsort(end_logit)[-n_best_size:]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip if not in the same chunk
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    # Skip if end < start
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append({
                        "score": start_logit[start_index] + end_logit[end_index],
                        "text": context[start_char:end_char]
                    })
        
        if version_2_with_negative:
            # For squad_v2, add null answer
            if min_null_score is not None:
                valid_answers.append({"score": min_null_score, "text": ""})
        
        # Select best answer
        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x["score"])
            predictions[example_id] = best_answer["text"]
        else:
            predictions[example_id] = ""
    
    return predictions

def make_compute_metrics_qa(eval_dataset, examples, version_2_with_negative=False):
    """Create a compute_metrics function that uses squad metric."""
    if squad_metric is None:
        def dummy_metrics(p):
            return {}
        return dummy_metrics
    
    # Convert eval_dataset (features) to list of dict for postprocessing
    features = [{k: eval_dataset[i][k] for k in eval_dataset.column_names} for i in range(len(eval_dataset))]
    
    def compute_qa_metrics(p):
        # p is EvalPrediction with predictions and label_ids
        predictions, labels = p
        if isinstance(predictions, tuple):
            start_logits, end_logits = predictions
        else:
            # predictions may be stacked? assume tuple
            start_logits = predictions[0]
            end_logits = predictions[1]
        
        # Postprocess predictions
        pred_dict = postprocess_qa_predictions(
            start_logits, end_logits, features, examples,
            version_2_with_negative=version_2_with_negative
        )
        
        # Format references
        references = {example["id"]: example["answers"] for example in examples}
        
        # Compute metrics
        results = squad_metric.compute(predictions=list(pred_dict.values()), references=list(references.values()))
        return results
    
    return compute_qa_metrics

class QADistillPipeline:
    """
    QA Distillation Pipeline for SQuAD.
    """

    def __init__(self, **kwargs):
        args = Addict(kwargs)
        # Add bench attribute
        args.bench = BENCH
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
        print(f'Max length: {args.max_length}')
        print(f'Doc stride: {args.doc_stride}')
        print(f'Distill loss: {args.distill_loss}')

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
        """Load SQuAD dataset from disk or Hugging Face Hub."""
        args = self.args
        dataset = get_raw_dataset(args.dataset_path, args.task, from_disk=bool(args.from_disk))
        print('Loaded dataset', dataset)
        return dataset

    def load_pretrained_model(self, model_name):
        """Load pretrained model for question answering."""
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        print('Loaded pretrained model', model_name)
        return model

    def load_pretrained_model_lora(self, model_name, lora_config=None):
        """Load pretrained model with LoRA for QA."""
        student_model = self.load_pretrained_model(model_name)
        args = self.args
        if lora_config is None:
            # Use get_peft_config from utils (will set task_type based on bench)
            lora_config = get_peft_config(args, model_name, args.peft)
        # Apply LoRA configuration to the student model
        student_model = get_peft_model(student_model, lora_config)
        print("Loaded pretrained model with LoRA", model_name)
        return student_model

    def tokenize_teacher_dataset(self, teacher_dataset):
        """Tokenize teacher dataset with standard SQuAD preprocessing."""
        args = self.args
        tokenized_teacher_dataset = get_tokenized_dataset_qa(
            task=args.task,
            tokenizer_name=args.teacher_model_name,
            with_indices=False,
            dataset_path=args.dataset_path,
            from_disk=bool(args.from_disk),
            max_length=args.max_length,
            doc_stride=args.doc_stride
        )
        return tokenized_teacher_dataset

    def tokenize_student_dataset(self, teacher_dataset):
        """Tokenize student dataset with indices for distillation alignment."""
        args = self.args
        tokenized_student_dataset = get_tokenized_dataset_qa(
            task=args.task,
            tokenizer_name=args.student_model_name,
            with_indices=True,
            dataset_path=args.dataset_path,
            from_disk=bool(args.from_disk),
            max_length=args.max_length,
            doc_stride=args.doc_stride
        )
        return tokenized_student_dataset

    def split_dataset(self, tokenized_datasets):
        """
        Prepare train/validation split.
        For SQuAD, we have 'train' and 'validation' splits.
        Returns (train_dataset, eval_dataset, raw_eval_examples).
        raw_eval_examples: list of dicts for validation set (for metrics).
        """
        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)
        # Load raw validation examples for metric computation
        raw_dataset = get_raw_dataset(self.args.dataset_path, self.args.task, from_disk=bool(self.args.from_disk))
        raw_eval_examples = raw_dataset["validation"]
        # Convert to list of dicts
        raw_eval_examples = [{k: raw_eval_examples[i][k] for k in raw_eval_examples.column_names} for i in range(len(raw_eval_examples))]
        # Store for compute_metrics
        self.raw_eval_examples = raw_eval_examples
        return train_dataset, eval_dataset, raw_eval_examples

    def get_compute_metrics(self, eval_dataset):
        """Return a compute_metrics function for QA using raw_eval_examples."""
        if not hasattr(self, 'raw_eval_examples'):
            raise ValueError("raw_eval_examples not set. Call split_dataset first.")
        version_2_with_negative = (self.args.task == "squad_v2")
        return make_compute_metrics_qa(eval_dataset, self.raw_eval_examples, version_2_with_negative)

    def train_lora(self, model, train_dataset, eval_dataset, ckpt_dir):
        args = self.args
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
            learning_rate=args.student_learning_rate,
            **self.training_params,
        )
        callback = MemoryTrackingCallback()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.get_compute_metrics(eval_dataset),
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
        callback = MemoryTrackingCallback()
        student_trainer = QADistillationTrainer(
            model=student_model,
            args=student_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.get_compute_metrics(eval_dataset),
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
        callback = MemoryTrackingCallback()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.get_compute_metrics(eval_dataset),
            callbacks=[callback],
        )
        # Train the model
        train_output = trainer.train()
        print("Finished Training")
        return trainer, get_train_metrics(train_output, model, callback)

    def get_teacher_soft_labels(self, teacher_trainer, tokenized_teacher_dataset):
        """Get teacher logits (start and end) for distillation."""
        predictions = teacher_trainer.predict(tokenized_teacher_dataset["train"])
        # predictions.predictions is tuple of (start_logits, end_logits)
        start_logits = torch.tensor(predictions.predictions[0])
        end_logits = torch.tensor(predictions.predictions[1])
        return (start_logits, end_logits)

    def evaluate_model(self, trainer, eval_dataset):
        """Evaluate model using SQuAD metrics."""
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Evaluation results: {eval_results}")
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



    # The three main pipelines: teacher FFT, teacher LoRA, student LoRA
    def run_teacher_fft(self):
        """
        Train, evaluate, and save teacher FFT model.
        Save teacher soft labels (start and end logits).
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
        teacher_train_dataset, teacher_eval_dataset, raw_eval_examples = self.split_dataset(tokenized_teacher_dataset)
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
            soft_labels_path = teacher_fft_dir / 'teacher_soft_labels.pth'
            print(f"Saving soft labels of shapes {teacher_soft_labels[0].shape}, {teacher_soft_labels[1].shape} to {soft_labels_path}")
            torch.save(teacher_soft_labels, str(soft_labels_path))
            print('Saved teacher soft-labels.')
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
        teacher_train_dataset, teacher_eval_dataset, _ = self.split_dataset(tokenized_teacher_dataset)

        peft_config = get_peft_config(args, args.teacher_model_name, args.peft)
        # Set total training steps for AdaLoRA
        if args.peft == 'adalora':
            steps_per_epoch = (len(teacher_train_dataset) + args.train_batch_size - 1) // args.train_batch_size
            total_step = steps_per_epoch * args.num_train_epochs
            peft_config.total_step = total_step
            # Adjust tinit, tfinal, deltaT to be within total_step bounds
            if total_step < peft_config.tinit + peft_config.tfinal:
                # Scale proportionally
                scale = total_step / (peft_config.tinit + peft_config.tfinal)
                peft_config.tinit = max(1, int(peft_config.tinit * scale))
                peft_config.tfinal = max(1, int(peft_config.tfinal * scale))
                peft_config.deltaT = max(1, int(peft_config.deltaT * scale))
                print(f"Scaled AdaLoRA pruning parameters: tinit={peft_config.tinit}, tfinal={peft_config.tfinal}, deltaT={peft_config.deltaT}")
            # Ensure tinit < total_step - tfinal (pruning interval positive)
            if peft_config.tinit >= total_step - peft_config.tfinal:
                # Reduce tfinal so that there is at least one step for pruning
                peft_config.tfinal = total_step - peft_config.tinit - 1
                if peft_config.tfinal < 1:
                    peft_config.tfinal = 1
                    peft_config.tinit = total_step - 2
                print(f"Adjusted tfinal={peft_config.tfinal}, tinit={peft_config.tinit}")
            print(f"AdaLoRA total_step set to {total_step} (steps per epoch: {steps_per_epoch})")
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

    def load_teacher_labels(self):
        args = self.args
        # Teacher labels are not related to lora settings.
        teacher_fft_dir = self.teacher_fft_dir.parent # last level is lora config
        teacher_soft_labels_path = next(teacher_fft_dir.rglob('teacher_soft_labels.pth'))
        teacher_soft_labels = torch.load(teacher_soft_labels_path, weights_only=False)
        print('Loaded teacher soft-labels.', args.task, teacher_soft_labels_path, teacher_soft_labels[0].shape, teacher_soft_labels[1].shape)
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
        # Set total training steps for AdaLoRA
        if args.peft == 'adalora':
            train_size = len(teacher_dataset['train'])
            steps_per_epoch = (train_size + args.train_batch_size - 1) // args.train_batch_size
            total_step = steps_per_epoch * args.num_train_epochs
            peft_config.total_step = total_step
            # Adjust tinit, tfinal, deltaT to be within total_step bounds
            if total_step < peft_config.tinit + peft_config.tfinal:
                # Scale proportionally
                scale = total_step / (peft_config.tinit + peft_config.tfinal)
                peft_config.tinit = max(1, int(peft_config.tinit * scale))
                peft_config.tfinal = max(1, int(peft_config.tfinal * scale))
                peft_config.deltaT = max(1, int(peft_config.deltaT * scale))
                print(f"Scaled AdaLoRA pruning parameters: tinit={peft_config.tinit}, tfinal={peft_config.tfinal}, deltaT={peft_config.deltaT}")
            # Ensure tinit < total_step - tfinal (pruning interval positive)
            if peft_config.tinit >= total_step - peft_config.tfinal:
                # Reduce tfinal so that there is at least one step for pruning
                peft_config.tfinal = total_step - peft_config.tinit - 1
                if peft_config.tfinal < 1:
                    peft_config.tfinal = 1
                    peft_config.tinit = total_step - 2
                print(f"Adjusted tfinal={peft_config.tfinal}, tinit={peft_config.tinit}")
            print(f"AdaLoRA total_step set to {total_step} (steps per epoch: {steps_per_epoch}, train size: {train_size})")
        tokenized_student_dataset = self.tokenize_student_dataset(teacher_dataset)
        student_model = self.load_pretrained_model_lora(args.student_model_name, lora_config=peft_config)
        student_train_dataset, student_eval_dataset, _ = self.split_dataset(tokenized_student_dataset)
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


# Custom trainer for QA distillation
class QADistillationTrainer(Trainer):
    teacher_soft_labels: tuple  # (start_logits, end_logits)

    def compute_loss(self, model, inputs, return_outputs=False, **kwds):
        # Extract idx for aligning teacher logits
        idx = inputs.pop("idx").long().cpu()
        # Extract start_positions and end_positions (hard labels) if present
        start_positions = inputs.pop("start_positions", None)
        end_positions = inputs.pop("end_positions", None)
        
        # Forward pass
        outputs = model(**inputs)
        student_start_logits = outputs.start_logits
        student_end_logits = outputs.end_logits
        
        # Get teacher logits
        teacher_start_logits, teacher_end_logits = self.teacher_soft_labels
        teacher_start_logits = teacher_start_logits[idx].to(student_start_logits.device).to(student_start_logits.dtype)
        teacher_end_logits = teacher_end_logits[idx].to(student_end_logits.device).to(student_end_logits.dtype)
        
        # Compute MSE loss between student and teacher logits
        loss_start = F.mse_loss(student_start_logits, teacher_start_logits)
        loss_end = F.mse_loss(student_end_logits, teacher_end_logits)
        loss = (loss_start + loss_end) / 2.0
        
        if return_outputs:
            return loss, outputs
        return loss


# Main loops similar to BERT_Distill_LoRA.py
def main_teacher_fft(args):
    for seed in seed_list:
        for task in QA_TASKS:
            for model_family in MODEL_FAMILY.keys():
                set_seed(seed)
                config = args.__dict__.copy()
                config['model_family'] = model_family
                config['task'] = task
                config['seed'] = seed
                add_model_name_to_config(model_family, config)
                pipe = QADistillPipeline(**config)
                try:
                    pipe.run_teacher_fft()
                except Exception as e:
                    print(e)
                    raise e
    print('All finish')


def main_lora(args, is_student: bool):
    for rank in RANK_VALUES:
        for seed in seed_list:
            for task in QA_TASKS:
                for model_family in MODEL_FAMILY.keys():
                    for peft_method in PEFT_FAMILY:
                        # Set alpha = 16 (fixed) as per our experimental setup
                        set_seed(seed)
                        config = args.__dict__.copy()
                        config['model_family'] = model_family
                        config['task'] = task
                        config['peft'] = peft_method
                        config['seed'] = seed
                        config['rank'] = rank
                        config['lora_alpha'] = 16
                        
                        # For MrLoRA, generate ranks list from highest rank down to 1
                        # unless user provided custom lora_ranks (non-default)
                        if 'mrlora' in peft_method:
                            config['lora_ranks'] = generate_mrlora_ranks(rank)
                        
                        add_model_name_to_config(model_family, config)
                        pipe = QADistillPipeline(**config)
                        try:
                            if is_student:
                                pipe.run_student_lora()
                            else:
                                pipe.run_teacher_lora()
                        except Exception as e:
                            print(e)
                            raise e
    print('All finish')


def main_mrlora(args):
    config = args.__dict__.copy()
    config['peft'] = 'mrlora'
    model_family = 'bert'
    add_model_name_to_config(model_family, config)
    pipe = QADistillPipeline(**config)
    pipe.run_teacher_lora()
    # pipe.run_student_lora()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation with LoRA-enhanced Student Model for QA")

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

    parser.add_argument("--dir_name", type=str, default="./results_qa", help="Directory name for saving models")

    # LoRA parameters
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout rate for LoRA layers")
    parser.add_argument('--use_rslora', action='store_true', help='Use rank-stabilized scaling for MrLoRA (lora_alpha/sqrt(r) instead of lora_alpha/max(ranks))')

    # Learning rates for teacher and student
    parser.add_argument("--teacher_learning_rate", type=float, default=5e-5, help="Learning rate for the teacher model")
    parser.add_argument("--student_learning_rate", type=float, default=5e-4, help="Learning rate for the student model")
    parser.add_argument('--task', type=str, default="squad", choices=tuple(QA_TASKS), help="Name of the task")
    parser.add_argument('--peft', type=str, default="lora", choices=tuple(PEFT_FAMILY), help="PEFT method name")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--type', '-t', type=int, choices=(0, 1, 2, 3),
                        help='0 => fft, 1 => student-lora, 2 => teacher-lora')
    parser.add_argument('--from_disk', type=int, default=1, help="If 1, use load_from_disk()")
    
    # QA-specific arguments
    parser.add_argument('--max_length', type=int, default=384, help="Maximum sequence length")
    parser.add_argument('--doc_stride', type=int, default=128, help="Document stride for sliding window")
    parser.add_argument('--distill_loss', type=str, default='mse', choices=('mse',),
                        help='Distillation loss type (currently only MSE)')

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