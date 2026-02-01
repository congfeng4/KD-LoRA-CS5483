#!/usr/bin/env python3
"""
Run AdaLoRA experiments (teacher and student distillation) with fixed configuration.
Based on BERT_Distill_LoRA.py but specialized for AdaLoRA only.
"""

import argparse
import json
import shutil
from pathlib import Path

from addict import Addict
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from peft import get_peft_model
from utils import *
from BERT_Distill_LoRA import BertDistillPipeline

# Suppress tokenizer warning about overflowing tokens not returned for 'longest_first' truncation strategy
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Hyperparameter search space (rank) - same as other experiments
RANK_VALUES = [8, 16, 32, 64]
seed_list = [42, 123, 2024]


def main_adalora(args, run_teacher=True, run_student=True, dry_run=False):
    """
    Run AdaLoRA experiments for all combinations of rank, seed, task, and model family.
    
    Args:
        args: Command-line arguments (same as BERT_Distill_LoRA.py)
        run_teacher: If True, run teacher AdaLoRA fine-tuning
        run_student: If True, run student AdaLoRA distillation
    """
    for rank in RANK_VALUES:
        for seed in seed_list:
            for task in GLUE_TASKS:
                for model_family in MODEL_FAMILY.keys():
                    # Set random seed for reproducibility
                    set_seed(seed)
                    
                    # Build configuration
                    config = args.__dict__.copy()
                    config.update({
                        'model_family': model_family,
                        'task': task,
                        'peft': 'adalora',          # Force AdaLoRA
                        'seed': seed,
                        'rank': rank,
                        'lora_alpha': 16,           # Fixed as per experimental setup
                        'lora_dropout': args.lora_dropout,
                    })
                    
                    # Add teacher/student model names based on family
                    add_model_name_to_config(model_family, config)
                    
                    print(f"\n{'='*80}")
                    print(f"Starting AdaLoRA experiment:")
                    print(f"  Task: {task}")
                    print(f"  Model family: {model_family}")
                    print(f"  Rank: {rank}")
                    print(f"  Seed: {seed}")
                    print(f"  Teacher: {config['teacher_model_name']}")
                    print(f"  Student: {config['student_model_name']}")
                    print(f"{'='*80}\n")
                    
                    try:
                        if dry_run:
                            print(f"[DRY RUN] Config: {config}")
                            # Simulate creating pipeline but don't run
                            continue
                        
                        pipe = BertDistillPipeline(**config)
                        
                        if run_teacher:
                            print("Running teacher AdaLoRA...")
                            pipe.run_teacher_lora()
                        
                        if run_student:
                            print("Running student AdaLoRA distillation...")
                            pipe.run_student_lora()
                            
                    except Exception as e:
                        print(f"Error in experiment (task={task}, family={model_family}, rank={rank}, seed={seed}):")
                        print(f"  {e}")
                        # Optionally continue with next experiment
                        continue
    
    print("\nAll AdaLoRA experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AdaLoRA experiments (teacher and student distillation) with fixed configuration"
    )
    
    # Model arguments
    parser.add_argument("--teacher_model_name", type=str, default="./models/bert-base-uncased",
                        help="Name of the teacher model (overridden by model_family)")
    parser.add_argument("--student_model_name", type=str, default="./models/distilbert-base-uncased",
                        help="Name of the student model (overridden by model_family)")
    
    # Dataset and training parameters
    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    parser.add_argument("--dir_name", type=str, default="./results", help="Directory name for saving models")
    
    # LoRA parameters (alpha fixed at 16, dropout as specified)
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA matrices (overridden by RANK_VALUES loop)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor (fixed at 16)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout rate for LoRA layers")
    
    # Learning rates for teacher and student
    parser.add_argument("--teacher_learning_rate", type=float, default=5e-5, help="Learning rate for the teacher model")
    parser.add_argument("--student_learning_rate", type=float, default=5e-4, help="Learning rate for the student model")
    
    parser.add_argument('--task', type=str, default="wnli", choices=tuple(GLUE_TASKS),
                        help="Name of the task (overridden by loop)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (overridden by loop)")
    
    parser.add_argument('--from_disk', type=int, default=1, help="If 1, use load_from_disk()")
    
    # Control which experiments to run
    parser.add_argument('--dry-run', action='store_true',
                        help="Print configurations without running training")
    parser.add_argument('--teacher_only', action='store_true',
                        help="Run only teacher AdaLoRA fine-tuning")
    parser.add_argument('--student_only', action='store_true',
                        help="Run only student AdaLoRA distillation")
    
    args_cmd = parser.parse_args()
    
    # Determine which experiments to run
    run_teacher = not args_cmd.student_only
    run_student = not args_cmd.teacher_only
    
    # If both flags are set (contradictory), run both
    if args_cmd.teacher_only and args_cmd.student_only:
        run_teacher = run_student = True
    
    print(f"AdaLoRA Experiment Configuration:")
    print(f"  Dataset path: {args_cmd.dataset_path}")
    print(f"  Output directory: {args_cmd.dir_name}")
    print(f"  Run teacher: {run_teacher}")
    print(f"  Run student: {run_student}")
    print(f"  Ranks to test: {RANK_VALUES}")
    print(f"  Seeds: {seed_list}")
    print(f"  Tasks: {GLUE_TASKS}")
    print(f"  Model families: {list(MODEL_FAMILY.keys())}")
    
    main_adalora(args_cmd, run_teacher, run_student, dry_run=args_cmd.dry_run)