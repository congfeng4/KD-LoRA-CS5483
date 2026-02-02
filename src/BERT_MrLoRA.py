import argparse
import json
import shutil

from addict import Addict
from copy import deepcopy
from pathlib import Path
from BERT_Distill_LoRA import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model
from utils import *

# Suppress tokenizer warning about overflowing tokens not returned for 'longest_first' truncation strategy
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

RANK_VALUES = [8]


PEFT_FAMILY_BASELINES = [
    # Baselines
    "lora",  # Vanilla lora
    "olora",  # orthonormal lora
    "dora",  # weight decomposed lora
    "rslora",  # Rank stablized lora
    "adalora", # Adaptive LoRA
]

PEFT_FAMILY_OURS = [
    # Ours
    "mrlora-rs", # Multi-Rank LoRA with rank-stabilized scaling
    "mrlora-rs-olora", # Multi-Rank LoRA with rank-stabilized scaling
    "mrlora-olora", # Multi-Rank LoRA with rank-stabilized scaling
    "mrlora", # Plain mrlora
]

GLUE_TASKS = ['wnli']


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
    PEFT_FAMILY = PEFT_FAMILY_OURS if args.ours else PEFT_FAMILY_BASELINES
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

    # Model arguments

    # Dataset and training parameters
    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    parser.add_argument("--dir_name", type=str, default="./results", help="Directory name for saving models")

    # LoRA parameters
    parser.add_argument('--use_rslora', action='store_true',
                        help='Use rank-stabilized scaling for MrLoRA (lora_alpha/sqrt(r) instead of lora_alpha/max(ranks))')

    # Learning rates for teacher and student
    parser.add_argument("--teacher_learning_rate", type=float, default=2e-5, help="Learning rate for the teacher model")
    parser.add_argument("--student_learning_rate", type=float, default=1e-4, help="Learning rate for the student model")
    parser.add_argument("--lora_learning_rate", type=float, default=2e-4, help="Learning rate for the student model")
    parser.add_argument("--num_train_epochs", type=int, default=MAX_EPOCHS, help="Number of training epochs")
    parser.add_argument("--peft", type=str, default='lora', help="Number of training epochs")
    parser.add_argument("--rank", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--lora_alpha", type=float, default=16, help="Number of training epochs")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout rate for LoRA layers")
    parser.add_argument("--ours", action='store_true')

    parser.add_argument('--type', '-t', type=int, choices=(0, 1, 2),
                        help='0 => fft, 1 => student-lora, 2 => teacher-lora')

    args_cmd = parser.parse_args()
    # Only LoRA...
    main_lora(args_cmd, is_student=False)
