import argparse
import json
import shutil

from addict import Addict
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model
from BERT_Distill_LoRA import BertDistillPipeline
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
    parser.add_argument("--student_learning_rate", type=float, default=1e-4, help="Learning rate for the student model")
    parser.add_argument("--lora_learning_rate", type=float, default=2e-4, help="Learning rate for the student model")

    args_cmd = parser.parse_args()
    main_teacher_fft(args_cmd)
    main_lora(args_cmd, is_student=True)
    main_lora(args_cmd, is_student=False)
