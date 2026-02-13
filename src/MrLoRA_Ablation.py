import argparse
import itertools
import json
import shutil
import sys

from addict import Addict
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model
from BERT_Distill_LoRA import *
from utils import *

# Suppress tokenizer warning about overflowing tokens not returned for 'longest_first' truncation strategy
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

RANK_VALUES = [8]
seed_list = [42]

# MRLORA_VARIANTS = ['-olora', '-rs', '-lcoef']#, '-bias']
# GLUE_TASKS = [
#     "rte", "qnli",
#     "mrpc", "qqp", "stsb",
#     "mnli", "cola", "sst2",
# ]
GLUE_TASKS = ['cola', 'qnli', 'rte']

MODEL_FAMILY = {
    'deberta': {
        'teacher': 'deberta-v3-base',
        'student': 'deberta-v3-small',
    }
}

MRLORA_VARIANTS = ['-rs', '-olora', '-lcoef']

PEFT_FAMILY = ['mrlora']

for i in range(len(MRLORA_VARIANTS)):
    PEFT_FAMILY.extend('mrlora' + "".join(item) for item in itertools.combinations(MRLORA_VARIANTS, i+1))

print(PEFT_FAMILY)


def main_lora(args, is_student: bool):
    for rank in RANK_VALUES:
        print('rank', rank)
        for seed in seed_list:
            print('seed', seed)
            for task in GLUE_TASKS:
                print('task', task)
                for model_family in MODEL_FAMILY.keys():
                    print('family', model_family)
                    for peft_method in PEFT_FAMILY:
                        print('peft', peft_method)
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

    parser.add_argument("--dir_name", type=str, default="./ablation3", help="Directory name for saving models")

    # LoRA parameters
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout rate for LoRA layers")
    parser.add_argument('--use_rslora', action='store_true',
                        help='Use rank-stabilized scaling for MrLoRA (lora_alpha/sqrt(r) instead of lora_alpha/max(ranks))')
    parser.add_argument('--use_olora', action='store_true',
                        help='Use orthonormal initialization for MrLoRA')
    parser.add_argument("--lora_alpha", type=float, default=16, help="Number of training epochs")

    # Learning rates for teacher and student
    parser.add_argument("--teacher_learning_rate", type=float, default=2e-5, help="Learning rate for the teacher model")
    parser.add_argument("--student_learning_rate", type=float, default=1e-4, help="Learning rate for the student model")
    parser.add_argument("--lora_learning_rate", type=float, default=2e-4, help="Learning rate for the student model")

    args_cmd = parser.parse_args()
    
    main_lora(args_cmd, is_student=False)
