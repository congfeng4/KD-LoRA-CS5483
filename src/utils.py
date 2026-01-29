import os
import logging
import numpy as np
from peft import LoraConfig
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from transformers.trainer_pt_utils import get_model_param_count
from transformers import Trainer, TrainerCallback, AutoTokenizer
import torch
import torch.nn.functional as F


GLUE_TASKS = [
  "wnli", "rte", "qnli",
     "mrpc", "qqp", "stsb",
   "mnli", "cola", "sst2",
]


# Suppress tokenizer warning about overflowing tokens not returned for 'longest_first' truncation strategy
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Caches for sharing datasets and tokenizers across runs in the same process
_RAW_DATASET_CACHE = {}
_TOKENIZER_CACHE = {}
_TOKENIZED_DATASET_CACHE = {}

def get_raw_dataset(dataset_path, task, from_disk=True):
    """Load raw GLUE dataset with caching."""
    key = (dataset_path, task, from_disk)
    if key not in _RAW_DATASET_CACHE:
        print(f"[CACHE MISS] Loading raw dataset: task={task}, from_disk={from_disk}")
        _RAW_DATASET_CACHE[key] = load_glue_dataset(dataset_path, task, from_disk)
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


MODEL_FAMILY = {
    'bert': {
        'teacher': 'bert-base-uncased',
        'student': 'distilbert-base-uncased',
    },
    'roberta': {
        'teacher': 'roberta-base',
        'student': 'distilroberta-base',
    },
    'deberta': {
        'teacher': 'deberta-v3-base',
        'student': 'deberta-v3-small',
    }
}

PEFT_FAMILY = [
    "lora",  # Vanilla lora
    "olora",  # orthonormal lora
    "dora",  # weight decomposed lora
    "adalora",  # Adaptive lora
    "rslora",  # Rank stablized lora
    "mrlora", # Multi-Rank lora
    "mrlora-rs", # Multi-Rank LoRA with rank-stabilized scaling
]


def load_glue_dataset(dataset_path, task, from_disk=True):
    if from_disk:
        from datasets import load_from_disk
        return load_from_disk(os.path.join(dataset_path, task))
    else:
        from datasets import load_dataset
        return load_dataset('glue', task, cache_dir=dataset_path)


def get_trainable_param_count(model):
    return get_model_param_count(model, trainable_only=True) / 1e6  # M


def get_target_modules(model_name):
    model_name_lower = model_name.lower()
    if "distilbert" in model_name_lower:
        target_modules = ["q_lin", "v_lin"]
    elif "deberta" in model_name_lower:
        target_modules = ["query_proj", "value_proj"]
    else:
        # 适用于 BERT, RoBERTa, DistilRoBERTa
        target_modules = ["query", "value"]
    # 动态匹配 target_modules
    print('get_target_moduele model_name_lower', model_name_lower, 'target_modules', target_modules)
    return target_modules


def get_peft_config(args, model_name, peft_method):
    target_modules = get_target_modules(model_name)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )
    if peft_method == 'lora':
        return lora_config

    if peft_method == 'olora':
        # https://github.com/huggingface/peft/blob/main/examples/olora_finetuning/README.md
        lora_config.init_lora_weights = 'olora'
        return lora_config

    if peft_method == 'dora':
        # https://github.com/huggingface/peft/tree/main/examples/dora_finetuning
        lora_config.use_dora = True
        return lora_config

    if peft_method == 'rslora':
        lora_config.use_rslora = True
        return lora_config

    if peft_method == 'adalora':
        from peft import AdaLoraConfig
        adalora_config = AdaLoraConfig(
            init_r=12,               # Start higher
            target_r=8,              # Final average rank
            beta1=0.85,              # Smoothing for importance score
            beta2=0.85,              # Uncertainty for importance score
            tinit=200,               # Steps before pruning starts
            tfinal=1000,             # Steps when pruning ends
            deltaT=10,               # Interval between pruning steps
            orth_reg_weight=0.5,     # Orthogonal regularization coefficient
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            task_type="SEQ_CLS",
        )
        return adalora_config

    if peft_method in ['mrlora', 'mrlora-rs']:
        from mrlora import MrLoraConfig
        # For 'mrlora-rs' variant, force use_rslora=True
        if peft_method == 'mrlora-rs':
            args.use_rslora = True
        mrlora_config = MrLoraConfig(
            ranks=args.lora_ranks,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="SEQ_CLS",
            use_rslora=args.use_rslora,
        )
        return mrlora_config

    print('Unknown peft method', peft_method)
    return lora_config


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    # Determine if this is a regression task (single output dimension)
    if student_logits.size(-1) == 1:
        # Regression task: use MSE loss
        # Squeeze the last dimension
        student_logits = student_logits.squeeze(-1)
        teacher_logits = teacher_logits.squeeze(-1)
        # Labels are already float, no conversion needed
        soft_loss = F.mse_loss(student_logits, teacher_logits)
        hard_loss = F.mse_loss(student_logits, labels)
        return alpha * soft_loss + (1 - alpha) * hard_loss
    else:
        # Classification task
        # Ensure labels are long integers for cross-entropy
        labels = labels.long()
        
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
        # Ensure teacher_logits match student_logits dtype for mixed precision compatibility
        teacher_logits = teacher_logits.to(student_logits.dtype)
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
        print(
            f"Epoch {state.epoch}: Allocated Memory: {allocated_memory:.2f} MB, Reserved Memory: {reserved_memory:.2f} MB")


def check_model_family():
    for model in MODEL_FAMILY.values():
        print('checking model', model)
        assert os.path.exists("./models/" + model['teacher'])
        assert os.path.exists("./models/" + model['student'])


def add_model_name_to_config(model_family, config: dict):
    param = MODEL_FAMILY[model_family]
    teacher = param['teacher']
    student = param['student']
    config['student_model_name'] = "./models/" + student
    config['teacher_model_name'] = "./models/" + teacher
    config['model_family'] = model_family


def get_train_metrics(trainer_output, model, callback: MemoryTrackingCallback):
    trainable_params_count = get_trainable_param_count(model)
    train_time = trainer_output.metrics['train_runtime']
    return dict(train_time=train_time, trainable_params_count=trainable_params_count,
                memory_allocated=callback.memory_allocated, memory_reserved=callback.memory_reserved)


def get_num_labels(args):
    # Determine the number of labels based on the GLUE task
    if args.task == "stsb":
        num_labels = 1  # Regression task for STS-B
    elif args.task == "mnli":
        num_labels = 3  # Multi-class classification for MNLI
    else:
        num_labels = 2  # Default binary classification
    return num_labels


# Define a function to compute evaluation metrics based on the task
def compute_metrics(args):
    def func(p):
        predictionss, labels = p
        # print('predictionss shape', predictionss.shape, 'labels shape', labels.shape)
        # Convert logits to class predictions for classification tasks
        if predictionss.ndim == 2:
            # If second dimension > 1, it's classification (logits)
            if predictionss.shape[1] > 1:
                predictions = np.argmax(predictionss, axis=1).astype(int)
            else:
                # Regression task (single output)
                predictions = predictionss.squeeze()
        else:
            # Already class predictions (1D array)
            predictions = predictionss
            
        if args.task == "mrpc":
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="binary")
            return {"accuracy": accuracy, "f1": f1}
        if args.task == "cola":
            mcc = matthews_corrcoef(labels, predictions)
            return {"matthews_correlation": mcc}
        if args.task == "sst2":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if args.task == "wnli":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if args.task == "rte":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if args.task == "qqp":
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="binary")
            return {"accuracy": accuracy, "f1": f1}
        if args.task == "stsb":
            preds = predictionss.squeeze().tolist()
            labels = labels.tolist()
            pearson_corr, _ = pearsonr(preds, labels)
            spearman_corr, _ = spearmanr(preds, labels)
            return {'pearson': pearson_corr, 'spearman': spearman_corr}
        if args.task == "qnli":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if args.task == "mnli":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        
        # Default return for unknown tasks
        return {}

    return func


# Define a function to tokenize the input data according to each task
def tokenize_function(args, tokenizer, with_indices=False):
    def func(examples):

        # Tokenize based on dataset-specific requirements
        if args.task == "mrpc":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation='longest_first', max_length=128, return_overflowing_tokens=False)
        if args.task == "cola":
            return tokenizer(examples["sentence"], truncation='longest_first', padding="max_length", max_length=128, return_overflowing_tokens=False)
        if args.task == "sst2":
            return tokenizer(examples["sentence"], truncation='longest_first', padding="max_length",max_length=128, return_overflowing_tokens=False)
        if args.task == "wnli":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation='longest_first',max_length=128, return_overflowing_tokens=False)
        if args.task == "rte":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation='longest_first',max_length=128, return_overflowing_tokens=False)
        if args.task == "qqp":
            return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation='longest_first',max_length=128, return_overflowing_tokens=False)
        if args.task == "stsb":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation='longest_first',max_length=128, return_overflowing_tokens=False)
        if args.task == "qnli":
            return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation='longest_first',max_length=128, return_overflowing_tokens=False)
        if args.task == "mnli":
            return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation='longest_first',max_length=128, return_overflowing_tokens=False)
        raise ValueError(f"Unsupported task: {args.task}")

    if with_indices:
        def func2(examples, idx):
            result = func(examples)
            result['idx'] = idx
            return result

        return func2

    return func


def get_tokenized_dataset(task, tokenizer_name, with_indices=False, dataset_path=None, from_disk=True):
    """Get tokenized dataset with caching.
    
    Tokenized dataset is cached per (task, tokenizer_name, with_indices).
    dataset_path is required for first-time loading.
    """
    key = (task, tokenizer_name, with_indices)
    if key not in _TOKENIZED_DATASET_CACHE:
        print(f"[CACHE MISS] Tokenizing dataset: task={task}, tokenizer={tokenizer_name}, with_indices={with_indices}")
        if dataset_path is None:
            raise ValueError("dataset_path must be provided for first-time tokenization")
        raw_dataset = get_raw_dataset(dataset_path, task, from_disk)
        tokenizer = get_tokenizer(tokenizer_name)
        # Create tokenization function with the given task
        args = type('Args', (), {'task': task})()
        tokenize_fn = tokenize_function(args, tokenizer, with_indices=with_indices)
        # Apply tokenization
        tokenized = raw_dataset.map(tokenize_fn, batched=True, keep_in_memory=True, with_indices=with_indices)
        _TOKENIZED_DATASET_CACHE[key] = tokenized
    else:
        print(f"[CACHE HIT] Using cached tokenized dataset: task={task}, tokenizer={tokenizer_name}, with_indices={with_indices}")
    return _TOKENIZED_DATASET_CACHE[key]

import torch
import gc


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保卷积等操作也是确定的（虽然 BERT 用得少，但建议加上）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

