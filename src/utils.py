import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from transformers.trainer_pt_utils import get_model_param_count
from transformers import Trainer, TrainerCallback
import torch
import torch.nn.functional as F


glue_tasks = [
    "cola", "sst2", "mrpc", "qqp", "stsb",
    "mnli", "qnli", "rte", "wnli",
]

model_family = {
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


def check_model_family():
    for model in model_family.values():
        print('checking model', model)
        assert os.path.exists("./models/" + model['teacher'])
        assert os.path.exists("./models/" + model['student'])


def get_train_metrics(trainer_output, model, callback: MemoryTrackingCallback):
    trainable_params_count = get_model_param_count(model, trainable_only=True)
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
        predictions = np.argmax(predictionss, axis=1)
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
        
    return func


# Define a function to tokenize the input data according to each task
def tokenize_function(args, tokenizer, with_indices=False):
    def func(examples):

        # Tokenize based on dataset-specific requirements
        if args.task == "mrpc":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if args.task == "cola":
            return tokenizer(examples["sentence"], truncation=True, padding="max_length")
        if args.task == "sst2":
            return tokenizer(examples["sentence"], truncation=True, padding="max_length")
        if args.task == "wnli":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if args.task == "rte":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if args.task == "qqp":
            return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True)
        if args.task == "stsb":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if args.task == "qnli":
            return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True)
        if args.task == "mnli":
            return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True)
    
    if with_indices:
        def func2(examples, idx):
            return {**func(examples), 'idx': idx}
        return func2

    return func
