
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

# Function to tokenize dataset based on task-specific requirements
def tokenize_function(task: str, tokenizer):

    def func(examples):
        if task == "mrpc":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if task == "cola":
            return tokenizer(examples["sentence"], truncation=True, padding="max_length")
        if task == "sst2":
            return tokenizer(examples["sentence"], truncation=True, padding="max_length")
        if task == "wnli":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if task == "rte":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if task == "qqp":
            return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True)
        if task == "stsb":
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
        if task == "qnli":
            return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True)
        if task == "mnli":
            return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True)
    return func

def compute_metrics(task: str):

    def func(p):
        predictionss, labels = p
        predictions = np.argmax(predictionss, axis=1)

        # Choose metrics based on dataset type
        if task == "mrpc":
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="binary")
            return {"accuracy": accuracy, "f1": f1}
        if task == "cola":
            mcc = matthews_corrcoef(labels, predictions)
            return {"matthews_correlation": mcc}
        if task == "sst2":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if task == "wnli":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if task == "rte":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if task == "qqp":
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="binary")
            return {"accuracy": accuracy, "f1": f1}
        if task == "stsb":
            preds = predictionss.squeeze().tolist()
            labels = labels.tolist()
            pearson_corr, _ = pearsonr(preds, labels)
            spearman_corr, _ = spearmanr(preds, labels)
            return {
                'pearson': pearson_corr,
                'spearman': spearman_corr
            }
        if task == "qnli":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        if task == "mnli":
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
    return func

glue_tasks = [
    "cola", "sst2", "mrpc", "qqp", "stsb",
    "mnli", "qnli", "rte", "wnli",
]
