import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr


glue_tasks = [
    "cola", "sst2", "mrpc", "qqp", "stsb",
    "mnli", "qnli", "rte", "wnli",
]

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
