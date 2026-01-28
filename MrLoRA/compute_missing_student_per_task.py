#!/usr/bin/env python3
"""
Compute per‑task percentages for student variants (AdaLoRA, DoRA, OLoRA, RS‑LoRA, MR‑LoRA‑RS)
using metrics.json files from results/kd‑lora/.
"""

import json
import os
import pandas as pd
import statistics
from pathlib import Path

# Mapping from task name to metric key in metrics.json
TASK_METRIC = {
    "cola": "eval_matthews_correlation",
    "sst2": "eval_accuracy",
    "mrpc": "eval_f1",  # MRPC uses average of accuracy and F1; we need to compute average
    "qqp": "eval_f1",   # QQP uses average of accuracy and F1
    "stsb": "eval_pearson",
    "qnli": "eval_accuracy",
    "rte": "eval_accuracy",
    "wnli": "eval_accuracy",
    "mnli": "eval_accuracy",  # matched/mismatched handled separately
}

# Model family mapping from directory names
FAMILY_MAP = {
    "bert": "bert",
    "roberta": "roberta",
    "deberta": "deberta",
}

# Method name mapping from peft directory prefix
METHOD_MAP = {
    "adalora": "AdaLoRA",
    "dora": "DoRA",
    "olora": "OLoRA",
    "rslora": "RS‑LoRA",
    "mrlora-rs": "MR‑LoRA‑RS",
}

def extract_score(metrics, task):
    """Extract the appropriate score from metrics dict for given task."""
    if task == "mrpc":
        # MRPC: average of accuracy and F1
        acc = metrics.get("eval_accuracy")
        f1 = metrics.get("eval_f1")
        if acc is not None and f1 is not None:
            return (acc + f1) / 2.0
        elif f1 is not None:
            return f1
        else:
            return acc
    elif task == "qqp":
        # QQP: average of accuracy and F1
        acc = metrics.get("eval_accuracy")
        f1 = metrics.get("eval_f1")
        if acc is not None and f1 is not None:
            return (acc + f1) / 2.0
        elif f1 is not None:
            return f1
        else:
            return acc
    else:
        metric_key = TASK_METRIC[task]
        return metrics.get(metric_key)

def collect_student_scores(results_root="/Users/fengcong/PycharmProjects/KD-LoRA/results/kd-lora"):
    """Walk through results directory and collect scores."""
    data = []  # list of (model_family, method, task, score)
    
    for path in Path(results_root).rglob("metrics.json"):
        # Parse path components
        parts = path.parts
        # Expect pattern: .../task_{task}_{model}_{seed}/base_.../peft_{method}_.../metrics.json
        task_dir = parts[-3]
        if not task_dir.startswith("task_"):
            continue
        # Extract task and model family
        # task_dir format: task_{task}_{model}_{seed}
        _, task, model_family, seed = task_dir.split("_", 3)
        # Normalize task name
        task = task.lower()
        if task not in TASK_METRIC:
            continue
        # Map model family
        model_family = FAMILY_MAP.get(model_family)
        if model_family is None:
            continue
        # Extract method from peft directory name
        peft_dir = parts[-2]
        if not peft_dir.startswith("peft_"):
            continue
        method = peft_dir.split("_")[1]  # peft_adalora_16_0.05_8 -> adalora
        if method not in METHOD_MAP:
            continue
        
        # Load metrics.json
        try:
            with open(path, 'r') as f:
                metrics = json.load(f)
        except:
            continue
        
        # Extract score
        score = extract_score(metrics, task)
        if score is None:
            continue
        
        data.append({
            "model_family": model_family,
            "method": method,
            "task": task,
            "score": score
        })
    
    return data

def compute_percentages(student_scores, fft_baseline):
    """
    Compute per‑task percentages relative to FFT baseline, averaged across model families.
    student_scores: list of dicts
    fft_baseline: dict {model_family: {task: score}}
    Returns: dict {method: {task: percentage}}
    """
    # Group by method and task
    from collections import defaultdict
    scores_by_method_task = defaultdict(list)  # (method, task) -> list of scores per family
    families_by_method_task = defaultdict(set)
    
    for entry in student_scores:
        key = (entry["method"], entry["task"])
        scores_by_method_task[key].append(entry["score"])
        families_by_method_task[key].add(entry["model_family"])
    
    # Compute percentages
    percentages = defaultdict(dict)
    for (method, task), scores in scores_by_method_task.items():
        # We need percentage per family then average
        pcts = []
        for entry in student_scores:
            if entry["method"] == method and entry["task"] == task:
                fam = entry["model_family"]
                if fam in fft_baseline and task in fft_baseline[fam]:
                    fft = fft_baseline[fam][task]
                    if fft != 0:
                        pct = (entry["score"] / fft) * 100
                        pcts.append(pct)
        if pcts:
            avg_pct = statistics.mean(pcts)
            percentages[method][task] = avg_pct
        else:
            percentages[method][task] = None
    
    return percentages

def load_fft_baseline():
    """Load FFT baseline scores from table_i_detailed.csv."""
    df = pd.read_csv("table_i_detailed.csv")
    # Columns: Task, BERT‑b/DBERT‑b FFT, RoB‑b/DRoB‑b FFT, DeB‑b/DeB‑s FFT
    fft = {}
    family_cols = {
        "bert": "BERT‑b/DBERT‑b FFT",
        "roberta": "RoB‑b/DRoB‑b FFT",
        "deberta": "DeB‑b/DeB‑s FFT"
    }
    # Map CSV task names to metrics.json task names
    task_map = {
        "cola": "cola",
        "sst‑2": "sst2",
        "mrpc": "mrpc",
        "qqp": "qqp",
        "sts‑b": "stsb",
        "qnli": "qnli",
        "rte": "rte",
        "wnli": "wnli"
    }
    for fam, col in family_cols.items():
        fft[fam] = {}
        for _, row in df.iterrows():
            task = row["Task"].lower()
            if task in task_map:
                task_key = task_map[task]
                fft[fam][task_key] = float(row[col])
    return fft

def main():
    print("Loading FFT baseline...")
    fft_baseline = load_fft_baseline()
    
    print("Collecting student scores from results/kd-lora...")
    student_scores = collect_student_scores()
    print(f"Found {len(student_scores)} data points.")
    
    # Group by method to see coverage
    from collections import defaultdict
    method_counts = defaultdict(int)
    for entry in student_scores:
        method_counts[entry["method"]] += 1
    print("Data points per method:", dict(method_counts))
    
    # Compute percentages
    percentages = compute_percentages(student_scores, fft_baseline)
    
    # Print results
    tasks = ["cola", "sst2", "mrpc", "qqp", "stsb", "qnli", "rte", "wnli"]
    for method in METHOD_MAP.keys():
        if method in percentages:
            print(f"\n{METHOD_MAP[method]}:")
            for task in tasks:
                pct = percentages[method].get(task)
                if pct is not None:
                    print(f"  {task}: {pct:.1f}%")
                else:
                    print(f"  {task}: --")
        else:
            print(f"\n{METHOD_MAP[method]}: no data")
    
    # Optionally output LaTeX table rows
    print("\nLaTeX rows:")
    for method in METHOD_MAP.keys():
        if method in percentages:
            row = METHOD_MAP[method]
            for task in tasks:
                pct = percentages[method].get(task)
                if pct is not None:
                    row += f" & {pct:.1f}\\%"
                else:
                    row += " & --"
            # GLUE percentage (from glue_averages.csv) – we can compute average across tasks
            task_pcts = [percentages[method].get(t) for t in tasks]
            task_pcts = [p for p in task_pcts if p is not None]
            if task_pcts:
                glue_pct = statistics.mean(task_pcts)
                row += f" & {glue_pct:.1f}\\%"
            else:
                row += " & --"
            row += " \\\\"
            print(row)

if __name__ == "__main__":
    main()