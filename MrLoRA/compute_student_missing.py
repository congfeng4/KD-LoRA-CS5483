#!/usr/bin/env python3
import json
import os
import statistics
import pandas as pd
from collections import defaultdict

def get_task_metric(data, task):
    if task == "cola":
        return data.get("eval_matthews_correlation")
    elif task == "sst2":
        return data.get("eval_accuracy")
    elif task == "mrpc":
        acc = data.get("eval_accuracy")
        f1 = data.get("eval_f1")
        if acc is not None and f1 is not None:
            return (acc + f1) / 2
        else:
            return None
    elif task == "qqp":
        acc = data.get("eval_accuracy")
        f1 = data.get("eval_f1")
        if acc is not None and f1 is not None:
            return (acc + f1) / 2
        else:
            return None
    elif task == "stsb":
        return data.get("eval_pearson")
    elif task in ["qnli", "rte", "wnli"]:
        return data.get("eval_accuracy")
    else:
        return None

def load_fft_baseline():
    df = pd.read_csv("table_i_detailed.csv")
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
    kd_lora_root = "/Users/fengcong/PycharmProjects/KD-LoRA/results/kd-lora"
    fft_baseline = load_fft_baseline()
    
    # Collect scores: dict[method][family][task] = list of scores (should be one per seed)
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for root, dirs, files in os.walk(kd_lora_root):
        for file in files:
            if file == "metrics.json":
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except:
                    continue
                args = data.get("args", {})
                model_family = args.get("model_family")
                task = args.get("task")
                method = args.get("peft")
                if not all([model_family, task, method]):
                    continue
                # Only keep student distillation (type=1)
                if args.get("type") != 1:
                    continue
                metric_val = get_task_metric(data, task)
                if metric_val is None:
                    continue
                metric_val *= 100  # convert to percentage
                scores[method][model_family][task].append(metric_val)
    
    print(f"Loaded data for methods: {list(scores.keys())}")
    
    # Average across seeds per family
    avg_scores = defaultdict(lambda: defaultdict(dict))
    for method in scores:
        for family in scores[method]:
            for task in scores[method][family]:
                vals = scores[method][family][task]
                avg_scores[method][family][task] = statistics.mean(vals) if vals else None
    
    # Compute percentages relative to FFT baseline, then average across families
    tasks = ["cola", "sst2", "mrpc", "qqp", "stsb", "qnli", "rte", "wnli"]
    method_names = {
        "adalora": "AdaLoRA",
        "dora": "DoRA",
        "olora": "OLoRA",
        "rslora": "RS‑LoRA",
        "mrlora-rs": "MR‑LoRA‑RS"
    }
    
    results = {}
    for method in method_names:
        if method not in avg_scores:
            print(f"No data for {method}")
            continue
        task_pcts = []
        for task in tasks:
            pcts = []
            for family in ["bert", "roberta", "deberta"]:
                if family in avg_scores[method] and task in avg_scores[method][family]:
                    raw = avg_scores[method][family][task]
                    if raw is not None and family in fft_baseline and task in fft_baseline[family]:
                        fft = fft_baseline[family][task]
                        if fft != 0:
                            pcts.append(raw / fft * 100)
            if pcts:
                avg = statistics.mean(pcts)
                task_pcts.append(avg)
                print(f"{method_names[method]} {task}: {avg:.1f}%")
            else:
                print(f"{method_names[method]} {task}: --")
        if task_pcts:
            glue_pct = statistics.mean(task_pcts)
            print(f"{method_names[method]} GLUE: {glue_pct:.1f}%")
        results[method] = task_pcts
    
    # Generate LaTeX rows
    print("\nLaTeX rows:")
    for method in method_names:
        name = method_names[method]
        if method not in results:
            continue
        task_pcts = results[method]
        if len(task_pcts) != len(tasks):
            # some missing tasks
            continue
        row = name
        for pct in task_pcts:
            row += f" & {pct:.1f}\\%"
        row += " & {glue_pct:.1f}\\% \\\\"
        print(row)

if __name__ == "__main__":
    main()