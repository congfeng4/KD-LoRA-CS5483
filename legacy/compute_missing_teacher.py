#!/usr/bin/env python3
import json
import statistics
import os
import pandas as pd

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
    elif task in ["qnli", "rte", "wnli", "mnli"]:
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
    lora_root = "/Users/fengcong/PycharmProjects/KD-LoRA/results/lora"
    fft_baseline = load_fft_baseline()
    
    # Collect raw scores for MR‑LoRA‑RS for tasks qqp, stsb, mnli
    scores = {}  # family -> task -> list
    for root, dirs, files in os.walk(lora_root):
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
                if method != "mrlora-rs":
                    continue
                # Only teacher fine-tuning (type != 1?)
                # Not sure; assume all in lora directory are teacher
                metric_val = get_task_metric(data, task)
                if metric_val is None:
                    continue
                metric_val *= 100  # to percentage
                if model_family not in scores:
                    scores[model_family] = {}
                if task not in scores[model_family]:
                    scores[model_family][task] = []
                scores[model_family][task].append(metric_val)
    
    print("Collected scores:", scores.keys())
    for fam in scores:
        for task in scores[fam]:
            print(f"{fam} {task}: {scores[fam][task]}")
    
    # Compute percentages relative to FFT baseline
    tasks_needed = ["qqp", "stsb", "mnli"]
    for task in tasks_needed:
        percentages = []
        for fam in ["bert", "roberta", "deberta"]:
            if fam in scores and task in scores[fam]:
                raw_avg = statistics.mean(scores[fam][task])
                fft = fft_baseline[fam].get(task)
                if fft and fft != 0:
                    pct = raw_avg / fft * 100
                    percentages.append(pct)
                    print(f"{fam} {task}: raw {raw_avg:.2f}, fft {fft:.2f}, pct {pct:.1f}%")
        if percentages:
            avg_pct = statistics.mean(percentages)
            print(f"Average across families for {task}: {avg_pct:.1f}%")
        else:
            print(f"No data for {task}")

if __name__ == "__main__":
    main()