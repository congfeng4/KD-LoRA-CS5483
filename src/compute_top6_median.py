#!/usr/bin/env python3
"""
Compute median of top 6 hyperparameter and PEFT setups for KD-LoRA results.
Reads all metrics.json files in results/{variant}/ directories, groups by
(task, model_family, seed, variant), ranks hyperparameter configurations by
validation metric, selects top 6, computes median of those top 6.
Outputs a CSV file with the median results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Metric mapping per task
TASK_METRIC = {
    "cola": "eval_matthews_correlation",
    "sst2": "eval_accuracy",
    "mrpc": "eval_accuracy",  # could also use eval_f1
    "qqp": "eval_accuracy",
    "stsb": "eval_pearson",
    "mnli": "matched_accuracy",
    "qnli": "eval_accuracy",
    "rte": "eval_accuracy",
    "wnli": "eval_accuracy",
}

def get_metric_values(data, task):
    """Extract primary metric value(s) from metrics.json data.
    For MNLI, returns dict with 'matched' and 'mismatched' accuracies.
    For other tasks, returns dict with 'primary' metric.
    """
    if task == 'mnli':
        matched = data.get('matched_accuracy')
        mismatched = data.get('mismatched_accuracy')
        result = {}
        if matched is not None:
            result['matched'] = matched
        if mismatched is not None:
            result['mismatched'] = mismatched
        # fallback to eval_accuracy from log_history
        if not result and 'log_history' in data:
            for entry in reversed(data['log_history']):
                if 'eval_accuracy' in entry:
                    result['matched'] = entry['eval_accuracy']
                    break
        return result
    else:
        metric_key = TASK_METRIC.get(task)
        if metric_key is None:
            # fallback: find any eval_* key
            for key in data.keys():
                if key.startswith('eval_'):
                    metric_key = key
                    break
        if metric_key and metric_key in data:
            return {'primary': data[metric_key]}
        # if not found, maybe log_history contains evaluation
        if 'log_history' in data:
            for entry in reversed(data['log_history']):
                for key in entry:
                    if key.startswith('eval_'):
                        return {'primary': entry[key]}
        return {}

def collect_results(results_dir='./results'):
    results = []
    base_path = Path(results_dir)
    for variant_dir in base_path.iterdir():
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name  # fft, lora, kd-lora
        # iterate over task directories
        for task_dir in variant_dir.iterdir():
            if not task_dir.is_dir():
                continue
            # task_dir name: task_{task}_{model_family}_{seed}
            parts = task_dir.name.split('_')
            if len(parts) < 4:
                continue
            task = parts[1]
            model_family = parts[2]
            seed = int(parts[3])
            # iterate over base hyperparameter directories
            for base_dir in task_dir.iterdir():
                if not base_dir.is_dir():
                    continue
                # base_dir name: base_{batch_size}_{teacher_lr}_{weight_decay}
                base_parts = base_dir.name.split('_')
                if len(base_parts) < 4 or base_parts[0] != 'base':
                    continue
                batch_size = int(base_parts[1])
                teacher_lr = float(base_parts[2])
                weight_decay = float(base_parts[3])
                # iterate over peft hyperparameter directories
                for peft_dir in base_dir.iterdir():
                    if not peft_dir.is_dir():
                        continue
                    # peft_dir name: peft_{peft}_{lora_alpha}_{lora_dropout}_{rank}
                    peft_parts = peft_dir.name.split('_')
                    if len(peft_parts) < 5 or peft_parts[0] != 'peft':
                        continue
                    peft = peft_parts[1]
                    lora_alpha = int(peft_parts[2])
                    lora_dropout = float(peft_parts[3])
                    rank = int(peft_parts[4])
                    metrics_file = peft_dir / 'metrics.json'
                    if not metrics_file.exists():
                        continue
                    try:
                        with open(metrics_file, 'r') as f:
                            data = json.load(f)
                        metric_dict = get_metric_values(data, task)
                        if not metric_dict:
                            print(f"Warning: Could not find metric for {metrics_file}")
                            continue
                        for metric_type, metric_value in metric_dict.items():
                            # For MNLI, create subtask name
                            if task == 'mnli' and metric_type in ['matched', 'mismatched']:
                                subtask = f'mnli_{metric_type}'
                            else:
                                subtask = task
                            results.append({
                                'variant': variant,
                                'task': subtask,
                                'model_family': model_family,
                                'seed': seed,
                                'batch_size': batch_size,
                                'teacher_lr': teacher_lr,
                                'weight_decay': weight_decay,
                                'peft': peft,
                                'lora_alpha': lora_alpha,
                                'lora_dropout': lora_dropout,
                                'rank': rank,
                                'metric': metric_value,
                                'file': str(metrics_file)
                            })
                    except Exception as e:
                        print(f"Error reading {metrics_file}: {e}")
    return pd.DataFrame(results)

def compute_median_top6(df):
    """Group by (task, model_family, seed, variant) and compute median of top 6 configs."""
    grouped = df.groupby(['task', 'model_family', 'seed', 'variant'])
    records = []
    for (task, model_family, seed, variant), group in grouped:
        # sort by metric descending (higher is better)
        sorted_group = group.sort_values('metric', ascending=False)
        # select top 6
        top6 = sorted_group.head(6)
        if len(top6) == 0:
            continue
        median_metric = top6['metric'].median()
        # also store the hyperparameters of the median config? maybe just median value
        records.append({
            'task': task,
            'model_family': model_family,
            'seed': seed,
            'variant': variant,
            'median_metric': median_metric,
            'num_configs': len(group),
            'top6_metrics': top6['metric'].tolist(),
            'top6_peft': top6['peft'].tolist(),
            'top6_rank': top6['rank'].tolist(),
            'top6_alpha': top6['lora_alpha'].tolist(),
        })
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description='Compute median of top 6 hyperparameter setups')
    parser.add_argument('--results_dir', default='./results', help='Path to results directory')
    parser.add_argument('--output', default='median_top6.csv', help='Output CSV file')
    args = parser.parse_args()
    
    print(f"Collecting results from {args.results_dir}...")
    df = collect_results(args.results_dir)
    if df.empty:
        print("No results found.")
        return
    print(f"Collected {len(df)} hyperparameter configurations.")
    
    print("Computing median of top 6 configurations per group...")
    median_df = compute_median_top6(df)
    
    # Save to CSV
    median_df.to_csv(args.output, index=False)
    print(f"Saved median results to {args.output}")
    
    # Print summary
    print("\nSummary:")
    print(median_df[['task', 'model_family', 'seed', 'variant', 'median_metric', 'num_configs']].to_string())
    
    # Optionally, also compute average across seeds?
    # For now, just output per seed.

if __name__ == '__main__':
    main()