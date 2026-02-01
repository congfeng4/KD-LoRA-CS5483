#!/usr/bin/env python3
"""
Aggregate metrics from results/lora directory.
Compute mean of primary metric across seeds per (model_family, task, method).
"""

import json
import os
import re
from collections import defaultdict
import pandas as pd
import numpy as np

RESULTS_ROOT = "results"

# Mapping from task to metric keys (list)
TASK_METRIC = {
    "cola": ["eval_matthews_correlation"],
    "mnli": ["matched_accuracy", "mismatched_accuracy"],
    "mrpc": ["eval_accuracy", "eval_f1"],
    "qnli": ["eval_accuracy"],
    "qqp": ["eval_accuracy", "eval_f1"],
    "rte": ["eval_accuracy"],
    "sst2": ["eval_accuracy"],
    "stsb": ["eval_pearson", "eval_spearman"],
    "wnli": ["eval_accuracy"],
}

# Primary metric for each task (used for average)
TASK_PRIMARY_METRIC = {
    "cola": "eval_matthews_correlation",
    "mnli": "matched_accuracy",
    "mrpc": "eval_f1",
    "qnli": "eval_accuracy",
    "qqp": "eval_f1",
    "rte": "eval_accuracy",
    "sst2": "eval_accuracy",
    "stsb": "eval_spearman",
    "wnli": "eval_accuracy",
}

# Display abbreviation for metric row
METRIC_DISPLAY = {
    "eval_matthews_correlation": "Mcc",
    "matched_accuracy": "m",
    "mismatched_accuracy": "mm",
    "eval_accuracy": "Acc",
    "eval_f1": "F1",
    "eval_pearson": "Pearson",
    "eval_spearman": "Spearman",
}

def extract_info_from_path(path):
    """
    Extract training_variant, task, model_family, seed, method from path.
    Path pattern: results/{training_variant}/task_{task}_{family}_{seed}/base_*/peft_{method}_{alpha}_{dropout}_{rank}/metrics.json
    """
    # Normalize path
    rel_path = os.path.relpath(path, RESULTS_ROOT)
    parts = rel_path.split(os.sep)
    if len(parts) < 4:
        return None
    training_variant = parts[0]  # fft, lora, kd-lora, legacy/...
    task_dir = parts[1]  # e.g., task_cola_deberta_2024
    method_dir = parts[3]  # e.g., peft_lora_16_0.05_8
    
    # Parse task_dir
    task_match = re.match(r'task_([a-z0-9]+)_([a-z]+[a-z0-9\-]*)_(\d+)', task_dir)
    if not task_match:
        return None
    task, model_family, seed = task_match.groups()
    seed = int(seed)
    
    # Parse method_dir
    if not method_dir.startswith('peft_'):
        return None
    method = method_dir.split('_')[1]  # second part after peft_
    # Handle hyphenated methods like mrlora-rs
    return {
        'training_variant': training_variant,
        'task': task,
        'model_family': model_family,
        'seed': seed,
        'method': method,
        'path': path
    }

def collect_data():
    """Walk through results directory and collect metrics."""
    data = []  # list of dicts
    missing_metric = defaultdict(int)
    
    for root, dirs, files in os.walk(RESULTS_ROOT):
        if 'metrics.json' in files:
            path = os.path.join(root, 'metrics.json')
            info = extract_info_from_path(path)
            if not info:
                # Skip legacy paths etc.
                continue
            training_variant = info['training_variant']
            # Only process fft, lora, kd-lora (skip legacy, test_adalora, etc.)
            if training_variant not in ['fft', 'lora', 'kd-lora']:
                continue
            try:
                with open(path, 'r') as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue
            
            task = info['task']
            metric_keys = TASK_METRIC.get(task)
            if not metric_keys:
                missing_metric[task] += 1
                continue
            
            # Override method for fft variant
            method = info['method']
            if training_variant == 'fft':
                method = 'fft'
            
            for metric_key in metric_keys:
                if metric_key not in metrics:
                    # For MNLI, matched_accuracy and mismatched_accuracy are at top level
                    if task == 'mnli' and metric_key in metrics:
                        value = metrics[metric_key]
                    else:
                        missing_metric[metric_key] += 1
                        continue
                else:
                    value = metrics[metric_key]
                
                # Also get trainable params count if needed
                params = metrics.get('train', {}).get('trainable_params_count', None)
                
                data.append({
                    'training_variant': training_variant,
                    'task': task,
                    'model_family': info['model_family'],
                    'seed': info['seed'],
                    'method': method,
                    'metric_name': metric_key,
                    'metric_value': value,
                    'trainable_params': params,
                })
    
    if missing_metric:
        print("Missing metric counts:", dict(missing_metric))
    
    return data

def compute_metric_means(data):
    """Compute mean metric across seeds for each (training_variant, model_family, task, method, metric_name)."""
    grouped = defaultdict(list)
    for entry in data:
        key = (entry['training_variant'], entry['model_family'], entry['task'], entry['method'], entry['metric_name'])
        grouped[key].append(entry['metric_value'])
    
    means = {}
    for key, values in grouped.items():
        # Multiply by 100 as requested
        means[key] = np.mean(values) * 100
    
    return means

def compute_param_means(data):
    """Compute mean trainable_params across seeds for each (training_variant, model_family, method)."""
    grouped = defaultdict(list)
    for entry in data:
        if entry['trainable_params'] is not None:
            key = (entry['training_variant'], entry['model_family'], entry['method'])
            grouped[key].append(entry['trainable_params'])
    
    means = {}
    for key, values in grouped.items():
        means[key] = np.mean(values)  # Already in millions? Keep as is.
    
    return means

def create_metric_df(metric_means):
    """Convert metric means dict into pandas DataFrame."""
    rows = []
    for (training_variant, model_family, task, method, metric_name), mean_val in metric_means.items():
        rows.append({
            'training_variant': training_variant,
            'model_family': model_family,
            'task': task,
            'method': method,
            'metric_name': metric_name,
            'mean_value': mean_val,
        })
    df = pd.DataFrame(rows)
    return df

def create_param_df(param_means):
    """Convert param means dict into pandas DataFrame."""
    rows = []
    for (training_variant, model_family, method), mean_val in param_means.items():
        rows.append({
            'training_variant': training_variant,
            'model_family': model_family,
            'method': method,
            'mean_params': mean_val,
        })
    df = pd.DataFrame(rows)
    return df

def build_table_data(metric_df, param_df, training_variant, model_family):
    """
    Build a structured table for given training_variant and model_family.
    Returns a dict with:
    - rows: list of method strings (including 'fft' first if present)
    - param_dict: method -> param count
    - metric_dict: method -> task -> list of (metric_name, value)
    - primary_dict: method -> task -> primary metric value (for average)
    """
    # Filter data
    metric_subset = metric_df[(metric_df['training_variant'] == training_variant) & 
                              (metric_df['model_family'] == model_family)].copy()
    param_subset = param_df[(param_df['training_variant'] == training_variant) & 
                            (param_df['model_family'] == model_family)].copy()
    
    # Get FFT data for this model_family (training_variant='fft')
    fft_metric = metric_df[(metric_df['training_variant'] == 'fft') & 
                           (metric_df['model_family'] == model_family) &
                           (metric_df['method'] == 'fft')].copy()
    fft_param = param_df[(param_df['training_variant'] == 'fft') & 
                         (param_df['model_family'] == model_family) &
                         (param_df['method'] == 'fft')].copy()
    
    # Determine methods order
    method_order = ['fft', 'lora', 'dora', 'mrlora', 'mrlora-rs', 'olora', 'rslora']
    # Keep only methods present in metric_subset or fft_metric
    present_methods = set(metric_subset['method'].unique())
    if not fft_metric.empty:
        present_methods.add('fft')
    # Order preserving
    rows = [m for m in method_order if m in present_methods]
    
    # Build param dict
    param_dict = {}
    for _, row in param_subset.iterrows():
        param_dict[row['method']] = row['mean_params']
    # Add FFT param if available
    if not fft_param.empty:
        param_dict['fft'] = fft_param.iloc[0]['mean_params']
    
    # Build metric dict: method -> task -> list of (metric_name, value)
    metric_dict = defaultdict(lambda: defaultdict(list))
    for _, row in metric_subset.iterrows():
        metric_dict[row['method']][row['task']].append((row['metric_name'], row['mean_value']))
    # Add FFT metrics
    for _, row in fft_metric.iterrows():
        metric_dict['fft'][row['task']].append((row['metric_name'], row['mean_value']))
    
    # Build primary dict: method -> task -> primary metric value
    primary_dict = defaultdict(dict)
    for method in rows:
        for task, metrics in metric_dict[method].items():
            primary_metric = TASK_PRIMARY_METRIC.get(task)
            # Find value for primary metric
            for metric_name, value in metrics:
                if metric_name == primary_metric:
                    primary_dict[method][task] = value
                    break
            else:
                # If primary metric not found, take first metric
                if metrics:
                    primary_dict[method][task] = metrics[0][1]
    
    return {
        'rows': rows,
        'param_dict': param_dict,
        'metric_dict': metric_dict,
        'primary_dict': primary_dict,
    }

def compute_average(primary_dict, method, task_list):
    """Compute average of primary metric values across tasks."""
    values = []
    for task in task_list:
        if task in primary_dict.get(method, {}):
            values.append(primary_dict[method][task])
    if values:
        return np.mean(values)
    else:
        return np.nan

def generate_table_csv(table_data, output_path):
    """Generate CSV representation of table."""
    rows = table_data['rows']
    param_dict = table_data['param_dict']
    metric_dict = table_data['metric_dict']
    primary_dict = table_data['primary_dict']
    
    # Define task order and metric display for header
    task_order = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
    # Build header: Method, # Params, tasks..., Average
    header = ['Method', '# Params'] + task_order + ['Average']
    
    # Build rows
    csv_rows = []
    for method in rows:
        row = [method, param_dict.get(method, '')]
        # For each task, format metric values
        for task in task_order:
            metrics = metric_dict[method].get(task, [])
            if not metrics:
                row.append('')
            else:
                # Sort metrics according to TASK_METRIC order
                metric_order = TASK_METRIC.get(task, [])
                sorted_metrics = []
                for metric_name in metric_order:
                    for mname, val in metrics:
                        if mname == metric_name:
                            sorted_metrics.append(f"{val:.2f}")
                            break
                # If we have two metrics, join with '/'
                cell = '/'.join(sorted_metrics)
                row.append(cell)
        # Average column
        avg = compute_average(primary_dict, method, task_order)
        row.append(f"{avg:.2f}" if not pd.isna(avg) else '')
        csv_rows.append(row)
    
    # Write CSV
    import csv
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)
    print(f"Saved table CSV to {output_path}")

def main():
    print("Collecting data...")
    data = collect_data()
    print(f"Collected {len(data)} data points.")
    
    print("Computing metric means...")
    metric_means = compute_metric_means(data)
    print(f"Computed metric means for {len(metric_means)} groups.")
    
    print("Computing param means...")
    param_means = compute_param_means(data)
    print(f"Computed param means for {len(param_means)} groups.")
    
    metric_df = create_metric_df(metric_means)
    param_df = create_param_df(param_means)
    
    # Save raw data
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    metric_df.to_csv(os.path.join(output_dir, "metric_means.csv"), index=False)
    param_df.to_csv(os.path.join(output_dir, "param_means.csv"), index=False)
    print(f"Saved raw data to {output_dir}/")
    
    # Generate tables for each training variant and model family
    training_variants = ['lora', 'kd-lora']  # tables for these variants
    families = metric_df['model_family'].unique()
    
    for tv in training_variants:
        for fam in families:
            table_data = build_table_data(metric_df, param_df, tv, fam)
            if not table_data['rows']:
                continue
            # Generate CSV
            csv_path = os.path.join(output_dir, f"table_{tv}_{fam}.csv")
            generate_table_csv(table_data, csv_path)
            # Also generate LaTeX? We'll have separate script.
    
    print("\nDone. Table CSVs generated in output/.")

if __name__ == "__main__":
    main()