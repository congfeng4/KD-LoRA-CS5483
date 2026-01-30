import json
import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np

results_root = "results"
# mapping from task to primary metric key
task_metric = {
    "cola": "eval_matthews_correlation",
    "mnli": "matched_accuracy",
    "mrpc": "eval_accuracy",  # could also use eval_f1
    "qnli": "eval_accuracy",
    "qqp": "eval_accuracy",   # could use eval_f1
    "rte": "eval_accuracy",
    "sst2": "eval_accuracy",
    "stsb": "eval_pearson",
    "wnli": "eval_accuracy",
}

def extract_efficiency_metrics(filepath):
    """Extract efficiency metrics from a metrics.json file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # parse path components
    parts = Path(filepath).parts
    # variant is the directory under results (fft, lora, kd-lora)
    variant = parts[parts.index("results") + 1]
    # task directory name like task_cola_bert_42
    task_dir = parts[parts.index("results") + 2]
    # extract task, model, seed
    _, task, model, seed = task_dir.split('_')
    # peft directory name like peft_lora_16_0.05_8
    peft_dir = parts[-2]
    peft = peft_dir.split('_')[1]  # e.g., lora, mrlora
    
    # metric selection
    metric_key = task_metric.get(task)
    if metric_key is None:
        # fallback: find any key starting with eval_ (excluding eval_loss, eval_runtime, etc.)
        for key in data.keys():
            if key.startswith("eval_") and key not in ("eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"):
                metric_key = key
                break
        if metric_key is None:
            # try matched_accuracy or mismatched_accuracy
            if "matched_accuracy" in data:
                metric_key = "matched_accuracy"
            elif "mismatched_accuracy" in data:
                metric_key = "mismatched_accuracy"
            else:
                metric_key = ""
    metric_value = data.get(metric_key, "")
    
    # Efficiency metrics from train dict
    train_dict = data.get("train", {})
    train_time = train_dict.get("train_time", "")
    trainable_params = train_dict.get("trainable_params_count", "")
    
    # Memory metrics
    memory_allocated = train_dict.get("memory_allocated", [])
    memory_reserved = train_dict.get("memory_reserved", [])
    
    # Compute average memory usage if available
    avg_memory_allocated = np.mean(memory_allocated) if memory_allocated else ""
    avg_memory_reserved = np.mean(memory_reserved) if memory_reserved else ""
    max_memory_allocated = max(memory_allocated) if memory_allocated else ""
    max_memory_reserved = max(memory_reserved) if memory_reserved else ""
    
    # Throughput and FLOPs from log_history
    log_history = data.get("log_history", [])
    train_runtime = ""
    train_samples_per_second = ""
    train_steps_per_second = ""
    total_flos = ""
    
    for entry in log_history:
        if "train_runtime" in entry:
            train_runtime = entry.get("train_runtime", "")
            train_samples_per_second = entry.get("train_samples_per_second", "")
            train_steps_per_second = entry.get("train_steps_per_second", "")
            total_flos = entry.get("total_flos", "")
            break
    
    # If train_time not in train dict, use train_runtime
    if train_time == "" and train_runtime != "":
        train_time = train_runtime
    
    # Args
    args = data.get("args", {})
    rank = args.get("rank", "")
    lora_alpha = args.get("lora_alpha", "")
    lora_dropout = args.get("lora_dropout", "")
    lora_ranks = args.get("lora_ranks", "")
    peft_type = args.get("peft", "")
    type_num = args.get("type", "")
    model_family = args.get("model_family", "")
    
    return {
        "variant": variant,
        "task": task,
        "model": model,
        "seed": seed,
        "peft": peft,
        "peft_type": peft_type,
        "type": type_num,
        "metric_key": metric_key,
        "metric_value": metric_value,
        "trainable_params_count": trainable_params,
        "train_time": train_time,
        "avg_memory_allocated_mb": avg_memory_allocated,
        "max_memory_allocated_mb": max_memory_allocated,
        "avg_memory_reserved_mb": avg_memory_reserved,
        "max_memory_reserved_mb": max_memory_reserved,
        "train_samples_per_second": train_samples_per_second,
        "train_steps_per_second": train_steps_per_second,
        "total_flos": total_flos,
        "rank": rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_ranks": lora_ranks,
        "model_family": model_family,
        "file": filepath,
    }

def extract_all_metrics():
    """Extract metrics from all JSON files for all model families."""
    # Pattern to match all metrics.json files
    pattern = os.path.join(results_root, "*", "task_*_*_*", "base_*", "peft_*", "metrics.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} metrics.json files")
    
    all_metrics = []
    for i, filepath in enumerate(sorted(files)):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(files)}...")
        try:
            metrics = extract_efficiency_metrics(filepath)
            all_metrics.append(metrics)
        except Exception as e:
            sys.stderr.write(f"Error processing {filepath}: {e}\n")
            continue
    
    return all_metrics

def main():
    """Main function to extract and save efficiency metrics."""
    print("Extracting efficiency metrics from all JSON files...")
    all_metrics = extract_all_metrics()
    
    if not all_metrics:
        print("No metrics extracted!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Save to CSV
    output_file = "efficiency_metrics_all.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")
    
    # Also create a summary file for each model family
    for model_family in df['model_family'].unique():
        if model_family:
            df_model = df[df['model_family'] == model_family]
            model_file = f"efficiency_metrics_{model_family}.csv"
            df_model.to_csv(model_file, index=False)
            print(f"Saved {len(df_model)} rows for {model_family} to {model_file}")
    
    # Basic statistics
    print("\n=== Efficiency Metrics Summary ===")
    print(f"Total experiments: {len(df)}")
    print(f"Model families: {df['model_family'].unique()}")
    print(f"Variants: {df['variant'].unique()}")
    print(f"PEFT methods: {df['peft'].unique()}")
    
    # Check for missing efficiency metrics
    efficiency_cols = ['train_time', 'trainable_params_count', 'avg_memory_allocated_mb', 'total_flos']
    for col in efficiency_cols:
        missing = df[col].isna().sum()
        total = len(df)
        print(f"{col}: {missing}/{total} missing ({missing/total*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    df = main()