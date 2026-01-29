#!/usr/bin/env python3
import json
import os
from collections import defaultdict
import statistics

def get_task_metric(data, task):
    """Return the appropriate metric value for a GLUE task."""
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
    elif task in ["mnli", "qnli", "rte", "wnli"]:
        return data.get("eval_accuracy")
    else:
        return None

def main():
    base_dir = "../results/kd-lora"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return
    
    # nested dict: model_family -> method -> task -> list of values across seeds
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "metrics.json":
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue
                
                args = data.get("args", {})
                model_family = args.get("model_family")
                task = args.get("task")
                method = args.get("peft")
                seed = args.get("seed")
                if not all([model_family, task, method, seed]):
                    continue
                
                metric_val = get_task_metric(data, task)
                if metric_val is None:
                    continue
                
                # Convert to percentage (0-100) as in GLUE scores
                metric_val *= 100
                
                results[model_family][method][task].append(metric_val)
                count += 1
    
    print(f"Processed {count} result files.")
    
    # Compute GLUE average per model family and method
    glue_averages = defaultdict(dict)
    for model_family in results:
        for method in results[model_family]:
            task_values = []
            for task, values in results[model_family][method].items():
                # Average across seeds for this task
                if values:
                    task_avg = statistics.mean(values)
                    task_values.append((task, task_avg))
            # Compute GLUE average (simple mean across tasks)
            if task_values:
                glue_avg = statistics.mean([v for _, v in task_values])
                glue_averages[model_family][method] = glue_avg
                print(f"{model_family} - {method}: GLUE = {glue_avg:.2f}")
                # Print per-task averages
                for task, val in sorted(task_values):
                    print(f"  {task}: {val:.2f}")
    
    # Write to CSV for easier inspection
    import csv
    with open("glue_averages.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model_family", "method", "glue_score"])
        for model_family in glue_averages:
            for method in glue_averages[model_family]:
                writer.writerow([model_family, method, f"{glue_averages[model_family][method]:.2f}"])
    print("\nWritten to glue_averages.csv")
    
    # Also compute efficiency metrics (parameter reduction, memory reduction, inference speedup)
    # Need to extract from trainable_params_count, memory_allocated, eval_runtime
    efficiency = defaultdict(lambda: defaultdict(list))  # model_family -> method -> list of dicts
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "metrics.json":
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except Exception:
                    continue
                
                args = data.get("args", {})
                model_family = args.get("model_family")
                method = args.get("peft")
                if not model_family or not method:
                    continue
                
                train_info = data.get("train", {})
                param_count = train_info.get("trainable_params_count")  # in millions?
                memory = train_info.get("memory_allocated")
                runtime = data.get("eval_runtime")
                
                # TODO: need baseline FFT numbers for parameter reduction
                # For now, compute average param count across runs
                if param_count is not None:
                    efficiency[model_family][method].append({
                        "params": param_count,
                        "memory": memory,
                        "runtime": runtime
                    })
    
    print("\nEfficiency metrics (average trainable parameters in millions):")
    for model_family in efficiency:
        for method in efficiency[model_family]:
            params = [e["params"] for e in efficiency[model_family][method] if e["params"] is not None]
            if params:
                avg_params = statistics.mean(params)
                print(f"{model_family} - {method}: {avg_params:.3f}M trainable params")

if __name__ == "__main__":
    main()