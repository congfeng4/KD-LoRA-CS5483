#!/usr/bin/env python3
"""
Create per-task performance and efficiency tables for LoRA variants.
Aggregates data from JSON result files in results/lora and results/kd-lora.
Outputs CSV files and LaTeX tables for manuscript appendix.
"""
import json
import os
import statistics
from collections import defaultdict
import pandas as pd
import numpy as np

# Mapping from task name to metric key in JSON
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

def load_results(directory):
    """
    Walk through directory and load all metrics.json files.
    Returns nested dict: model_family -> method -> task -> list of metric values.
    Also returns efficiency dict: model_family -> method -> list of dicts with params, memory, runtime.
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    efficiency = defaultdict(lambda: defaultdict(list))
    
    count = 0
    for root, dirs, files in os.walk(directory):
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
                if not all([model_family, task, method]):
                    continue
                
                # Determine variant type: teacher (type=2) or student (type=1?) or FFT (type=0)
                # We'll rely on directory structure: lora/ for teacher, kd-lora/ for student
                variant_type = "teacher" if "lora" in directory and "kd-lora" not in directory else "student"
                # Override with type field if present
                if "type" in args:
                    if args["type"] == 0:
                        variant_type = "fft"
                    elif args["type"] == 1:
                        variant_type = "student"
                    elif args["type"] == 2:
                        variant_type = "teacher"
                
                # Get task metric
                metric_val = get_task_metric(data, task)
                if metric_val is None:
                    continue
                
                # Convert to percentage (0‑100) as in GLUE scores
                metric_val *= 100
                
                # Store result
                key = f"{method}_{variant_type}" if variant_type != "teacher" else method
                results[model_family][key][task].append(metric_val)
                
                # Extract efficiency metrics
                train_info = data.get("train", {})
                param_count = train_info.get("trainable_params_count")  # in millions
                memory = train_info.get("memory_allocated")  # list per epoch
                runtime = data.get("eval_runtime")
                
                if param_count is not None or memory is not None or runtime is not None:
                    efficiency[model_family][key].append({
                        "params": param_count,
                        "memory": memory[0] if memory else None,  # first epoch
                        "runtime": runtime,
                        "seed": seed,
                        "task": task
                    })
                
                count += 1
    
    print(f"Loaded {count} result files from {directory}")
    return results, efficiency

def main():
    # Load teacher results (lora directory)
    teacher_results, teacher_efficiency = load_results("../results/lora")
    # Load student results (kd-lora directory)
    student_results, student_efficiency = load_results("../results/kd-lora")
    # Load FFT baselines (fft directory)
    fft_results, fft_efficiency = load_results("../results/fft")
    
    # Combine results
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for model_family in teacher_results:
        for method in teacher_results[model_family]:
            for task in teacher_results[model_family][method]:
                all_results[model_family][method][task].extend(teacher_results[model_family][method][task])
    for model_family in student_results:
        for method in student_results[model_family]:
            for task in student_results[model_family][method]:
                all_results[model_family][method][task].extend(student_results[model_family][method][task])
    for model_family in fft_results:
        for method in fft_results[model_family]:
            for task in fft_results[model_family][method]:
                all_results[model_family][method][task].extend(fft_results[model_family][method][task])
    
    # Combine efficiency data
    all_efficiency = defaultdict(lambda: defaultdict(list))
    for model_family in teacher_efficiency:
        for method in teacher_efficiency[model_family]:
            all_efficiency[model_family][method].extend(teacher_efficiency[model_family][method])
    for model_family in student_efficiency:
        for method in student_efficiency[model_family]:
            all_efficiency[model_family][method].extend(student_efficiency[model_family][method])
    for model_family in fft_efficiency:
        for method in fft_efficiency[model_family]:
            all_efficiency[model_family][method].extend(fft_efficiency[model_family][method])
    
    # List of tasks in GLUE order
    tasks = ["cola", "sst2", "mrpc", "qqp", "stsb", "qnli", "rte", "wnli", "mnli"]
    task_display = {
        "cola": "CoLA", "sst2": "SST‑2", "mrpc": "MRPC", "qqp": "QQP",
        "stsb": "STS‑B", "qnli": "QNLI", "rte": "RTE", "wnli": "WNLI",
        "mnli": "MNLI"
    }
    
    # Methods we care about (teacher variants)
    teacher_methods = ["lora", "mrlora", "adalora", "dora", "olora", "rslora", "mrlora-rs"]
    student_methods = ["lora_student", "mrlora_student"]  # from student directory
    
    # First, compute per-task averages for each model family and method
    per_task_avg = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for model_family in all_results:
        for method in all_results[model_family]:
            for task in all_results[model_family][method]:
                values = all_results[model_family][method][task]
                if values:
                    per_task_avg[model_family][method][task] = statistics.mean(values)
    
    # Compute GLUE average (mean across tasks) for each model family and method
    glue_avg = defaultdict(dict)
    for model_family in per_task_avg:
        for method in per_task_avg[model_family]:
            task_values = [per_task_avg[model_family][method][t] for t in tasks if t in per_task_avg[model_family][method]]
            if task_values:
                glue_avg[model_family][method] = statistics.mean(task_values)
    
    # Write per-task CSV for teacher variants
    with open("teacher_per_task.csv", "w") as f:
        f.write("Model Family,Method," + ",".join([task_display[t] for t in tasks]) + ",GLUE\n")
        for model_family in ["bert", "roberta", "deberta"]:
            for method in teacher_methods:
                if method not in glue_avg[model_family]:
                    continue
                row = [model_family, method]
                for task in tasks:
                    val = per_task_avg[model_family][method].get(task, "")
                    row.append(f"{val:.2f}" if val != "" else "")
                row.append(f"{glue_avg[model_family][method]:.2f}")
                f.write(",".join(row) + "\n")
    
    # Write per-task CSV for student variants
    with open("student_per_task.csv", "w") as f:
        f.write("Model Family,Method," + ",".join([task_display[t] for t in tasks]) + ",GLUE\n")
        for model_family in ["bert", "roberta", "deberta"]:
            for method in student_methods:
                if method not in glue_avg[model_family]:
                    continue
                row = [model_family, method]
                for task in tasks:
                    val = per_task_avg[model_family][method].get(task, "")
                    row.append(f"{val:.2f}" if val != "" else "")
                row.append(f"{glue_avg[model_family][method]:.2f}")
                f.write(",".join(row) + "\n")
    
    # Compute efficiency metrics
    # Need FFT baselines for each model family
    fft_param = {}
    fft_memory = {}
    fft_runtime = {}
    for model_family in ["bert", "roberta", "deberta"]:
        fft_key = "lora"  # FFT results stored under "lora" method
        if fft_key in all_efficiency[model_family]:
            params = [e["params"] for e in all_efficiency[model_family][fft_key] if e["params"] is not None]
            memory = [e["memory"] for e in all_efficiency[model_family][fft_key] if e["memory"] is not None]
            runtime = [e["runtime"] for e in all_efficiency[model_family][fft_key] if e["runtime"] is not None]
            if params:
                fft_param[model_family] = statistics.mean(params)
            if memory:
                fft_memory[model_family] = statistics.mean(memory)
            if runtime:
                fft_runtime[model_family] = statistics.mean(runtime)
    
    # Compute efficiency metrics for each method
    efficiency_data = []
    for model_family in ["bert", "roberta", "deberta"]:
        for method in teacher_methods + student_methods:
            if method not in all_efficiency[model_family]:
                continue
            entries = all_efficiency[model_family][method]
            params = [e["params"] for e in entries if e["params"] is not None]
            memory = [e["memory"] for e in entries if e["memory"] is not None]
            runtime = [e["runtime"] for e in entries if e["runtime"] is not None]
            
            param_avg = statistics.mean(params) if params else None
            memory_avg = statistics.mean(memory) if memory else None
            runtime_avg = statistics.mean(runtime) if runtime else None
            
            # Compute percentages relative to FFT
            param_pct = (param_avg / fft_param[model_family] * 100) if param_avg and model_family in fft_param else None
            memory_red = ((fft_memory[model_family] - memory_avg) / fft_memory[model_family] * 100) if memory_avg and model_family in fft_memory else None
            speedup = fft_runtime[model_family] / runtime_avg if runtime_avg and model_family in fft_runtime else None
            
            efficiency_data.append({
                "Model Family": model_family,
                "Method": method,
                "Params (%FFT)": f"{param_pct:.3f}" if param_pct else "",
                "Memory Reduction (%)": f"{memory_red:.1f}" if memory_red else "",
                "Inference Speedup": f"{speedup:.1f}×" if speedup else ""
            })
    
    # Write efficiency CSV
    eff_df = pd.DataFrame(efficiency_data)
    eff_df.to_csv("efficiency_metrics.csv", index=False)
    
    # Generate LaTeX tables
    generate_latex_tables(per_task_avg, glue_avg, efficiency_data)
    
    print("\nGenerated files:")
    print("  - teacher_per_task.csv")
    print("  - student_per_task.csv")
    print("  - efficiency_metrics.csv")
    print("  - teacher_per_task.tex")
    print("  - student_per_task.tex")
    print("  - efficiency_table.tex")

def generate_latex_tables(per_task_avg, glue_avg, efficiency_data):
    """Generate LaTeX tables for per-task performance and efficiency."""
    tasks = ["cola", "sst2", "mrpc", "qqp", "stsb", "qnli", "rte", "wnli", "mnli"]
    task_display = ["CoLA", "SST‑2", "MRPC", "QQP", "STS‑B", "QNLI", "RTE", "WNLI", "MNLI"]
    
    # Teacher table
    teacher_methods = ["lora", "mrlora", "adalora", "dora", "olora", "rslora", "mrlora-rs"]
    method_names = {
        "lora": "LoRA", "mrlora": "MR‑LoRA", "adalora": "AdaLoRA",
        "dora": "DoRA", "olora": "OLoRA", "rslora": "RS‑LoRA",
        "mrlora-rs": "MR‑LoRA‑RS"
    }
    
    # Create teacher table (averaged across model families)
    teacher_table = []
    for method in teacher_methods:
        row = {"Method": method_names[method]}
        # Collect per-task averages across model families
        task_values = []
        for task in tasks:
            vals = []
            for model_family in ["bert", "roberta", "deberta"]:
                if method in per_task_avg[model_family] and task in per_task_avg[model_family][method]:
                    vals.append(per_task_avg[model_family][method][task])
            if vals:
                avg = statistics.mean(vals)
                row[task] = f"{avg:.1f}"
                task_values.append(avg)
            else:
                row[task] = "--"
        # GLUE average
        if task_values:
            row["GLUE"] = f"{statistics.mean(task_values):.1f}"
        else:
            row["GLUE"] = "--"
        teacher_table.append(row)
    
    # Write teacher LaTeX table
    with open("teacher_per_task.tex", "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Per‑task GLUE scores for teacher LoRA variants, averaged across BERT, RoBERTa, and DeBERTa model families.}\n")
        f.write("\\label{tab:teacher-per-task}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{l" + "c" * (len(tasks) + 1) + "}\n")
        f.write("\\toprule\n")
        f.write("Method & " + " & ".join(task_display) + " & GLUE \\\\\n")
        f.write("\\midrule\n")
        for row in teacher_table:
            line = row["Method"]
            for task in tasks:
                line += " & " + row[task]
            line += " & " + row["GLUE"] + " \\\\\n"
            f.write(line)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # Student table (only LoRA and MR‑LoRA)
    student_methods = ["lora_student", "mrlora_student"]
    student_names = {"lora_student": "LoRA", "mrlora_student": "MR‑LoRA"}
    
    student_table = []
    for method in student_methods:
        row = {"Method": student_names[method]}
        task_values = []
        for task in tasks:
            vals = []
            for model_family in ["bert", "roberta", "deberta"]:
                if method in per_task_avg[model_family] and task in per_task_avg[model_family][method]:
                    vals.append(per_task_avg[model_family][method][task])
            if vals:
                avg = statistics.mean(vals)
                row[task] = f"{avg:.1f}"
                task_values.append(avg)
            else:
                row[task] = "--"
        if task_values:
            row["GLUE"] = f"{statistics.mean(task_values):.1f}"
        else:
            row["GLUE"] = "--"
        student_table.append(row)
    
    with open("student_per_task.tex", "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Per‑task GLUE scores for student LoRA variants (knowledge distillation), averaged across BERT, RoBERTa, and DeBERTa model families.}\n")
        f.write("\\label{tab:student-per-task}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{l" + "c" * (len(tasks) + 1) + "}\n")
        f.write("\\toprule\n")
        f.write("Method & " + " & ".join(task_display) + " & GLUE \\\\\n")
        f.write("\\midrule\n")
        for row in student_table:
            line = row["Method"]
            for task in tasks:
                line += " & " + row[task]
            line += " & " + row["GLUE"] + " \\\\\n"
            f.write(line)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # Efficiency table (average across model families)
    eff_df = pd.DataFrame(efficiency_data)
    # Group by method, average across model families
    eff_avg = eff_df.groupby("Method").agg({
        "Params (%FFT)": lambda x: statistics.mean([float(v) for v in x if v]),
        "Memory Reduction (%)": lambda x: statistics.mean([float(v) for v in x if v]),
        "Inference Speedup": lambda x: statistics.mean([float(v.rstrip('×')) for v in x if v])
    }).reset_index()
    
    with open("efficiency_table.tex", "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Efficiency metrics for LoRA variants, averaged across BERT, RoBERTa, and DeBERTa model families.}\n")
        f.write("\\label{tab:efficiency-metrics}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Parameters (\\% FFT) & Memory Reduction (\\%) & Inference Speedup \\\\\n")
        f.write("\\midrule\n")
        for _, row in eff_avg.iterrows():
            method = row["Method"]
            # Clean method name
            if method.endswith("_student"):
                method = method[:-8] + " (Student)"
            f.write(f"{method} & {row['Params (%FFT)']:.3f} & {row['Memory Reduction (%)']:.1f} & {row['Inference Speedup']:.1f}$\\times$ \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

if __name__ == "__main__":
    main()