#!/usr/bin/env python3
"""
Compute final efficiency table with teacher/student values.
"""
import json
import os
import statistics
from collections import defaultdict
import pandas as pd
import numpy as np

def scan_all_metrics():
    """Scan all JSON files and collect memory, runtime, and parameter counts."""
    base_dir = "../results"
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} does not exist")
    
    memory_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    runtime_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    param_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    count = 0
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
                seed = args.get("seed")
                variant = data.get("variant")
                if not all([model_family, method, variant]):
                    continue
                
                # Memory
                train_info = data.get("train", {})
                memory_list = train_info.get("memory_allocated", [])
                if memory_list and isinstance(memory_list, list):
                    avg_memory = sum(memory_list) / len(memory_list)
                    memory_data[variant][model_family][method].append(avg_memory)
                
                # Runtime
                runtime = data.get("eval_runtime")
                if runtime is not None:
                    runtime_data[variant][model_family][method].append(runtime)
                
                # Parameter count
                param_count = train_info.get("trainable_params_count")
                if param_count is not None:
                    param_data[variant][model_family][method].append(param_count)
                
                count += 1
    
    print(f"Processed {count} result files")
    return memory_data, runtime_data, param_data

def compute_averages(memory_data, runtime_data, param_data):
    """Compute averages across model families."""
    model_families = ["bert", "roberta", "deberta"]
    teacher_methods = ["lora", "mrlora", "adalora", "dora", "olora", "rslora", "mrlora-rs"]
    student_methods = ["lora", "mrlora"]  # in kd-lora variant
    
    # FFT baselines
    fft_memory = {}
    fft_runtime = {}
    fft_params = {}
    for mf in model_families:
        if mf in memory_data.get("fft", {}) and "lora" in memory_data["fft"][mf]:
            fft_memory[mf] = statistics.mean(memory_data["fft"][mf]["lora"])
        if mf in runtime_data.get("fft", {}) and "lora" in runtime_data["fft"][mf]:
            fft_runtime[mf] = statistics.mean(runtime_data["fft"][mf]["lora"])
        if mf in param_data.get("fft", {}) and "lora" in param_data["fft"][mf]:
            fft_params[mf] = statistics.mean(param_data["fft"][mf]["lora"])
    
    print("FFT baselines:")
    for mf in model_families:
        if mf in fft_memory:
            print(f"  {mf}: memory {fft_memory[mf]:.1f} MB, runtime {fft_runtime.get(mf, 'N/A')} s, params {fft_params.get(mf, 'N/A')} M")
    
    # Teacher fine‑tuning (variant 'lora')
    teacher_results = {}
    for method in teacher_methods:
        mem_reds = []
        speedups = []
        param_reds = []
        for mf in model_families:
            if mf in memory_data.get("lora", {}) and method in memory_data["lora"][mf]:
                mem_vals = memory_data["lora"][mf][method]
                if mem_vals and mf in fft_memory:
                    avg_mem = statistics.mean(mem_vals)
                    reduction = (fft_memory[mf] - avg_mem) / fft_memory[mf] * 100
                    mem_reds.append(reduction)
            
            if mf in runtime_data.get("lora", {}) and method in runtime_data["lora"][mf]:
                rt_vals = runtime_data["lora"][mf][method]
                if rt_vals and mf in fft_runtime:
                    avg_rt = statistics.mean(rt_vals)
                    speedup = fft_runtime[mf] / avg_rt
                    speedups.append(speedup)
            
            if mf in param_data.get("lora", {}) and method in param_data["lora"][mf]:
                param_vals = param_data["lora"][mf][method]
                if param_vals and mf in fft_params:
                    avg_param = statistics.mean(param_vals)
                    reduction = (fft_params[mf] - avg_param) / fft_params[mf] * 100
                    param_reds.append(reduction)
        
        teacher_results[method] = {
            "memory_reduction": statistics.mean(mem_reds) if mem_reds else None,
            "speedup": statistics.mean(speedups) if speedups else None,
            "param_reduction": statistics.mean(param_reds) if param_reds else None,
            "n_mem": len(mem_reds),
            "n_speed": len(speedups),
            "n_param": len(param_reds)
        }
    
    # Student distillation (variant 'kd-lora')
    student_results = {}
    for method in student_methods:
        mem_reds = []
        speedups = []
        param_reds = []
        for mf in model_families:
            if mf in memory_data.get("kd-lora", {}) and method in memory_data["kd-lora"][mf]:
                mem_vals = memory_data["kd-lora"][mf][method]
                if mem_vals and mf in fft_memory:
                    avg_mem = statistics.mean(mem_vals)
                    reduction = (fft_memory[mf] - avg_mem) / fft_memory[mf] * 100
                    mem_reds.append(reduction)
            
            if mf in runtime_data.get("kd-lora", {}) and method in runtime_data["kd-lora"][mf]:
                rt_vals = runtime_data["kd-lora"][mf][method]
                if rt_vals and mf in fft_runtime:
                    avg_rt = statistics.mean(rt_vals)
                    speedup = fft_runtime[mf] / avg_rt
                    speedups.append(speedup)
            
            if mf in param_data.get("kd-lora", {}) and method in param_data["kd-lora"][mf]:
                param_vals = param_data["kd-lora"][mf][method]
                if param_vals and mf in fft_params:
                    avg_param = statistics.mean(param_vals)
                    reduction = (fft_params[mf] - avg_param) / fft_params[mf] * 100
                    param_reds.append(reduction)
        
        student_results[method] = {
            "memory_reduction": statistics.mean(mem_reds) if mem_reds else None,
            "speedup": statistics.mean(speedups) if speedups else None,
            "param_reduction": statistics.mean(param_reds) if param_reds else None,
            "n_mem": len(mem_reds),
            "n_speed": len(speedups),
            "n_param": len(param_reds)
        }
    
    return teacher_results, student_results, fft_params

def format_value(val, fmt=".1f"):
    """Format a float or None."""
    if val is None:
        return "--"
    return f"{val:{fmt}}"

def main():
    memory_data, runtime_data, param_data = scan_all_metrics()
    teacher_res, student_res, fft_params = compute_averages(memory_data, runtime_data, param_data)
    
    method_names = {
        "lora": "LoRA",
        "mrlora": "MR‑LoRA",
        "adalora": "AdaLoRA",
        "dora": "DoRA",
        "olora": "OLoRA",
        "rslora": "RS‑LoRA",
        "mrlora-rs": "MR‑LoRA‑RS"
    }
    
    rows = []
    for method in method_names:
        t = teacher_res.get(method, {})
        s = student_res.get(method, {})
        
        # Parameter reduction
        if t.get("param_reduction") is not None:
            if s.get("param_reduction") is not None:
                param_str = f"{t['param_reduction']:.1f}%/{s['param_reduction']:.1f}%"
            else:
                param_str = f"{t['param_reduction']:.1f}%"
        else:
            param_str = "--"
        
        # Memory reduction
        if t.get("memory_reduction") is not None:
            if s.get("memory_reduction") is not None:
                mem_str = f"{t['memory_reduction']:.1f}%/{s['memory_reduction']:.1f}%"
            else:
                mem_str = f"{t['memory_reduction']:.1f}%"
        else:
            mem_str = "--"
        
        # Inference speedup
        if t.get("speedup") is not None:
            if s.get("speedup") is not None:
                speed_str = f"{t['speedup']:.1f}×/{s['speedup']:.1f}×"
            else:
                speed_str = f"{t['speedup']:.1f}×"
        else:
            speed_str = "--"
        
        rows.append({
            "Method": method_names[method],
            "Parameter Reduction": param_str,
            "Memory Reduction": mem_str,
            "Inference Speedup": speed_str,
            "Teacher Param": t.get("param_reduction"),
            "Student Param": s.get("param_reduction"),
            "Teacher Mem": t.get("memory_reduction"),
            "Student Mem": s.get("memory_reduction"),
            "Teacher Speed": t.get("speedup"),
            "Student Speed": s.get("speedup"),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv("final_efficiency_table.csv", index=False)
    print("\nFinal efficiency table:")
    print(df[["Method", "Parameter Reduction", "Memory Reduction", "Inference Speedup"]].to_string())
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Efficiency metrics for LoRA variants. Parameter reduction is relative to full fine‑tuning (FFT); each cell shows teacher fine‑tuning/student distillation values separated by a slash. Memory reduction and inference speedup are averaged across BERT, RoBERTa, and DeBERTa model families (only available for LoRA and MR‑LoRA).}")
    latex_lines.append("\\label{tab:efficiency}")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Method & Parameter Reduction & Memory Reduction & Inference Speedup \\\\")
    latex_lines.append("\\midrule")
    for _, row in df.iterrows():
        latex_lines.append(f"{row['Method']} & {row['Parameter Reduction']} & {row['Memory Reduction']} & {row['Inference Speedup']} \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    with open("final_efficiency_table.tex", "w") as f:
        f.write("\n".join(latex_lines))
    print("\nLaTeX table saved to final_efficiency_table.tex")
    
    # Compare with original table values
    print("\nComparison with original Table 4 values:")
    print("Original LoRA: 99.6%/99.5% | 74.3%/82.1% | 3.4×/4.2×")
    print("Our LoRA:      ", df.loc[df["Method"] == "LoRA", "Parameter Reduction"].values[0],
          "|", df.loc[df["Method"] == "LoRA", "Memory Reduction"].values[0],
          "|", df.loc[df["Method"] == "LoRA", "Inference Speedup"].values[0])
    print("Original MR‑LoRA: 99.4%/99.4% | 74.1%/82.0% | 2.9×/4.9×")
    print("Our MR‑LoRA:      ", df.loc[df["Method"] == "MR‑LoRA", "Parameter Reduction"].values[0],
          "|", df.loc[df["Method"] == "MR‑LoRA", "Memory Reduction"].values[0],
          "|", df.loc[df["Method"] == "MR‑LoRA", "Inference Speedup"].values[0])

if __name__ == "__main__":
    main()