#!/usr/bin/env python3
"""
Compute memory reduction and inference speedup for all variants relative to FFT.
"""
import json
import os
import statistics
from collections import defaultdict
import pandas as pd
import numpy as np

def scan_results():
    base_dir = "../results"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return {}
    
    # dict: variant -> model_family -> method -> list of memory values
    memory_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    runtime_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "metrics.json":
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    continue
                
                args = data.get("args", {})
                model_family = args.get("model_family")
                method = args.get("peft")
                seed = args.get("seed")
                variant = data.get("variant")
                if not all([model_family, method, variant]):
                    continue
                
                # Extract memory allocated (list)
                train_info = data.get("train", {})
                memory_list = train_info.get("memory_allocated", [])
                if memory_list and isinstance(memory_list, list):
                    # Use average memory across training steps
                    avg_memory = sum(memory_list) / len(memory_list)
                    memory_data[variant][model_family][method].append(avg_memory)
                
                # Extract inference runtime
                runtime = data.get("eval_runtime")
                if runtime is not None:
                    runtime_data[variant][model_family][method].append(runtime)
                
                count += 1
    
    print(f"Processed {count} result files")
    return memory_data, runtime_data

def compute_averages(memory_data, runtime_data):
    # Model families
    model_families = ["bert", "roberta", "deberta"]
    
    # Methods of interest (teacher variants)
    methods = ["lora", "mrlora", "adalora", "dora", "olora", "rslora", "mrlora-rs"]
    
    # Get FFT baselines
    fft_memory = {}
    fft_runtime = {}
    for model_family in model_families:
        if model_family in memory_data.get("fft", {}) and "lora" in memory_data["fft"][model_family]:
            fft_memory[model_family] = statistics.mean(memory_data["fft"][model_family]["lora"])
        if model_family in runtime_data.get("fft", {}) and "lora" in runtime_data["fft"][model_family]:
            fft_runtime[model_family] = statistics.mean(runtime_data["fft"][model_family]["lora"])
    
    print("FFT baselines:")
    for mf in model_families:
        if mf in fft_memory:
            print(f"  {mf}: memory {fft_memory[mf]:.1f} MB, runtime {fft_runtime.get(mf, 'N/A')} s")
    
    # Compute reduction and speedup for each method
    results = []
    for method in methods:
        memory_vals = []
        runtime_vals = []
        for model_family in model_families:
            # Memory
            if model_family in memory_data.get("lora", {}) and method in memory_data["lora"][model_family]:
                mem_vals = memory_data["lora"][model_family][method]
                if mem_vals and model_family in fft_memory:
                    avg_mem = statistics.mean(mem_vals)
                    reduction = (fft_memory[model_family] - avg_mem) / fft_memory[model_family] * 100
                    memory_vals.append(reduction)
            # Runtime
            if model_family in runtime_data.get("lora", {}) and method in runtime_data["lora"][model_family]:
                rt_vals = runtime_data["lora"][model_family][method]
                if rt_vals and model_family in fft_runtime:
                    avg_rt = statistics.mean(rt_vals)
                    speedup = fft_runtime[model_family] / avg_rt
                    runtime_vals.append(speedup)
        
        avg_memory_reduction = statistics.mean(memory_vals) if memory_vals else None
        avg_speedup = statistics.mean(runtime_vals) if runtime_vals else None
        results.append({
            "method": method,
            "memory_reduction_%": avg_memory_reduction,
            "inference_speedup": avg_speedup,
            "n_memory": len(memory_vals),
            "n_runtime": len(runtime_vals)
        })
        print(f"{method:10s} memory reduction: {avg_memory_reduction:.1f}% (n={len(memory_vals)}), speedup: {avg_speedup:.1f}x (n={len(runtime_vals)})")
    
    return results

def main():
    memory_data, runtime_data = scan_results()
    results = compute_averages(memory_data, runtime_data)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv("memory_speedup_averages.csv", index=False)
    print("\nSaved to memory_speedup_averages.csv")
    
    # Also generate combined efficiency table with parameter reduction
    # Load parameter reduction from earlier computation
    param_df = pd.read_csv("table_ii_efficiency_complete.csv")
    # Merge
    method_map = {
        "lora": "LoRA",
        "mrlora": "MR‑LoRA",
        "adalora": "AdaLoRA",
        "dora": "DoRA",
        "olora": "OLoRA",
        "rslora": "RS‑LoRA",
        "mrlora-rs": "MR‑LoRA‑RS"
    }
    for res in results:
        name = method_map[res["method"]]
        # Update param_df rows
        mask = param_df["Method"] == name
        if mask.any():
            param_df.loc[mask, "Memory Reduction"] = f"{res['memory_reduction_%']:.1f}%"
            param_df.loc[mask, "Inference Speedup"] = f"{res['inference_speedup']:.1f}×"
    
    param_df.to_csv("table_ii_efficiency_updated.csv", index=False)
    print("\nUpdated efficiency table saved to table_ii_efficiency_updated.csv")
    print(param_df.to_string())

if __name__ == "__main__":
    main()