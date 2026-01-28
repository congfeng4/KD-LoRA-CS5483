#!/usr/bin/env python3
"""
Generate complete LaTeX tables for per-task performance and efficiency metrics.
Uses existing CSV files: teacher_per_task.csv, student_per_task.csv, 
table_i_detailed.csv, tradeoff_data.csv, table_ii_efficiency.csv.
"""
import pandas as pd
import numpy as np
import statistics

# Read data
teacher_per_task = pd.read_csv("teacher_per_task.csv")
student_per_task = pd.read_csv("student_per_task.csv")
table_i_detailed = pd.read_csv("table_i_detailed.csv")
tradeoff_df = pd.read_csv("tradeoff_data.csv")
efficiency_df = pd.read_csv("table_ii_efficiency.csv")

# Tasks (excluding GLUE column)
tasks = ["CoLA", "SST‑2", "MRPC", "QQP", "STS‑B", "QNLI", "RTE", "WNLI", "MNLI"]
# Note: MNLI column in teacher_per_task.csv is empty; we'll need to get from table_i_detailed

# Teacher variants
teacher_methods = ["lora", "mrlora", "adalora", "dora", "olora", "rslora", "mrlora-rs"]
method_names = {
    "lora": "LoRA", "mrlora": "MR‑LoRA", "adalora": "AdaLoRA",
    "dora": "DoRA", "olora": "OLoRA", "rslora": "RS‑LoRA",
    "mrlora-rs": "MR‑LoRA‑RS"
}

# Compute per-task averages across model families for each teacher method
teacher_avg = {}
for method in teacher_methods:
    # Filter rows for this method
    df_method = teacher_per_task[teacher_per_task["Method"] == method]
    if df_method.empty:
        continue
    # Average across model families for each task
    task_avgs = {}
    for task in tasks:
        if task in df_method.columns:
            # Drop missing values
            vals = df_method[task].dropna()
            if len(vals) > 0:
                # Convert to float (they are strings with .2f)
                vals = vals.astype(float)
                task_avgs[task] = vals.mean()
    # GLUE average (provided column)
    if "GLUE" in df_method.columns:
        glue_vals = df_method["GLUE"].dropna().astype(float)
        if len(glue_vals) > 0:
            task_avgs["GLUE"] = glue_vals.mean()
    teacher_avg[method] = task_avgs

# Student variants
student_methods = ["lora_student", "mrlora_student"]
student_names = {"lora_student": "LoRA", "mrlora_student": "MR‑LoRA"}

student_avg = {}
for method in student_methods:
    df_method = student_per_task[student_per_task["Method"] == method]
    if df_method.empty:
        continue
    task_avgs = {}
    for task in tasks:
        if task in df_method.columns:
            vals = df_method[task].dropna().astype(float)
            if len(vals) > 0:
                task_avgs[task] = vals.mean()
    if "GLUE" in df_method.columns:
        glue_vals = df_method["GLUE"].dropna().astype(float)
        if len(glue_vals) > 0:
            task_avgs["GLUE"] = glue_vals.mean()
    student_avg[method] = task_avgs

# Get MNLI data from table_i_detailed.csv (FFT, LoRA, KD-LoRA)
# We'll need to extend to other variants later, but for now we only have for LoRA and KD-LoRA
# We'll ignore for other variants (leave blank)

# Generate LaTeX teacher table
with open("teacher_per_task_complete.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Per‑task GLUE scores for teacher LoRA variants, averaged across BERT, RoBERTa, and DeBERTa model families.}\n")
    f.write("\\label{tab:teacher-per-task}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{l" + "c" * (len(tasks) + 1) + "}\n")
    f.write("\\toprule\n")
    f.write("Method & " + " & ".join(tasks) + " & GLUE \\\\\n")
    f.write("\\midrule\n")
    for method in teacher_methods:
        if method not in teacher_avg:
            continue
        avgs = teacher_avg[method]
        line = method_names[method]
        for task in tasks:
            val = avgs.get(task, None)
            line += " & " + (f"{val:.1f}" if val is not None else "--")
        glue = avgs.get("GLUE", None)
        line += " & " + (f"{glue:.1f}" if glue is not None else "--") + " \\\\\n"
        f.write(line)
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

# Generate LaTeX student table
with open("student_per_task_complete.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Per‑task GLUE scores for student LoRA variants (knowledge distillation), averaged across BERT, RoBERTa, and DeBERTa model families.}\n")
    f.write("\\label{tab:student-per-task}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{l" + "c" * (len(tasks) + 1) + "}\n")
    f.write("\\toprule\n")
    f.write("Method & " + " & ".join(tasks) + " & GLUE \\\\\n")
    f.write("\\midrule\n")
    for method in student_methods:
        if method not in student_avg:
            continue
        avgs = student_avg[method]
        line = student_names[method]
        for task in tasks:
            val = avgs.get(task, None)
            line += " & " + (f"{val:.1f}" if val is not None else "--")
        glue = avgs.get("GLUE", None)
        line += " & " + (f"{glue:.1f}" if glue is not None else "--") + " \\\\\n"
        f.write(line)
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

# Efficiency metrics
# Use tradeoff_data.csv for parameter percentages
# Compute parameter reduction relative to FFT
fft_params = {}
for model_family in ["bert", "roberta", "deberta"]:
    fft_row = tradeoff_df[(tradeoff_df["model_family"] == model_family) & (tradeoff_df["method"] == "fft")]
    if not fft_row.empty:
        fft_params[model_family] = fft_row["param_count_m"].values[0]

# Compute parameter percentage for each method
param_pct = {}
for model_family in ["bert", "roberta", "deberta"]:
    for method in teacher_methods + ["lora", "mrlora"]:  # include teacher methods
        row = tradeoff_df[(tradeoff_df["model_family"] == model_family) & (tradeoff_df["method"] == method)]
        if not row.empty and model_family in fft_params:
            pct = row["param_count_m"].values[0] / fft_params[model_family] * 100
            param_pct[(model_family, method)] = pct

# Average across model families
param_avg = {}
for method in teacher_methods + ["lora", "mrlora"]:
    vals = [param_pct[(fam, method)] for fam in ["bert", "roberta", "deberta"] if (fam, method) in param_pct]
    if vals:
        param_avg[method] = statistics.mean(vals)

# Memory reduction and inference speedup from table_ii_efficiency.csv (only LoRA and MR-LoRA)
efficiency_dict = {}
for _, row in efficiency_df.iterrows():
    method = row["Method"]
    # Parse percentages
    param_red = float(row["Parameter Reduction"].rstrip('%'))
    mem_red = float(row["Memory Reduction"].rstrip('%'))
    speedup = float(row["Inference Speedup"].rstrip('×'))
    efficiency_dict[method] = {"param": param_red, "mem": mem_red, "speed": speedup}

# Generate efficiency table
with open("efficiency_complete.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Efficiency metrics for LoRA variants. Parameter reduction is relative to full fine‑tuning (FFT). Memory reduction and inference speedup are averaged across BERT, RoBERTa, and DeBERTa model families.}\n")
    f.write("\\label{tab:efficiency-complete}\n")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\toprule\n")
    f.write("Method & Parameter Reduction (\\% FFT) & Memory Reduction (\\%) & Inference Speedup \\\\\n")
    f.write("\\midrule\n")
    # Add rows for each method
    for method in teacher_methods:
        name = method_names[method]
        param_val = param_avg.get(method, None)
        if param_val is None:
            continue
        # Memory and speedup only available for LoRA and MR‑LoRA
        mem = efficiency_dict.get(name, {}).get("mem", None)
        speed = efficiency_dict.get(name, {}).get("speed", None)
        param_str = f"{100 - param_val:.1f}%"  # convert to reduction percentage
        mem_str = f"{mem:.1f}%" if mem is not None else "--"
        speed_str = f"{speed:.1f}$\\times$" if speed is not None else "--"
        f.write(f"{name} & {param_str} & {mem_str} & {speed_str} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print("Generated LaTeX tables:")
print("  - teacher_per_task_complete.tex")
print("  - student_per_task_complete.tex")
print("  - efficiency_complete.tex")