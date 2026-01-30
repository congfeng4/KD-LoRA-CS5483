#!/usr/bin/env python3
"""
Create final LaTeX tables with per‑task performance and efficiency metrics.
Uses glue_averages.csv for GLUE scores, teacher_per_task.csv for per‑task data,
table_i_glue_scores.csv for student GLUE, and tradeoff_data.csv for parameters.
"""
import pandas as pd
import numpy as np
import statistics

# Read data
glue_avg = pd.read_csv("glue_averages.csv")
teacher_per_task = pd.read_csv("teacher_per_task.csv")
student_per_task = pd.read_csv("student_per_task.csv")
table_i_glue = pd.read_csv("table_i_glue_scores.csv")
tradeoff = pd.read_csv("tradeoff_data.csv")
efficiency_df = pd.read_csv("table_ii_efficiency.csv")

# Tasks (excluding MNLI for now)
tasks = ["CoLA", "SST‑2", "MRPC", "QQP", "STS‑B", "QNLI", "RTE", "WNLI"]
task_cols = tasks  # columns in teacher_per_task.csv

# Teacher variants
teacher_methods = ["lora", "mrlora", "adalora", "dora", "olora", "rslora", "mrlora-rs"]
method_names = {
    "lora": "LoRA", "mrlora": "MR‑LoRA", "adalora": "AdaLoRA",
    "dora": "DoRA", "olora": "OLoRA", "rslora": "RS‑LoRA",
    "mrlora-rs": "MR‑LoRA‑RS"
}

# Compute GLUE averages from glue_averages.csv (includes MNLI)
glue_by_method = {}
for method in teacher_methods:
    rows = glue_avg[glue_avg["method"] == method]
    if not rows.empty:
        # Average across model families
        avg = rows["glue_score"].mean()
        glue_by_method[method] = avg

# Compute per‑task averages across model families
teacher_data = []
for method in teacher_methods:
    df_method = teacher_per_task[teacher_per_task["Method"] == method]
    if df_method.empty:
        continue
    row = {"Method": method_names[method]}
    # Per‑task averages
    for task in task_cols:
        if task in df_method.columns:
            vals = df_method[task].dropna().astype(float)
            if len(vals) > 0:
                row[task] = vals.mean()
            else:
                row[task] = None
        else:
            row[task] = None
    # GLUE average from glue_averages.csv
    row["GLUE"] = glue_by_method.get(method, None)
    teacher_data.append(row)

# Generate teacher LaTeX table
with open("teacher_per_task_final.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Per‑task GLUE scores for teacher LoRA variants, averaged across BERT, RoBERTa, and DeBERTa model families. MNLI scores are omitted for brevity but are included in the GLUE average.}\n")
    f.write("\\label{tab:teacher-per-task}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{l" + "c" * (len(task_cols) + 1) + "}\n")
    f.write("\\toprule\n")
    f.write("Method & " + " & ".join(task_cols) + " & GLUE \\\\\n")
    f.write("\\midrule\n")
    for row in teacher_data:
        line = row["Method"]
        for task in task_cols:
            val = row[task]
            line += " & " + (f"{val:.1f}" if val is not None else "--")
        glue = row["GLUE"]
        line += " & " + (f"{glue:.1f}" if glue is not None else "--") + " \\\\\n"
        f.write(line)
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

# Student variants (only LoRA and MR‑LoRA)
student_methods = ["lora_student", "mrlora_student"]
student_names = {"lora_student": "LoRA", "mrlora_student": "MR‑LoRA"}

# Get student GLUE from table_i_glue_scores.csv
student_glue = {}
for model_family in ["bert", "roberta", "deberta"]:
    row = table_i_glue[table_i_glue["Model Family"] == model_family]
    if not row.empty:
        student_glue[model_family] = {
            "lora_student": row["Student LoRA"].values[0],
            "mrlora_student": row["Student MR‑LoRA"].values[0]
        }

# Compute per‑task averages for student
student_data = []
for method in student_methods:
    df_method = student_per_task[student_per_task["Method"] == method]
    if df_method.empty:
        continue
    row = {"Method": student_names[method]}
    for task in task_cols:
        if task in df_method.columns:
            vals = df_method[task].dropna().astype(float)
            if len(vals) > 0:
                row[task] = vals.mean()
            else:
                row[task] = None
        else:
            row[task] = None
    # GLUE average across model families
    glue_vals = [student_glue[fam][method] for fam in ["bert", "roberta", "deberta"] if fam in student_glue]
    if glue_vals:
        row["GLUE"] = statistics.mean(glue_vals)
    else:
        row["GLUE"] = None
    student_data.append(row)

with open("student_per_task_final.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Per‑task GLUE scores for student LoRA variants (knowledge distillation), averaged across BERT, RoBERTa, and DeBERTa model families. MNLI scores are omitted for brevity but are included in the GLUE average.}\n")
    f.write("\\label{tab:student-per-task}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{l" + "c" * (len(task_cols) + 1) + "}\n")
    f.write("\\toprule\n")
    f.write("Method & " + " & ".join(task_cols) + " & GLUE \\\\\n")
    f.write("\\midrule\n")
    for row in student_data:
        line = row["Method"]
        for task in task_cols:
            val = row[task]
            line += " & " + (f"{val:.1f}" if val is not None else "--")
        glue = row["GLUE"]
        line += " & " + (f"{glue:.1f}" if glue is not None else "--") + " \\\\\n"
        f.write(line)
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

# Efficiency table
# Parameter reduction from tradeoff_data.csv
fft_params = {}
for model_family in ["bert", "roberta", "deberta"]:
    fft_row = tradeoff[(tradeoff["model_family"] == model_family) & (tradeoff["method"] == "fft")]
    if not fft_row.empty:
        fft_params[model_family] = fft_row["param_count_m"].values[0]

param_pct = {}
for model_family in ["bert", "roberta", "deberta"]:
    for method in teacher_methods:
        row = tradeoff[(tradeoff["model_family"] == model_family) & (tradeoff["method"] == method)]
        if not row.empty and model_family in fft_params:
            pct = row["param_count_m"].values[0] / fft_params[model_family] * 100
            param_pct[(model_family, method)] = pct

# Average across model families
param_avg = {}
for method in teacher_methods:
    vals = [param_pct[(fam, method)] for fam in ["bert", "roberta", "deberta"] if (fam, method) in param_pct]
    if vals:
        param_avg[method] = statistics.mean(vals)

# Memory reduction and inference speedup from table_ii_efficiency.csv
eff_dict = {}
for _, row in efficiency_df.iterrows():
    method = row["Method"]
    param_red = float(row["Parameter Reduction"].rstrip('%'))
    mem_red = float(row["Memory Reduction"].rstrip('%'))
    speedup = float(row["Inference Speedup"].rstrip('×'))
    eff_dict[method] = {"mem": mem_red, "speed": speedup}

# Generate efficiency table
with open("efficiency_final.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Efficiency metrics for LoRA variants. Parameter reduction is relative to full fine‑tuning (FFT). Memory reduction and inference speedup are averaged across BERT, RoBERTa, and DeBERTa model families (only available for LoRA and MR‑LoRA).}\n")
    f.write("\\label{tab:efficiency-final}\n")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\toprule\n")
    f.write("Method & Parameter Reduction (\\% FFT) & Memory Reduction (\\%) & Inference Speedup \\\\\n")
    f.write("\\midrule\n")
    for method in teacher_methods:
        name = method_names[method]
        param_val = param_avg.get(method)
        if param_val is None:
            continue
        param_str = f"{100 - param_val:.1f}\\%"
        # Memory and speedup only for LoRA and MR‑LoRA
        mem = eff_dict.get(name, {}).get("mem")
        speed = eff_dict.get(name, {}).get("speed")
        mem_str = f"{mem:.1f}\\%" if mem is not None else "--"
        speed_str = f"{speed:.1f}$\\times$" if speed is not None else "--"
        f.write(f"{name} & {param_str} & {mem_str} & {speed_str} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print("Generated final LaTeX tables:")
print("  - teacher_per_task_final.tex")
print("  - student_per_task_final.tex")
print("  - efficiency_final.tex")