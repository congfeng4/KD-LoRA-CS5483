#!/usr/bin/env python3
"""
Compute percentage‑based per‑task GLUE tables relative to FFT baselines.
Inputs:
  - teacher_per_task.csv: raw per‑task scores for teacher variants
  - student_per_task.csv: raw per‑task scores for student variants (only LoRA, MR‑LoRA)
  - glue_averages.csv: GLUE averages for all variants (teacher & student)
  - table_i_detailed.csv: FFT, LoRA, KD‑LoRA per‑task scores
  - table_i_mrlora.csv: FFT, Teacher MR‑LoRA, Student MR‑LoRA per‑task scores
Outputs:
  - teacher_per_task_pct.tex: LaTeX table with percentages for teacher variants
  - student_per_task_pct.tex: LaTeX table with percentages for student variants
"""

import pandas as pd
import numpy as np
import statistics

# ========== Load data ==========
teacher_raw = pd.read_csv("teacher_per_task.csv")
student_raw = pd.read_csv("student_per_task.csv")
glue_avg = pd.read_csv("glue_averages.csv")
table_i = pd.read_csv("table_i_detailed.csv")
table_mr = pd.read_csv("table_i_mrlora.csv")

# Task columns (excluding MNLI for consistency with existing tables)
tasks = ["CoLA", "SST‑2", "MRPC", "QQP", "STS‑B", "QNLI", "RTE", "WNLI"]
task_cols = tasks

# Model families
families = ["bert", "roberta", "deberta"]

# Method names mapping
method_names = {
    "lora": "LoRA",
    "mrlora": "MR‑LoRA",
    "adalora": "AdaLoRA",
    "dora": "DoRA",
    "olora": "OLoRA",
    "rslora": "RS‑LoRA",
    "mrlora-rs": "MR‑LoRA‑RS"
}

# ========== Build FFT baseline dictionary ==========
# FFT per‑task scores for each model family
fft_baseline = {}  # family -> task -> score

# Extract from table_i_detailed.csv (FFT columns)
# Columns: "BERT‑b/DBERT‑b FFT", "RoB‑b/DRoB‑b FFT", "DeB‑b/DeB‑s FFT"
fft_cols = ["BERT‑b/DBERT‑b FFT", "RoB‑b/DRoB‑b FFT", "DeB‑b/DeB‑s FFT"]
family_to_col = {"bert": fft_cols[0], "roberta": fft_cols[1], "deberta": fft_cols[2]}

for fam, col in family_to_col.items():
    fft_baseline[fam] = {}
    for _, row in table_i.iterrows():
        task = row["Task"]
        if task in tasks:
            fft_baseline[fam][task] = float(row[col])

# Add MNLI scores from table_i_mrlora.csv (same column names)
for fam, col in family_to_col.items():
    for _, row in table_mr.iterrows():
        task = row["Task"]
        if task in tasks:
            # Use MR‑LoRA table's FFT column (should be identical)
            fft_baseline[fam][task] = float(row[col])

# ========== Compute GLUE baseline (FFT average across tasks) ==========
fft_glue = {}
for fam in families:
    task_scores = [fft_baseline[fam][t] for t in tasks if t in fft_baseline[fam]]
    if task_scores:
        fft_glue[fam] = statistics.mean(task_scores)

# ========== Helper: compute percentage ==========
def to_pct(raw, fft_score):
    """Convert raw score to percentage of FFT."""
    if pd.isna(raw) or pd.isna(fft_score) or fft_score == 0:
        return None
    return (raw / fft_score) * 100

# ========== Teacher variants ==========
teacher_data = []
for method in method_names.keys():
    df_method = teacher_raw[teacher_raw["Method"] == method]
    if df_method.empty:
        continue
    
    row = {"Method": method_names[method]}
    task_pcts = []
    
    for task in task_cols:
        if task not in df_method.columns:
            row[task] = None
            continue
        
        # Collect percentages per model family
        pcts = []
        for fam in families:
            fam_rows = df_method[df_method["Model Family"] == fam]
            if fam_rows.empty:
                continue
            raw_val = fam_rows[task].values[0]
            if pd.isna(raw_val):
                continue
            if fam in fft_baseline and task in fft_baseline[fam]:
                pct = to_pct(float(raw_val), fft_baseline[fam][task])
                if pct:
                    pcts.append(pct)
        
        if pcts:
            avg_pct = statistics.mean(pcts)
            row[task] = f"{avg_pct:.1f}\\%"
            task_pcts.append(avg_pct)
        else:
            row[task] = "--"
    
    # GLUE percentage (from glue_averages.csv)
    glue_rows = glue_avg[glue_avg["method"] == method]
    if not glue_rows.empty:
        # Average across families
        glue_vals = []
        for fam in families:
            fam_glue = glue_rows[glue_rows["model_family"] == fam]
            if not fam_glue.empty:
                raw_glue = fam_glue["glue_score"].values[0]
                if fam in fft_glue:
                    pct = to_pct(raw_glue, fft_glue[fam])
                    if pct:
                        glue_vals.append(pct)
        if glue_vals:
            row["GLUE"] = f"{statistics.mean(glue_vals):.1f}\\%"
        else:
            row["GLUE"] = "--"
    else:
        row["GLUE"] = "--"
    
    teacher_data.append(row)

# ========== Student variants ==========
# We have per‑task data only for LoRA and MR‑LoRA; for other variants we only have GLUE averages
student_methods_raw = ["lora_student", "mrlora_student"]
student_methods_all = ["lora", "mrlora", "adalora", "dora", "olora", "rslora", "mrlora-rs"]

student_data = []
for method in student_methods_all:
    display_name = method_names[method]
    row = {"Method": display_name}
    
    # Check if we have per‑task raw data
    raw_key = f"{method}_student" if method in ["lora", "mrlora"] else None
    has_per_task = False
    
    if raw_key and raw_key in student_raw["Method"].values:
        df_method = student_raw[student_raw["Method"] == raw_key]
        has_per_task = True
        
        task_pcts = []
        for task in task_cols:
            if task not in df_method.columns:
                row[task] = "--"
                continue
            
            pcts = []
            for fam in families:
                fam_rows = df_method[df_method["Model Family"] == fam]
                if fam_rows.empty:
                    continue
                raw_val = fam_rows[task].values[0]
                if pd.isna(raw_val):
                    continue
                if fam in fft_baseline and task in fft_baseline[fam]:
                    pct = to_pct(float(raw_val), fft_baseline[fam][task])
                    if pct:
                        pcts.append(pct)
            
            if pcts:
                avg_pct = statistics.mean(pcts)
                row[task] = f"{avg_pct:.1f}\\%"
                task_pcts.append(avg_pct)
            else:
                row[task] = "--"
    else:
        # No per‑task data → fill with "--"
        for task in task_cols:
            row[task] = "--"
    
    # GLUE percentage (from glue_averages.csv)
    glue_rows = glue_avg[glue_avg["method"] == method]
    if not glue_rows.empty:
        glue_vals = []
        for fam in families:
            fam_glue = glue_rows[glue_rows["model_family"] == fam]
            if not fam_glue.empty:
                raw_glue = fam_glue["glue_score"].values[0]
                if fam in fft_glue:
                    pct = to_pct(raw_glue, fft_glue[fam])
                    if pct:
                        glue_vals.append(pct)
        if glue_vals:
            row["GLUE"] = f"{statistics.mean(glue_vals):.1f}\\%"
        else:
            row["GLUE"] = "--"
    else:
        row["GLUE"] = "--"
    
    student_data.append(row)

# ========== Generate LaTeX tables ==========
# Teacher table
with open("teacher_per_task_pct.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Per‑task GLUE scores (percentage of FFT) for teacher LoRA variants, averaged across BERT, RoBERTa, and DeBERTa model families.}\n")
    f.write("\\label{tab:teacher-per-task-pct}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{l" + "c" * (len(task_cols) + 1) + "}\n")
    f.write("\\toprule\n")
    f.write("Method & " + " & ".join(task_cols) + " & GLUE \\\\\n")
    f.write("\\midrule\n")
    for row in teacher_data:
        line = row["Method"]
        for task in task_cols:
            line += " & " + row.get(task, "--")
        line += " & " + row.get("GLUE", "--") + " \\\\\n"
        f.write(line)
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

# Student table
with open("student_per_task_pct.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Per‑task GLUE scores (percentage of teacher FFT) for student LoRA variants (knowledge distillation), averaged across BERT, RoBERTa, and DeBERTa model families. Per‑task data are available only for LoRA and MR‑LoRA; other variants show only the overall GLUE percentage.}\n")
    f.write("\\label{tab:student-per-task-pct}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{l" + "c" * (len(task_cols) + 1) + "}\n")
    f.write("\\toprule\n")
    f.write("Method & " + " & ".join(task_cols) + " & GLUE \\\\\n")
    f.write("\\midrule\n")
    for row in student_data:
        line = row["Method"]
        for task in task_cols:
            line += " & " + row.get(task, "--")
        line += " & " + row.get("GLUE", "--") + " \\\\\n"
        f.write(line)
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print("Generated:")
print("  - teacher_per_task_pct.tex")
print("  - student_per_task_pct.tex")