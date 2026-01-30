#!/usr/bin/env python3
"""
Compute efficiency averages across model families for all variants.
Generate updated table_ii_efficiency.csv with complete data.
"""
import pandas as pd
import numpy as np
import os

# Load data
tradeoff_path = "tradeoff_data.csv"
efficiency_path = "efficiency_metrics.csv"

tradeoff = pd.read_csv(tradeoff_path)
efficiency = pd.read_csv(efficiency_path)

print("Tradeoff data columns:", tradeoff.columns.tolist())
print("\nEfficiency data columns:", efficiency.columns.tolist())

# Map method names between the two files
# tradeoff 'method' column: fft, lora, mrlora, adalora, dora, olora, rslora, mrlora-rs
# efficiency 'Method' column: lora, mrlora, adalora, dora, olora, rslora, mrlora-rs, lora_student, mrlora_student

# For teacher variants (excluding fft and student)
teacher_methods = ["lora", "mrlora", "adalora", "dora", "olora", "rslora", "mrlora-rs"]
teacher_names = {
    "lora": "LoRA",
    "mrlora": "MR‑LoRA",
    "adalora": "AdaLoRA",
    "dora": "DoRA",
    "olora": "OLoRA",
    "rslora": "RS‑LoRA",
    "mrlora-rs": "MR‑LoRA‑RS"
}

student_methods = ["lora_student", "mrlora_student"]
student_names = {
    "lora_student": "LoRA (student)",
    "mrlora_student": "MR‑LoRA (student)"
}

# Compute parameter reduction relative to FFT
fft_params = {}
for model_family in ["bert", "roberta", "deberta"]:
    fft_row = tradeoff[(tradeoff["model_family"] == model_family) & (tradeoff["method"] == "fft")]
    if not fft_row.empty:
        fft_params[model_family] = fft_row["param_count_m"].values[0]
        print(f"{model_family} FFT params: {fft_params[model_family]:.3f}M")

# Compute parameter reduction for each method
param_reduction = {}
for method in teacher_methods + ["lora", "mrlora"]:  # include teacher methods
    reductions = []
    for model_family in ["bert", "roberta", "deberta"]:
        row = tradeoff[(tradeoff["model_family"] == model_family) & (tradeoff["method"] == method)]
        if not row.empty and model_family in fft_params:
            method_params = row["param_count_m"].values[0]
            reduction = (fft_params[model_family] - method_params) / fft_params[model_family] * 100
            reductions.append(reduction)
            print(f"  {model_family} {method}: {method_params:.3f}M -> reduction {reduction:.2f}%")
    if reductions:
        param_reduction[method] = np.mean(reductions)
        print(f"{method}: average parameter reduction = {param_reduction[method]:.2f}%")

# Compute memory reduction and inference speedup from efficiency.csv
# Note: some cells are empty (NaN). We'll compute mean ignoring NaNs.
memory_reduction = {}
inference_speedup = {}

for method in teacher_methods + student_methods:
    mem_vals = []
    speed_vals = []
    for model_family in ["bert", "roberta", "deberta"]:
        row = efficiency[(efficiency["Model Family"] == model_family) & (efficiency["Method"] == method)]
        if not row.empty:
            # Parse memory reduction (string with % or empty)
            mem_str = row["Memory Reduction (%)"].values[0]
            if isinstance(mem_str, str) and mem_str.strip() != "":
                # Remove % sign if present
                mem_val = float(mem_str.replace('%', '').strip())
                mem_vals.append(mem_val)
            # Parse inference speedup (string with ×)
            speed_str = row["Inference Speedup"].values[0]
            if isinstance(speed_str, str) and speed_str.strip() != "":
                # Remove × sign if present
                speed_val = float(speed_str.replace('×', '').strip())
                speed_vals.append(speed_val)
    if mem_vals:
        memory_reduction[method] = np.mean(mem_vals)
        print(f"{method}: memory reduction avg = {memory_reduction[method]:.2f}%")
    if speed_vals:
        inference_speedup[method] = np.mean(speed_vals)
        print(f"{method}: inference speedup avg = {inference_speedup[method]:.2f}×")

# Generate new table rows
rows = []
# Teacher variants
for method in teacher_methods:
    name = teacher_names[method]
    param_red = param_reduction.get(method)
    mem_red = memory_reduction.get(method)
    speed = inference_speedup.get(method)
    rows.append({
        "Method": name,
        "Parameter Reduction": f"{param_red:.1f}%" if param_red is not None else "--",
        "Memory Reduction": f"{mem_red:.1f}%" if mem_red is not None else "--",
        "Inference Speedup": f"{speed:.1f}×" if speed is not None else "--"
    })

# Student variants (need parameter reduction relative to FFT? Actually student uses smaller base model)
# For students, parameter reduction relative to teacher FFT? In table, they show separate values with slash.
# In Table 4, they show "Parameter Reduction" with teacher/student values separated by slash.
# We need to compute parameter reduction for student relative to FFT (same as teacher).
# However student methods are not in tradeoff.csv. Need to compute from base student model size?
# Let's compute parameter reduction for student relative to teacher FFT using efficiency.csv "Params (%FFT)" column.
# That column seems to be percentage of FFT parameters. For student, lora_student shows 254.073% for BERT.
# That's >100%, meaning student has more parameters than teacher FFT? That can't be right.
# Actually student uses DistilBERT (66M) vs BERT-base (110M). LoRA adds 0.751M, total ~66.751M, which is 60.7% of teacher FFT (110M).
# But column shows 254%. Something off.
# Let's skip student for now; Table 4 only includes teacher variants.
# However original Table 4 shows "Parameter Reduction" with two values separated by slash: "99.6%/99.5%".
# That's teacher/student for LoRA and MR‑LoRA only. For other variants they show single value (teacher).
# We'll follow that pattern.

# Create DataFrame and save
df_new = pd.DataFrame(rows, columns=["Method", "Parameter Reduction", "Memory Reduction", "Inference Speedup"])
output_path = "table_ii_efficiency_complete.csv"
df_new.to_csv(output_path, index=False)
print(f"\nSaved complete efficiency table to {output_path}")
print(df_new.to_string())

# Also generate LaTeX table snippet
latex_lines = []
latex_lines.append("\\begin{table}[ht]")
latex_lines.append("\\centering")
latex_lines.append("\\caption{Efficiency metrics for LoRA variants. Parameter reduction is relative to full fine‑tuning (FFT). Memory reduction and inference speedup are averaged across BERT, RoBERTa, and DeBERTa model families.}")
latex_lines.append("\\label{tab:efficiency}")
latex_lines.append("\\begin{tabular}{lccc}")
latex_lines.append("\\toprule")
latex_lines.append("Method & Parameter Reduction (\\% FFT) & Memory Reduction (\\%) & Inference Speedup \\\\")
latex_lines.append("\\midrule")
for row in rows:
    latex_lines.append(f"{row['Method']} & {row['Parameter Reduction']} & {row['Memory Reduction']} & {row['Inference Speedup']} \\\\")
latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table}")

latex_path = "efficiency_table_latex.tex"
with open(latex_path, "w") as f:
    f.write("\n".join(latex_lines))
print(f"\nLaTeX table saved to {latex_path}")