#!/usr/bin/env python3
"""
Replace the efficiency paragraph and table with updated values.
"""
import sys

# New paragraph (updated)
new_paragraph = "Table~\\ref{tab:efficiency} compares the resource requirements of each method (measured at rank 8). All LoRA variants reduce trainable parameters by approximately 99.5\\% relative to FFT. Memory reduction (74--82\\%) and inference speedup (2.4--4.5$\\times$) are averaged across BERT, RoBERTa, and DeBERTa model families."

# New table content (lines)
new_table_lines = [
    "\\begin{table}[ht]",
    "\\centering",
    "\\caption{Efficiency metrics for LoRA variants. Parameter reduction is relative to full fine‑tuning (FFT); each cell shows teacher fine‑tuning/student distillation values separated by a slash. Memory reduction and inference speedup are averaged across BERT, RoBERTa, and DeBERTa model families.}",
    "\\label{tab:efficiency}",
    "\\begin{tabular}{lccc}",
    "\\toprule",
    "Method & Parameter Reduction & Memory Reduction & Inference Speedup \\\\",
    "\\midrule",
    "LoRA & 99.6\\%/99.5\\% & 74.3\\%/82.1\\% & 3.7$\\times$/4.0$\\times$ \\\\",
    "MR‑LoRA & 99.4\\%/99.4\\% & 74.0\\%/82.0\\% & 3.3$\\times$/4.0$\\times$ \\\\",
    "AdaLoRA & 99.5\\% & 74.2\\% & 3.5$\\times$ \\\\",
    "DoRA & 99.6\\% & 74.3\\% & 2.9$\\times$ \\\\",
    "OLoRA & 99.6\\% & 74.2\\% & 4.0$\\times$ \\\\",
    "RS‑LoRA & 99.6\\% & 74.3\\% & 4.5$\\times$ \\\\",
    "MR‑LoRA‑RS & 99.4\\% & 74.0\\% & 2.4$\\times$ \\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
    "",  # empty line after table
]

def main():
    with open("main.tex", "r") as f:
        lines = f.readlines()
    
    # Find start line of paragraph
    start_idx = None
    for i, line in enumerate(lines):
        if "Table~\\ref{tab:efficiency} compares" in line:
            start_idx = i
            break
    if start_idx is None:
        print("Could not find efficiency paragraph")
        sys.exit(1)
    
    # Find end of the table (look for \end{table} after start_idx)
    end_idx = None
    in_table = False
    for i in range(start_idx, len(lines)):
        if "\\begin{table}[ht]" in lines[i]:
            in_table = True
        if in_table and "\\end{table}" in lines[i]:
            end_idx = i
            break
    if end_idx is None:
        print("Could not find end of table")
        sys.exit(1)
    
    print(f"Replacing lines {start_idx+1} to {end_idx+1}")
    
    # Replace lines[start_idx:end_idx+1] with new paragraph + new table
    new_content = [new_paragraph + "\n", "\n"] + new_table_lines
    lines[start_idx:end_idx+1] = new_content
    
    with open("main.tex", "w") as f:
        f.writelines(lines)
    
    print("Efficiency section updated successfully.")

if __name__ == "__main__":
    main()