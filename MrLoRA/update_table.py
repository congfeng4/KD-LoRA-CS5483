#!/usr/bin/env python3
"""
Replace Table 4 in main.tex with updated efficiency metrics.
"""
import re

# New table content (with updated caption and all values)
new_table = """\\begin{table}[ht]
\\centering
\\caption{Efficiency metrics for LoRA variants. Parameter reduction is relative to full fine‑tuning (FFT); each cell shows teacher fine‑tuning/student distillation values separated by a slash. Memory reduction and inference speedup are averaged across BERT, RoBERTa, and DeBERTa model families.}
\\label{tab:efficiency}
\\begin{tabular}{lccc}
\\toprule
Method & Parameter Reduction & Memory Reduction & Inference Speedup \\\\
\\midrule
LoRA & 99.6\\%/99.5\\% & 74.3\\%/82.1\\% & 3.7$\\times$/4.0$\\times$ \\\\
MR‑LoRA & 99.4\\%/99.4\\% & 74.0\\%/82.0\\% & 3.3$\\times$/4.0$\\times$ \\\\
AdaLoRA & 99.5\\% & 74.2\\% & 3.5$\\times$ \\\\
DoRA & 99.6\\% & 74.3\\% & 2.9$\\times$ \\\\
OLoRA & 99.6\\% & 74.2\\% & 4.0$\\times$ \\\\
RS‑LoRA & 99.6\\% & 74.3\\% & 4.5$\\times$ \\\\
MR‑LoRA‑RS & 99.4\\% & 74.0\\% & 2.4$\\times$ \\\\
\\bottomrule
\\end{tabular}
\\end{table}"""

# Read main.tex
with open("main.tex", "r") as f:
    content = f.read()

# Pattern to find the efficiency table
pattern = r'\\begin{table}\[ht\].*?\\caption\{Efficiency metrics for LoRA variants\..*?\\end{table}'
# Use re.DOTALL to match across lines
match = re.search(pattern, content, re.DOTALL)
if match:
    old_table = match.group(0)
    print("Found table, replacing...")
    content = content.replace(old_table, new_table)
    with open("main.tex", "w") as f:
        f.write(content)
    print("Table updated successfully.")
else:
    print("Table not found! Trying alternative pattern...")
    # Try simpler pattern
    pattern2 = r'\\begin{table}\[ht\][\s\S]*?\\end{table}'
    matches = list(re.finditer(pattern2, content))
    for i, match in enumerate(matches):
        if "Efficiency metrics" in match.group(0):
            old_table = match.group(0)
            print(f"Found table at match {i}")
            content = content.replace(old_table, new_table)
            with open("main.tex", "w") as f:
                f.write(content)
            print("Table updated.")
            break
    else:
        print("Could not locate table.")