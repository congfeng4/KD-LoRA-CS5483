#!/usr/bin/env python3
"""
Generate LaTeX tables from pivot CSV files.
"""

import pandas as pd
import os
import argparse

# Mapping from task codes to display names
TASK_DISPLAY = {
    'cola': 'CoLA',
    'mnli': 'MNLI',
    'mrpc': 'MRPC',
    'qnli': 'QNLI',
    'qqp': 'QQP',
    'rte': 'RTE',
    'sst2': 'SST-2',
    'stsb': 'STS-B',
    'wnli': 'WNLI',
}

# Mapping from method codes to display names
METHOD_DISPLAY = {
    'lora': 'LoRA',
    'dora': 'DoRA',
    'mrlora': 'MrLoRA',
    'mrlora-rs': 'MrLoRA-RS',
    'olora': 'OLoRA',
    'rslora': 'RSLoRA',
}

def load_pivot_csv(filepath):
    """Load pivot CSV, set index to method."""
    df = pd.read_csv(filepath)
    df = df.set_index('method')
    return df

def format_value(val, max_val, decimal_places=3):
    """Format value with decimal places, bold if equals max_val."""
    if pd.isna(val):
        return '--'
    fmt = f"{val:.{decimal_places}f}"
    if abs(val - max_val) < 1e-9:  # consider equal within precision
        return f"\\textbf{{{fmt}}}"
    else:
        return fmt

def generate_latex_table(df, model_family, caption=None, label=None):
    """
    Generate LaTeX tabular code.
    Rows = methods, columns = tasks.
    """
    # Ensure methods and tasks order
    method_order = ['lora', 'dora', 'mrlora', 'mrlora-rs', 'olora', 'rslora']
    method_order = [m for m in method_order if m in df.index]
    task_order = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
    task_order = [t for t in task_order if t in df.columns]
    
    df = df.loc[method_order, task_order]
    
    # Compute max per column (ignore NaN)
    max_per_column = df.max()
    
    # Build header
    header = " & ".join([TASK_DISPLAY.get(t, t) for t in task_order])
    header = "Method & " + header + " \\\\"
    
    rows = []
    for method in method_order:
        row_vals = []
        for task in task_order:
            val = df.loc[method, task]
            row_vals.append(format_value(val, max_per_column[task]))
        method_display = METHOD_DISPLAY.get(method, method)
        row = method_display + " & " + " & ".join(row_vals) + " \\\\"
        rows.append(row)
    
    # LaTeX tabular environment
    num_cols = len(task_order) + 1
    col_spec = "l" + "c" * len(task_order)
    tabular = "\\begin{tabular}{" + col_spec + "}\n"
    tabular += "\\toprule\n"
    tabular += header + "\n"
    tabular += "\\midrule\n"
    tabular += "\n".join(rows) + "\n"
    tabular += "\\bottomrule\n"
    tabular += "\\end{tabular}"
    
    # Wrap in table environment with caption and label
    if caption is None:
        caption = f"Performance of LoRA variants on GLUE tasks ({model_family} family)"
    if label is None:
        label = f"tab:{model_family}_glue"
    
    table_env = "\\begin{table}[ht]\n"
    table_env += "\\centering\n"
    table_env += tabular + "\n"
    table_env += f"\\caption{{{caption}}}\n"
    table_env += f"\\label{{{label}}}\n"
    table_env += "\\end{table}"
    
    return table_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output", help="Directory containing pivot CSV files")
    parser.add_argument("--output_dir", default="tables", help="Directory to write LaTeX tables")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    families = ['bert', 'roberta', 'deberta']
    for fam in families:
        csv_path = os.path.join(args.input_dir, f"pivot_{fam}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue
        df = load_pivot_csv(csv_path)
        latex = generate_latex_table(df, fam)
        output_path = os.path.join(args.output_dir, f"table_{fam}.tex")
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"Generated {output_path}")
    
    # Also generate a combined LaTeX file with all three tables
    combined = ""
    for fam in families:
        csv_path = os.path.join(args.input_dir, f"pivot_{fam}.csv")
        if os.path.exists(csv_path):
            df = load_pivot_csv(csv_path)
            latex = generate_latex_table(df, fam)
            combined += latex + "\n\n"
    combined_path = os.path.join(args.output_dir, "all_tables.tex")
    with open(combined_path, 'w') as f:
        f.write(combined)
    print(f"Generated combined table at {combined_path}")

if __name__ == "__main__":
    main()