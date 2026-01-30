#!/usr/bin/env python3
"""
Generate LaTeX tables from CSV tables produced by aggregate_metrics.py.
Includes metric row, FFT row, parameter column, average column, bold best.
"""

import pandas as pd
import os
import argparse
import numpy as np

# Mapping from task to metric abbreviation string (for metric row)
TASK_METRIC_ABBR = {
    'cola': 'Mcc',
    'mnli': 'm/mm',
    'mrpc': 'Acc/F1',
    'qnli': 'Acc',
    'qqp': 'Acc/F1',
    'rte': 'Acc',
    'sst2': 'Acc',
    'stsb': 'Pearson/Spearman',
    'wnli': 'Acc',
}

# Mapping from method code to display name
METHOD_DISPLAY = {
    'fft': 'FFT',
    'lora': 'LoRA',
    'dora': 'DoRA',
    'mrlora': 'MrLoRA',
    'mrlora-rs': 'MrLoRA-RS',
    'olora': 'OLoRA',
    'rslora': 'RSLoRA',
}

def read_table_csv(csv_path):
    """Read CSV table, return DataFrame with multi-index columns? Actually simple."""
    df = pd.read_csv(csv_path)
    return df

def find_best_values(df, task_columns):
    """Find best value (max) for each task column across all rows."""
    best = {}
    for col in task_columns:
        # Extract numeric values (might have slash separated)
        values = []
        for cell in df[col]:
            if pd.isna(cell):
                continue
            # Cell may be string like "50.88" or "82.11/82.84"
            # Take the first value (primary metric) for comparison
            # Actually we need to compare primary metric (first before slash)
            parts = str(cell).split('/')
            try:
                val = float(parts[0])
                values.append(val)
            except:
                continue
        if values:
            best[col] = max(values)
        else:
            best[col] = None
    return best

def format_cell(cell, best_val, is_bold=True):
    """Format cell value, optionally bold if equals best_val."""
    if pd.isna(cell):
        return '--'
    cell_str = str(cell)
    # Determine if this cell's primary metric equals best_val
    parts = cell_str.split('/')
    try:
        primary = float(parts[0])
    except:
        return cell_str
    # Compare with tolerance
    if is_bold and best_val is not None and abs(primary - best_val) < 0.01:
        # Bold the entire cell (including slash-separated values)
        return f"\\textbf{{{cell_str}}}"
    else:
        return cell_str

def generate_latex_table(df, training_variant, model_family):
    """Generate LaTeX tabular code from DataFrame."""
    # Ensure method column is first, param column second
    # DataFrame columns: Method, # Params, cola, mnli, ..., Average
    method_col = 'Method'
    param_col = '# Params'
    task_cols = [c for c in df.columns if c not in [method_col, param_col, 'Average']]
    avg_col = 'Average'
    
    # Determine best values per task column (for bolding)
    best_vals = find_best_values(df, task_cols)
    
    # Build header with metric row
    header1 = "Method & \\# Params & " + " & ".join([TASK_METRIC_ABBR.get(t, t) for t in task_cols]) + " & Avg. \\\\"
    # Second header row with task names
    header2 = " & & " + " & ".join([t.upper() for t in task_cols]) + " & \\\\"
    
    # Build rows
    rows = []
    for _, row in df.iterrows():
        method = row[method_col]
        method_display = METHOD_DISPLAY.get(method, method)
        param_val = row[param_col]
        # Format param with 3 decimal places
        if pd.isna(param_val):
            param_str = ''
        else:
            param_str = f"{param_val:.3f}"
        row_cells = [method_display, param_str]
        # Add task cells
        for col in task_cols:
            cell = row[col]
            best = best_vals.get(col)
            row_cells.append(format_cell(cell, best))
        # Average cell
        avg_cell = row[avg_col]
        # Bold average? Maybe not
        if pd.isna(avg_cell):
            avg_str = ''
        else:
            avg_str = f"{avg_cell:.2f}"
        row_cells.append(avg_str)
        row_str = " & ".join(row_cells) + " \\\\"
        rows.append(row_str)
    
    # LaTeX tabular environment
    num_cols = len(task_cols) + 3  # Method, # Params, tasks, Average
    col_spec = "ll" + "c" * len(task_cols) + "c"
    tabular = "\\begin{tabular}{" + col_spec + "}\n"
    tabular += "\\toprule\n"
    tabular += header1 + "\n"
    tabular += header2 + "\n"
    tabular += "\\midrule\n"
    tabular += "\n".join(rows) + "\n"
    tabular += "\\bottomrule\n"
    tabular += "\\end{tabular}"
    
    # Wrap in table environment
    caption = f"Performance of LoRA variants on GLUE tasks ({model_family} family, {training_variant} training)"
    label = f"tab:{training_variant}_{model_family}_glue"
    
    table_env = "\\begin{table}[ht]\n"
    table_env += "\\centering\n"
    table_env += tabular + "\n"
    table_env += f"\\caption{{{caption}}}\n"
    table_env += f"\\label{{{label}}}\n"
    table_env += "\\end{table}"
    
    return table_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output", help="Directory containing table CSV files")
    parser.add_argument("--output_dir", default="tables_full", help="Directory to write LaTeX tables")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find table CSV files
    import glob
    csv_files = glob.glob(os.path.join(args.input_dir, "table_*.csv"))
    print(f"Found {len(csv_files)} table CSV files.")
    
    all_latex = ""
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        # Parse training_variant and model_family from filename
        # format: table_{training_variant}_{model_family}.csv
        basename = filename.replace('table_', '').replace('.csv', '')
        parts = basename.split('_')
        if len(parts) != 2:
            print(f"Warning: unexpected filename {filename}, skipping.")
            continue
        training_variant, model_family = parts
        df = read_table_csv(csv_path)
        latex = generate_latex_table(df, training_variant, model_family)
        output_path = os.path.join(args.output_dir, f"table_{training_variant}_{model_family}.tex")
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"Generated {output_path}")
        all_latex += latex + "\n\n"
    
    # Write combined LaTeX file
    combined_path = os.path.join(args.output_dir, "all_tables.tex")
    with open(combined_path, 'w') as f:
        f.write(all_latex)
    print(f"Combined tables written to {combined_path}")

if __name__ == "__main__":
    main()