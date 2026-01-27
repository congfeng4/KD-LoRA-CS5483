#!/usr/bin/env python3
"""
Create Table I for KD-LoRA paper comparing FFT, LoRA (MrLoRA), and KD-LoRA (MrLoRA)
across three encoder-only LLMs (bert, roberta, deberta-v3) on GLUE tasks.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Define GLUE tasks and their official metrics
GLUE_TASKS = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']
TASK_METRIC_MAP = {
    'cola': 'eval_matthews_correlation',
    'sst2': 'eval_accuracy',
    'mrpc': 'eval_accuracy',  # MRPC uses accuracy/F1, JSON has accuracy
    'qqp': 'eval_accuracy',   # QQP uses accuracy/F1
    'stsb': 'eval_pearson',   # STS-B uses Pearson correlation
    'mnli': 'matched_accuracy',  # Will handle mismatched separately
    'qnli': 'eval_accuracy',
    'rte': 'eval_accuracy',
    'wnli': 'eval_accuracy'
}

# Model families (encoder-only LLMs)
MODEL_FAMILIES = ['bert', 'roberta', 'deberta']

# Finetuning strategies
STRATEGIES = ['FFT', 'LoRA', 'KD-LoRA']

def dictor(data: Dict, path: str, default: Any = None) -> Any:
    """Simple implementation of dictor to access nested dict keys."""
    keys = path.split('.')
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def extract_experiment_data(json_file: Path) -> List[Dict]:
    """Extract experiment data from a metrics.json file.
    Adapted from summarize.ipynb but with strategy mapping."""
    variant = json_file.relative_to('results').parts[0]
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return []
    
    # Determine strategy based on variant and peft method
    peft_method = dictor(data, 'args.peft', '')
    
    # Map to Table I strategies
    if variant == 'fft':
        strategy = 'FFT'
    elif variant == 'lora' and (peft_method == 'mrlora' or peft_method == 'mrlora-rs'):
        strategy = 'LoRA'
    elif variant == 'kd-lora' and (peft_method == 'mrlora' or peft_method == 'mrlora-rs'):
        strategy = 'KD-LoRA'
    else:
        # Not part of Table I comparison
        return []
    
    # Extract metadata
    model_family = dictor(data, 'args.model_family', '')
    task = dictor(data, 'args.task', '')
    seed = dictor(data, 'args.seed', 0)
    
    # Skip if model_family not in our target list
    if model_family not in MODEL_FAMILIES:
        return []
    
    # Extract metrics
    results = []
    
    # For MNLI, handle matched and mismatched separately
    if task == 'mnli':
        matched_acc = data.get('matched_accuracy')
        mismatched_acc = data.get('mismatched_accuracy')
        
        if matched_acc is not None:
            results.append({
                'Model Family': model_family,
                'Strategy': strategy,
                'Task': 'mnli_m',
                'Metric Value': matched_acc,
                'Metric Name': 'matched_accuracy',
                'Seed': seed,
                'Variant': variant,
                'PEFT Method': peft_method,
                'File': str(json_file)
            })
        
        if mismatched_acc is not None:
            results.append({
                'Model Family': model_family,
                'Strategy': strategy,
                'Task': 'mnli_mm',
                'Metric Value': mismatched_acc,
                'Metric Name': 'mismatched_accuracy',
                'Seed': seed,
                'Variant': variant,
                'PEFT Method': peft_method,
                'File': str(json_file)
            })
    else:
        # For other tasks, find the appropriate metric
        metric_name = TASK_METRIC_MAP.get(task)
        if metric_name and metric_name in data:
            metric_value = data[metric_name]
        else:
            # Fallback: search for any known metric
            for key in ['eval_accuracy', 'eval_matthews_correlation', 'eval_pearson', 'eval_spearman']:
                if key in data:
                    metric_value = data[key]
                    metric_name = key
                    break
            else:
                # No metric found
                return []
        
        results.append({
            'Model Family': model_family,
            'Strategy': strategy,
            'Task': task,
            'Metric Value': metric_value,
            'Metric Name': metric_name,
            'Seed': seed,
            'Variant': variant,
            'PEFT Method': peft_method,
            'File': str(json_file)
        })
    
    return results

def collect_all_data(results_dir: str = 'results') -> pd.DataFrame:
    """Collect all data relevant for Table I."""
    all_records = []
    
    # Find all JSON files
    json_files = list(Path(results_dir).rglob('*.json'))
    print(f"Found {len(json_files)} JSON files total")
    
    for json_file in json_files:
        records = extract_experiment_data(json_file)
        all_records.extend(records)
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    if df.empty:
        print("No data collected!")
        return df
    
    print(f"Collected {len(df)} data records")
    print(f"Unique strategies found: {df['Strategy'].unique().tolist()}")
    print(f"Unique model families found: {df['Model Family'].unique().tolist()}")
    print(f"Unique tasks found: {df['Task'].unique().tolist()}")
    
    return df

def create_table_i_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create Table I DataFrame with multi-index columns."""
    if raw_df.empty:
        return pd.DataFrame()
    
    # Average metric values across seeds for each combination
    grouped = raw_df.groupby(['Task', 'Model Family', 'Strategy'], as_index=False)
    avg_df = grouped.agg({
        'Metric Value': 'mean',
        'Seed': 'count'
    }).rename(columns={'Seed': 'Num_Seeds'})
    
    # Pivot to create multi-index columns
    # First, create a DataFrame with Task as index and multi-index columns
    pivot_df = avg_df.pivot_table(
        index='Task',
        columns=['Model Family', 'Strategy'],
        values='Metric Value'
    )
    
    # Reorder columns to match desired structure
    column_tuples = []
    for model in MODEL_FAMILIES:
        for strategy in STRATEGIES:
            column_tuples.append((model, strategy))
    
    # Reindex columns to ensure all combinations exist (fill with NaN)
    pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_tuples(column_tuples, names=['Model Family', 'Strategy']))
    
    # Sort rows: GLUE tasks in standard order, then MNLI_m, MNLI_mm
    task_order = GLUE_TASKS.copy()
    # Replace 'mnli' with 'mnli_m' and 'mnli_mm'
    task_order.remove('mnli')
    task_order.extend(['mnli_m', 'mnli_mm'])
    
    # Keep only tasks that exist in data
    existing_tasks = [t for t in task_order if t in pivot_df.index]
    pivot_df = pivot_df.loc[existing_tasks]
    
    # Add average row
    avg_row = {}
    for model in MODEL_FAMILIES:
        for strategy in STRATEGIES:
            values = []
            for task in pivot_df.index:
                val = pivot_df.loc[task, (model, strategy)]
                if not pd.isna(val):
                    values.append(val)
            avg_row[(model, strategy)] = np.mean(values) if values else np.nan
    
    avg_df_row = pd.DataFrame([avg_row], index=['Average'], columns=pivot_df.columns)
    pivot_df = pd.concat([pivot_df, avg_df_row])
    
    return pivot_df

def generate_latex_table(table_df: pd.DataFrame, caption: str = 'Performance of MrLoRA across GLUE tasks') -> str:
    """Generate LaTeX code for Table I."""
    if table_df.empty:
        return ""
    
    # Format values: 4 decimal places, replace NaN with 'N/A'
    formatted_df = table_df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f'{x:.4f}' if not pd.isna(x) else 'N/A')
    
    # Build LaTeX
    latex_lines = []
    latex_lines.append('\\begin{table}[htbp]')
    latex_lines.append('  \\centering')
    latex_lines.append('  \\caption{' + caption + '}')
    latex_lines.append('  \\label{tab:table_i}')
    latex_lines.append('  \\begin{tabular}{l' + 'c' * len(table_df.columns) + '}')
    latex_lines.append('    \\toprule')
    
    # Column headers (multi-level)
    col_header1 = 'Task'
    for model in MODEL_FAMILIES:
        col_header1 += ' & \\multicolumn{3}{c}{' + model + '}'
    col_header1 += ' \\\\'
    latex_lines.append('    ' + col_header1)
    
    col_header2 = ''
    for _ in MODEL_FAMILIES:
        for strategy in STRATEGIES:
            col_header2 += ' & ' + strategy
    col_header2 += ' \\\\'
    latex_lines.append('    ' + col_header2)
    latex_lines.append('    \\midrule')
    
    # Data rows
    for idx, row in formatted_df.iterrows():
        row_str = str(idx)
        for model in MODEL_FAMILIES:
            for strategy in STRATEGIES:
                row_str += ' & ' + row[(model, strategy)]
        row_str += ' \\\\'
        latex_lines.append('    ' + row_str)
    
    latex_lines.append('    \\bottomrule')
    latex_lines.append('  \\end{tabular}')
    latex_lines.append('\\end{table}')
    
    return '\n'.join(latex_lines)

def plot_strategy_comparison(table_df: pd.DataFrame):
    """Generate bar plots comparing strategies for each model family."""
    # Remove Average row for plots
    plot_df = table_df.drop('Average', errors='ignore')
    
    # Create subplots for each model family
    fig, axes = plt.subplots(1, len(MODEL_FAMILIES), figsize=(15, 5), sharey=True)
    if len(MODEL_FAMILIES) == 1:
        axes = [axes]
    
    for idx, model in enumerate(MODEL_FAMILIES):
        ax = axes[idx]
        
        # Extract data for this model
        model_data = {}
        for strategy in STRATEGIES:
            if (model, strategy) in plot_df.columns:
                values = plot_df[(model, strategy)].values
                # Remove NaN for plotting
                model_data[strategy] = values[~np.isnan(values)]
        
        # Create bar positions
        tasks = plot_df.index.tolist()
        x = np.arange(len(tasks))
        width = 0.25
        
        # Plot bars for each strategy
        for i, (strategy, values) in enumerate(model_data.items()):
            offset = (i - 1) * width  # Center FFT, left LoRA, right KD-LoRA
            bars = ax.bar(x + offset, values, width, label=strategy, alpha=0.8)
            
            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Task')
        ax.set_ylabel('Metric Score')
        ax.set_title(f'{model.capitalize()}')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Strategy Comparison Across GLUE Tasks', fontsize=16)
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=150)
    plt.close()
    print("Strategy comparison plot saved to strategy_comparison.png")

def plot_heatmap(table_df: pd.DataFrame):
    """Generate heatmap of Table I."""
    plt.figure(figsize=(12, 8))
    
    # Convert to numeric for heatmap (NaN will be white)
    heatmap_data = table_df.copy()
    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    
    # Create mask for NaN values
    mask = heatmap_data.isna()
    
    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Metric Score'}, linewidths=0.5,
                mask=mask, annot_kws={'size': 8})
    
    plt.title('Table I: Performance Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig('table_i_heatmap.png', dpi=150)
    plt.close()
    print("Heatmap saved to table_i_heatmap.png")

def main():
    print("=" * 80)
    print("KD-LoRA Table I Generator")
    print("Comparing FFT, LoRA (MrLoRA), and KD-LoRA (MrLoRA)")
    print("across bert, roberta, deberta-v3 on GLUE tasks")
    print("=" * 80)
    
    # Step 1: Collect data
    print("\n1. Collecting data from results directory...")
    raw_df = collect_all_data()
    
    if raw_df.empty:
        print("No data found for Table I!")
        return
    
    # Step 2: Create Table I DataFrame
    print("\n2. Creating Table I DataFrame...")
    table_df = create_table_i_dataframe(raw_df)
    
    # Display table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print("\n" + "=" * 80)
    print("TABLE I")
    print("=" * 80)
    print(table_df.round(4))
    
    # Save to CSV
    csv_path = 'table_i_results.csv'
    table_df.to_csv(csv_path)
    print(f"\nTable saved to {csv_path}")
    
    # Step 3: Generate LaTeX
    print("\n3. Generating LaTeX table...")
    latex_table = generate_latex_table(table_df)
    latex_path = 'table_i_latex.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {latex_path}")
    
    # Step 4: Generate visualizations
    print("\n4. Generating visualizations...")
    try:
        plot_strategy_comparison(table_df)
        plot_heatmap(table_df)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    # Step 5: Data availability report
    print("\n5. Data availability report:")
    print("-" * 40)
    
    total_cells = len(table_df.index) * len(table_df.columns)
    missing_cells = table_df.isna().sum().sum()
    available_cells = total_cells - missing_cells
    
    print(f"Total cells in table: {total_cells}")
    print(f"Available cells: {available_cells} ({available_cells/total_cells*100:.1f}%)")
    print(f"Missing cells: {missing_cells} ({missing_cells/total_cells*100:.1f}%)")
    
    # Print missing combinations
    if missing_cells > 0:
        print("\nMissing combinations:")
        for task in table_df.index:
            for model in MODEL_FAMILIES:
                for strategy in STRATEGIES:
                    if pd.isna(table_df.loc[task, (model, strategy)]):
                        print(f"  {task} - {model} - {strategy}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()