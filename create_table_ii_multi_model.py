#!/usr/bin/env python3
"""
Create Table II equivalents for all model families (BERT, RoBERTa, DeBERTa) 
comparing FFT baseline with different LoRA variants across GLUE tasks.

For each model family, generate:
- Table IIa: LoRA-only strategy (type=2) with FFT baseline
- Table IIb: KD-LoRA strategy (type=1) with FFT baseline
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

# Model families to analyze
MODEL_FAMILIES = ['bert', 'roberta', 'deberta']

# PEFT variants to include (excluding 'lora' which is used for FFT)
PEFT_VARIANTS = ['adalora', 'dora', 'lora', 'mrlora', 'olora', 'rslora']

# Finetuning strategies for Table II
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

def extract_experiment_data(json_file: Path, target_model_family: str = None) -> List[Dict]:
    """Extract experiment data from a metrics.json file for Table II."""
    variant = json_file.relative_to('results').parts[0]
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return []
    
    # Get experiment type from args.type
    exp_type = dictor(data, 'args.type', -1)
    
    # Extract metadata
    model_family = dictor(data, 'args.model_family', '')
    task = dictor(data, 'args.task', '')
    seed = dictor(data, 'args.seed', 0)
    peft_method = dictor(data, 'args.peft', '')
    
    # Skip if model_family doesn't match target
    if target_model_family is not None and model_family != target_model_family:
        return []
    
    # Determine strategy based on variant and type
    if variant == 'fft' or exp_type == 0:
        strategy = 'FFT'
        # For FFT, peft_method might be empty or 'lora', but we treat it as FFT baseline
        peft_variant = 'FFT'
    elif variant == 'lora' and exp_type in [2, 3]:
        strategy = 'LoRA'
        peft_variant = peft_method
    elif variant == 'kd-lora' and exp_type in [1, 3]:
        strategy = 'KD-LoRA'
        peft_variant = peft_method
    else:
        # Not part of Table II comparison
        return []
    
    # Skip if peft_variant not in our target list (for LoRA/KD-LoRA)
    if strategy in ['LoRA', 'KD-LoRA'] and peft_variant not in PEFT_VARIANTS:
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
                'PEFT Variant': peft_variant,
                'Task': 'mnli_m',
                'Metric Value': matched_acc,
                'Metric Name': 'matched_accuracy',
                'Seed': seed,
                'Variant': variant,
                'PEFT Method': peft_method,
                'Type': exp_type,
                'File': str(json_file)
            })
        
        if mismatched_acc is not None:
            results.append({
                'Model Family': model_family,
                'Strategy': strategy,
                'PEFT Variant': peft_variant,
                'Task': 'mnli_mm',
                'Metric Value': mismatched_acc,
                'Metric Name': 'mismatched_accuracy',
                'Seed': seed,
                'Variant': variant,
                'PEFT Method': peft_method,
                'Type': exp_type,
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
            'PEFT Variant': peft_variant,
            'Task': task,
            'Metric Value': metric_value,
            'Metric Name': metric_name,
            'Seed': seed,
            'Variant': variant,
            'PEFT Method': peft_method,
            'Type': exp_type,
            'File': str(json_file)
        })
    
    return results

def collect_all_data(results_dir: str = 'results') -> pd.DataFrame:
    """Collect all data relevant for Table II for all model families."""
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
    print(f"Model families found: {sorted(df['Model Family'].unique().tolist())}")
    print(f"Unique strategies found: {df['Strategy'].unique().tolist()}")
    print(f"Unique PEFT variants found: {df['PEFT Variant'].unique().tolist()}")
    print(f"Unique tasks found: {df['Task'].unique().tolist()}")
    
    return df

def create_table_iia_dataframe(raw_df: pd.DataFrame, model_family: str) -> pd.DataFrame:
    """Create Table IIa DataFrame: FFT baseline + LoRA variants (LoRA-only strategy)."""
    # Filter for specific model family
    model_df = raw_df[raw_df['Model Family'] == model_family].copy()
    
    # Filter for FFT and LoRA (type=2) only
    table_df = model_df[model_df['Strategy'].isin(['FFT', 'LoRA'])].copy()
    
    if table_df.empty:
        return pd.DataFrame()
    
    # Average metric values across seeds for each combination
    grouped = table_df.groupby(['Task', 'Strategy', 'PEFT Variant'], as_index=False)
    avg_df = grouped.agg({
        'Metric Value': 'mean',
        'Seed': 'count'
    }).rename(columns={'Seed': 'Num_Seeds'})
    
    # Create separate DataFrames for FFT and LoRA variants
    fft_df = avg_df[avg_df['Strategy'] == 'FFT'].copy()
    lora_df = avg_df[avg_df['Strategy'] == 'LoRA'].copy()
    
    # Pivot LoRA variants
    lora_pivot = lora_df.pivot_table(
        index='Task',
        columns='PEFT Variant',
        values='Metric Value'
    )
    
    # Add FFT as a column
    if not fft_df.empty:
        fft_series = fft_df.set_index('Task')['Metric Value']
        lora_pivot['FFT'] = fft_series
    
    # Reorder columns: FFT first, then PEFT variants in consistent order
    column_order = ['FFT'] + [v for v in PEFT_VARIANTS if v in lora_pivot.columns]
    lora_pivot = lora_pivot[column_order]
    
    # Sort rows: GLUE tasks in standard order, then MNLI_m, MNLI_mm
    task_order = GLUE_TASKS.copy()
    task_order.remove('mnli')
    task_order.extend(['mnli_m', 'mnli_mm'])
    
    # Keep only tasks that exist in data
    existing_tasks = [t for t in task_order if t in lora_pivot.index]
    lora_pivot = lora_pivot.loc[existing_tasks]
    
    # Add average row
    avg_row = {}
    for col in lora_pivot.columns:
        values = lora_pivot[col].dropna()
        avg_row[col] = np.mean(values) if len(values) > 0 else np.nan
    
    avg_df_row = pd.DataFrame([avg_row], index=['Average'], columns=lora_pivot.columns)
    lora_pivot = pd.concat([lora_pivot, avg_df_row])
    
    return lora_pivot

def create_table_iib_dataframe(raw_df: pd.DataFrame, model_family: str) -> pd.DataFrame:
    """Create Table IIb DataFrame: FFT baseline + LoRA variants (KD-LoRA strategy)."""
    # Filter for specific model family
    model_df = raw_df[raw_df['Model Family'] == model_family].copy()
    
    # Filter for FFT and KD-LoRA (type=1) only
    table_df = model_df[model_df['Strategy'].isin(['FFT', 'KD-LoRA'])].copy()
    
    if table_df.empty:
        return pd.DataFrame()
    
    # Average metric values across seeds for each combination
    grouped = table_df.groupby(['Task', 'Strategy', 'PEFT Variant'], as_index=False)
    avg_df = grouped.agg({
        'Metric Value': 'mean',
        'Seed': 'count'
    }).rename(columns={'Seed': 'Num_Seeds'})
    
    # Create separate DataFrames for FFT and KD-LoRA variants
    fft_df = avg_df[avg_df['Strategy'] == 'FFT'].copy()
    kd_lora_df = avg_df[avg_df['Strategy'] == 'KD-LoRA'].copy()
    
    # Pivot KD-LoRA variants
    kd_lora_pivot = kd_lora_df.pivot_table(
        index='Task',
        columns='PEFT Variant',
        values='Metric Value'
    )
    
    # Add FFT as a column
    if not fft_df.empty:
        fft_series = fft_df.set_index('Task')['Metric Value']
        kd_lora_pivot['FFT'] = fft_series
    
    # Reorder columns: FFT first, then PEFT variants in consistent order
    column_order = ['FFT'] + [v for v in PEFT_VARIANTS if v in kd_lora_pivot.columns]
    kd_lora_pivot = kd_lora_pivot[column_order]
    
    # Sort rows: GLUE tasks in standard order, then MNLI_m, MNLI_mm
    task_order = GLUE_TASKS.copy()
    task_order.remove('mnli')
    task_order.extend(['mnli_m', 'mnli_mm'])
    
    # Keep only tasks that exist in data
    existing_tasks = [t for t in task_order if t in kd_lora_pivot.index]
    kd_lora_pivot = kd_lora_pivot.loc[existing_tasks]
    
    # Add average row
    avg_row = {}
    for col in kd_lora_pivot.columns:
        values = kd_lora_pivot[col].dropna()
        avg_row[col] = np.mean(values) if len(values) > 0 else np.nan
    
    avg_df_row = pd.DataFrame([avg_row], index=['Average'], columns=kd_lora_pivot.columns)
    kd_lora_pivot = pd.concat([kd_lora_pivot, avg_df_row])
    
    return kd_lora_pivot

def generate_latex_table(table_df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX code for Table IIa or IIb."""
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
    latex_lines.append('  \\label{' + label + '}')
    
    # Determine column format
    n_cols = len(table_df.columns)
    latex_lines.append('  \\begin{tabular}{l' + 'c' * n_cols + '}')
    latex_lines.append('    \\toprule')
    
    # Column headers
    col_header = 'Task'
    for col in formatted_df.columns:
        if col == 'FFT':
            col_header += ' & FFT'
        else:
            col_header += ' & ' + col
    col_header += ' \\\\'
    latex_lines.append('    ' + col_header)
    latex_lines.append('    \\midrule')
    
    # Data rows
    for idx, row in formatted_df.iterrows():
        row_str = str(idx)
        for col in formatted_df.columns:
            row_str += ' & ' + row[col]
        row_str += ' \\\\'
        latex_lines.append('    ' + row_str)
    
    latex_lines.append('    \\bottomrule')
    latex_lines.append('  \\end{tabular}')
    latex_lines.append('\\end{table}')
    
    return '\n'.join(latex_lines)

def plot_heatmap(table_df: pd.DataFrame, output_path: str, title: str):
    """Generate heatmap of Table II."""
    plt.figure(figsize=(max(10, len(table_df.columns) * 1.2), max(6, len(table_df.index) * 0.6)))
    
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
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {output_path}")

def plot_model_comparison(model_tables: Dict[str, Dict[str, pd.DataFrame]]):
    """Generate comparison plots across model families."""
    # Extract average rows for each model and strategy
    comparison_data = []
    
    for model_family in MODEL_FAMILIES:
        if model_family not in model_tables:
            continue
        
        tables = model_tables[model_family]
        
        # LoRA-only averages
        if 'iia' in tables and not tables['iia'].empty and 'Average' in tables['iia'].index:
            avg_iia = tables['iia'].loc['Average']
            for variant in PEFT_VARIANTS:
                if variant in avg_iia and not pd.isna(avg_iia[variant]):
                    comparison_data.append({
                        'Model Family': model_family,
                        'Strategy': 'LoRA-only',
                        'Variant': variant,
                        'Average Score': avg_iia[variant],
                        'FFT Baseline': avg_iia.get('FFT', np.nan)
                    })
        
        # KD-LoRA averages
        if 'iib' in tables and not tables['iib'].empty and 'Average' in tables['iib'].index:
            avg_iib = tables['iib'].loc['Average']
            for variant in PEFT_VARIANTS:
                if variant in avg_iib and not pd.isna(avg_iib[variant]):
                    comparison_data.append({
                        'Model Family': model_family,
                        'Strategy': 'KD-LoRA',
                        'Variant': variant,
                        'Average Score': avg_iib[variant],
                        'FFT Baseline': avg_iib.get('FFT', np.nan)
                    })
    
    if not comparison_data:
        print("Cannot generate model comparison plots: missing data")
        return
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Plot 1: Bar plot of average scores by model family and variant
    plt.figure(figsize=(14, 8))
    
    # Pivot for grouped bar plot
    pivot_df = comp_df.pivot_table(
        index=['Model Family', 'Variant'],
        columns='Strategy',
        values='Average Score'
    ).reset_index()
    
    # Create grouped bar plot
    x = np.arange(len(MODEL_FAMILIES) * len(PEFT_VARIANTS))
    width = 0.35
    
    # Plot for each model family and variant
    for i, model in enumerate(MODEL_FAMILIES):
        model_df = pivot_df[pivot_df['Model Family'] == model]
        model_indices = x[i*len(PEFT_VARIANTS):(i+1)*len(PEFT_VARIANTS)]
        
        if 'LoRA-only' in pivot_df.columns:
            lora_values = [model_df[model_df['Variant'] == v]['LoRA-only'].iloc[0] 
                          if not model_df[model_df['Variant'] == v].empty and 'LoRA-only' in model_df.columns 
                          else np.nan for v in PEFT_VARIANTS]
            plt.bar(model_indices - width/2, lora_values, width, 
                    label='LoRA-only' if i == 0 else '', alpha=0.8)
        
        if 'KD-LoRA' in pivot_df.columns:
            kd_lora_values = [model_df[model_df['Variant'] == v]['KD-LoRA'].iloc[0] 
                             if not model_df[model_df['Variant'] == v].empty and 'KD-LoRA' in model_df.columns 
                             else np.nan for v in PEFT_VARIANTS]
            plt.bar(model_indices + width/2, kd_lora_values, width, 
                    label='KD-LoRA' if i == 0 else '', alpha=0.8)
    
    # Set x-axis labels
    x_labels = []
    for model in MODEL_FAMILIES:
        x_labels.extend([f'{model}\n{v}' for v in PEFT_VARIANTS])
    
    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.xlabel('Model Family and PEFT Variant')
    plt.ylabel('Average Metric Score')
    plt.title('Average Performance Across GLUE Tasks: Model Family Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_family_comparison.png', dpi=150)
    plt.close()
    print("Model family comparison plot saved to model_family_comparison.png")
    
    # Save comparison data to CSV
    comp_df.to_csv('model_comparison_data.csv', index=False)
    print("Comparison data saved to model_comparison_data.csv")

def main():
    print("=" * 80)
    print("Multi-Model Table II Generator")
    print("Comparing FFT baseline with LoRA variants for BERT, RoBERTa, DeBERTa on GLUE tasks")
    print("=" * 80)
    
    # Step 1: Collect data for all model families
    print("\n1. Collecting data from results directory...")
    raw_df = collect_all_data()
    
    if raw_df.empty:
        print("No data found for Table II!")
        return
    
    # Dictionary to store tables for each model family
    model_tables = {}
    
    # Process each model family
    for model_family in MODEL_FAMILIES:
        print(f"\n{'='*60}")
        print(f"PROCESSING MODEL FAMILY: {model_family.upper()}")
        print(f"{'='*60}")
        
        # Check if data exists for this model family
        model_data = raw_df[raw_df['Model Family'] == model_family]
        if model_data.empty:
            print(f"No data found for {model_family}. Skipping.")
            continue
        
        print(f"Found {len(model_data)} records for {model_family}")
        
        # Create output directory for this model family
        output_dir = f"table_ii_{model_family}"
        Path(output_dir).mkdir(exist_ok=True)
        
        # Step 2: Create Table IIa (LoRA-only)
        print(f"\n2. Creating Table IIa for {model_family} (FFT + LoRA variants)...")
        table_iia = create_table_iia_dataframe(raw_df, model_family)
        
        if not table_iia.empty:
            print(f"\nTABLE IIa for {model_family.upper()}: FFT Baseline vs LoRA Variants (LoRA-only strategy)")
            print(table_iia.round(4))
            
            # Save to CSV
            csv_path = f'{output_dir}/table_iia_results.csv'
            table_iia.to_csv(csv_path)
            print(f"Table IIa saved to {csv_path}")
            
            # Generate LaTeX
            latex_table = generate_latex_table(
                table_iia,
                f'Performance of LoRA variants (LoRA-only strategy) for {model_family.upper()} on GLUE tasks',
                f'tab:table_iia_{model_family}'
            )
            latex_path = f'{output_dir}/table_iia_latex.tex'
            with open(latex_path, 'w') as f:
                f.write(latex_table)
            print(f"LaTeX table saved to {latex_path}")
            
            # Generate heatmap
            plot_heatmap(
                table_iia,
                f'{output_dir}/table_iia_heatmap.png',
                f'Table IIa: {model_family.upper()} - FFT Baseline vs LoRA Variants (LoRA-only)'
            )
            
            model_tables.setdefault(model_family, {})['iia'] = table_iia
        else:
            print(f"No data for Table IIa for {model_family}!")
            model_tables.setdefault(model_family, {})['iia'] = pd.DataFrame()
        
        # Step 3: Create Table IIb (KD-LoRA)
        print(f"\n3. Creating Table IIb for {model_family} (FFT + KD-LoRA variants)...")
        table_iib = create_table_iib_dataframe(raw_df, model_family)
        
        if not table_iib.empty:
            print(f"\nTABLE IIb for {model_family.upper()}: FFT Baseline vs LoRA Variants (KD-LoRA strategy)")
            print(table_iib.round(4))
            
            # Save to CSV
            csv_path = f'{output_dir}/table_iib_results.csv'
            table_iib.to_csv(csv_path)
            print(f"Table IIb saved to {csv_path}")
            
            # Generate LaTeX
            latex_table = generate_latex_table(
                table_iib,
                f'Performance of LoRA variants (KD-LoRA strategy) for {model_family.upper()} on GLUE tasks',
                f'tab:table_iib_{model_family}'
            )
            latex_path = f'{output_dir}/table_iib_latex.tex'
            with open(latex_path, 'w') as f:
                f.write(latex_table)
            print(f"LaTeX table saved to {latex_path}")
            
            # Generate heatmap
            plot_heatmap(
                table_iib,
                f'{output_dir}/table_iib_heatmap.png',
                f'Table IIb: {model_family.upper()} - FFT Baseline vs LoRA Variants (KD-LoRA)'
            )
            
            model_tables.setdefault(model_family, {})['iib'] = table_iib
        else:
            print(f"No data for Table IIb for {model_family}!")
            model_tables.setdefault(model_family, {})['iib'] = pd.DataFrame()
    
    # Step 4: Generate cross-model comparison plots
    print("\n4. Generating cross-model comparison plots...")
    plot_model_comparison(model_tables)
    
    # Step 5: Data availability report
    print("\n5. Data availability report:")
    print("-" * 40)
    
    for model_family in MODEL_FAMILIES:
        model_data = raw_df[raw_df['Model Family'] == model_family]
        if model_data.empty:
            print(f"{model_family:10s}: No data")
            continue
        
        # Count by strategy
        strategies = model_data['Strategy'].unique()
        print(f"\n{model_family.upper()}:")
        for strategy in ['FFT', 'LoRA', 'KD-LoRA']:
            if strategy in strategies:
                count = len(model_data[model_data['Strategy'] == strategy])
                print(f"  {strategy:10s}: {count} experiments")
        
        # Tasks with data
        tasks_with_data = sorted(model_data['Task'].unique())
        print(f"  Tasks: {len(tasks_with_data)} tasks: {tasks_with_data}")
    
    print("\n" + "=" * 80)
    print("Multi-model Table II generation complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()