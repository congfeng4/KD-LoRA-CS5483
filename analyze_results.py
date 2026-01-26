#!/usr/bin/env python3
"""
Analyze experimental results for KD-LoRA paper Table I.
Focus on mrlora variant across fft, lora, kd-lora strategies and bert, roberta, deberta-v3 models.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define GLUE task to metric mapping based on GLUE benchmark
TASK_METRIC_MAP = {
    'cola': 'eval_matthews_correlation',
    'sst2': 'eval_accuracy',
    'mrpc': 'eval_accuracy',  # Note: MRPC uses accuracy/F1, but JSON has accuracy
    'qqp': 'eval_accuracy',   # QQP uses accuracy/F1
    'stsb': 'eval_pearson',   # STS-B uses Pearson correlation (also has spearman)
    'mnli': 'matched_accuracy',  # Will handle mismatched separately
    'qnli': 'eval_accuracy',
    'rte': 'eval_accuracy',
    'wnli': 'eval_accuracy'
}

# Model families we care about (based on paper)
TARGET_MODEL_FAMILIES = ['bert', 'roberta', 'deberta']

# Finetuning strategies (variants)
STRATEGIES = ['fft', 'lora', 'kd-lora']

def extract_data_from_json(filepath: Path) -> Optional[Dict]:
    """Extract relevant data from a metrics.json file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    args = data.get('args', {})
    peft = args.get('peft', '')
    
    # We only care about mrlora for this analysis
    if peft != 'mrlora':
        return None
    
    task = args.get('task', '')
    model_family = args.get('model_family', '')
    variant = data.get('variant', '')
    seed = args.get('seed', 0)
    
    # Extract metric value based on task
    metric_value = None
    if task == 'mnli':
        matched = data.get('matched_accuracy')
        mismatched = data.get('mismatched_accuracy')
        metric_value = (matched, mismatched)
    else:
        metric_name = TASK_METRIC_MAP.get(task)
        if metric_name and metric_name in data:
            metric_value = data.get(metric_name)
        else:
            # Fallback: search for any known metric key
            for key in ['eval_accuracy', 'eval_matthews_correlation', 'eval_pearson', 'eval_spearman']:
                if key in data:
                    metric_value = data.get(key)
                    break
    
    if metric_value is None:
        print(f"Warning: No metric found for {task} in {filepath}")
        return None
    
    return {
        'task': task,
        'model_family': model_family,
        'variant': variant,
        'seed': seed,
        'metric_value': metric_value,
        'filepath': str(filepath)
    }

def collect_all_mrlora_data(results_dir: str = 'results') -> List[Dict]:
    """Collect all mrlora data from results directory."""
    all_data = []
    mrlora_files = []
    
    # Find all JSON files that might contain mrlora data
    # More efficient: check directory names first
    for variant_dir in Path(results_dir).iterdir():
        if not variant_dir.is_dir():
            continue
        
        variant_name = variant_dir.name
        if variant_name not in STRATEGIES:
            continue
        
        # Search for mrlora in subdirectories
        for json_file in variant_dir.rglob('*.json'):
            if 'mrlora' in str(json_file).lower():
                mrlora_files.append(json_file)
    
    print(f"Found {len(mrlora_files)} potential mrlora files")
    
    # Process each file
    for filepath in mrlora_files:
        data = extract_data_from_json(filepath)
        if data:
            all_data.append(data)
    
    return all_data

def create_table_i_dataframe(all_data: List[Dict]) -> pd.DataFrame:
    """Create Table I DataFrame with multi-index columns."""
    # Initialize dictionary to store results
    # Structure: task -> model_family -> strategy -> list of (seed, value) tuples
    results = {}
    
    for entry in all_data:
        task = entry['task']
        model_family = entry['model_family']
        variant = entry['variant']
        seed = entry['seed']
        metric_value = entry['metric_value']
        
        if model_family not in TARGET_MODEL_FAMILIES:
            continue
        
        if variant not in STRATEGIES:
            continue
        
        # Initialize nested dicts
        if task not in results:
            results[task] = {}
        if model_family not in results[task]:
            results[task][model_family] = {}
        if variant not in results[task][model_family]:
            results[task][model_family][variant] = []
        
        results[task][model_family][variant].append((seed, metric_value))
    
    # For MNLI, we need to handle matched and mismatched separately
    # Let's restructure: create separate rows for mnli_m and mnli_mm
    processed_results = {}
    
    for task, model_dict in results.items():
        if task == 'mnli':
            # Create entries for matched and mismatched
            for model_family, variant_dict in model_dict.items():
                for variant, seed_values in variant_dict.items():
                    # Calculate average across seeds
                    matched_values = []
                    mismatched_values = []
                    
                    for seed, (matched, mismatched) in seed_values:
                        if matched is not None:
                            matched_values.append(matched)
                        if mismatched is not None:
                            mismatched_values.append(mismatched)
                    
                    # Store averages
                    if matched_values:
                        if 'mnli_m' not in processed_results:
                            processed_results['mnli_m'] = {}
                        if model_family not in processed_results['mnli_m']:
                            processed_results['mnli_m'][model_family] = {}
                        processed_results['mnli_m'][model_family][variant] = np.mean(matched_values)
                    
                    if mismatched_values:
                        if 'mnli_mm' not in processed_results:
                            processed_results['mnli_mm'] = {}
                        if model_family not in processed_results['mnli_mm']:
                            processed_results['mnli_mm'][model_family] = {}
                        processed_results['mnli_mm'][model_family][variant] = np.mean(mismatched_values)
        else:
            # For other tasks, average across seeds
            for model_family, variant_dict in model_dict.items():
                for variant, seed_values in variant_dict.items():
                    values = [v for _, v in seed_values if v is not None]
                    if values:
                        if task not in processed_results:
                            processed_results[task] = {}
                        if model_family not in processed_results[task]:
                            processed_results[task][model_family] = {}
                        processed_results[task][model_family][variant] = np.mean(values)
    
    # Now create DataFrame with multi-index columns
    # Rows: tasks (including mnli_m, mnli_mm)
    # Columns: multi-index (model_family, strategy)
    
    # Get all tasks sorted
    all_tasks = sorted(processed_results.keys())
    
    # Create multi-index columns
    column_tuples = []
    for model in TARGET_MODEL_FAMILIES:
        for strategy in STRATEGIES:
            column_tuples.append((model, strategy))
    
    columns = pd.MultiIndex.from_tuples(column_tuples, names=['Model Family', 'Strategy'])
    
    # Create DataFrame
    df = pd.DataFrame(index=all_tasks, columns=columns, dtype=object)
    
    # Fill DataFrame
    for task in all_tasks:
        for model in TARGET_MODEL_FAMILIES:
            for strategy in STRATEGIES:
                value = processed_results.get(task, {}).get(model, {}).get(strategy, np.nan)
                df.loc[task, (model, strategy)] = value
    
    # Add average row
    avg_row = {}
    for model in TARGET_MODEL_FAMILIES:
        for strategy in STRATEGIES:
            # Average across all tasks (excluding NaN)
            values = []
            for task in all_tasks:
                val = df.loc[task, (model, strategy)]
                if not pd.isna(val):
                    values.append(val)
            avg_row[(model, strategy)] = np.mean(values) if values else np.nan
    
    # Add average row to DataFrame
    df.loc['Average'] = avg_row
    
    return df

def main():
    print("Collecting mrlora data from results directory...")
    all_data = collect_all_mrlora_data()
    
    if not all_data:
        print("No mrlora data found!")
        return
    
    print(f"Collected {len(all_data)} mrlora data entries")
    
    # Create Table I DataFrame
    print("\nCreating Table I DataFrame...")
    table_df = create_table_i_dataframe(all_data)
    
    # Display the table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print("\n" + "="*80)
    print("TABLE I: Performance of MrLoRA across GLUE tasks")
    print("Rows: GLUE tasks (including MNLI matched/mismatched)")
    print("Columns: Multi-index (Model Family × Finetuning Strategy)")
    print("Values: Metric scores (accuracy, correlation, etc.)")
    print("="*80 + "\n")
    
    print(table_df.round(4))
    
    # Save to CSV
    output_path = 'table_i_mrlora_results.csv'
    table_df.to_csv(output_path)
    print(f"\nTable saved to {output_path}")
    
    # Print summary of missing data
    print("\n" + "="*80)
    print("DATA AVAILABILITY SUMMARY")
    print("="*80)
    
    missing_count = 0
    total_cells = len(table_df.index) * len(table_df.columns)
    
    for task in table_df.index:
        for col in table_df.columns:
            if pd.isna(table_df.loc[task, col]):
                missing_count += 1
                if task != 'Average':  # Don't report missing averages
                    print(f"Missing: {task} - {col}")
    
    print(f"\nTotal missing cells: {missing_count}/{total_cells} ({missing_count/total_cells*100:.1f}%)")
    
    # Generate a simple heatmap visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create a numeric version for heatmap (replace NaN with -1 for visualization)
        heatmap_data = table_df.copy()
        for col in heatmap_data.columns:
            heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Metric Score'}, linewidths=0.5)
        plt.title('Table I: MrLoRA Performance Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig('table_i_heatmap.png', dpi=150)
        print("\nHeatmap saved to table_i_heatmap.png")
    except ImportError:
        print("\nNote: matplotlib/seaborn not available for visualization")

if __name__ == '__main__':
    main()