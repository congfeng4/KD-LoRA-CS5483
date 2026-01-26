#!/usr/bin/env python3
"""
Analyze completion rate of experiments for KD-LoRA paper.
Counts completed experiments vs. expected experiments based on the loops in BERT_Distill_LoRA.py.
Outputs completion rates by model family, LoRA variant, training variant, and task.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Any
import warnings
warnings.filterwarnings('ignore')

# Constants from utils.py and BERT_Distill_LoRA.py
GLUE_TASKS = ["wnli", "rte", "qnli", "mrpc", "qqp", "stsb", "mnli", "cola", "sst2"]
MODEL_FAMILIES = ["bert", "roberta", "deberta"]
PEFT_VARIANTS = ["lora", "olora", "dora", "adalora", "rslora", "mrlora"]
SEEDS = [42, 123, 2024]

# Mapping from directory name to training variant (type)
VARIANT_TO_TYPE = {
    "fft": 0,      # FFT baseline
    "lora": 2,     # Teacher LoRA (LoRA-only)
    "kd-lora": 1,  # Student LoRA (KD-LoRA)
}

def parse_experiment_path(json_file: Path) -> Dict[str, Any]:
    """
    Parse experiment parameters from file path and JSON content.
    Returns dict with keys: variant, model_family, task, peft, seed, type.
    """
    # Extract parts from path
    parts = json_file.relative_to('results').parts
    
    # First part is variant (fft, lora, kd-lora)
    variant = parts[0]
    
    # Second part is like "task_wnli_bert_42"
    task_part = parts[1]
    task_parts = task_part.split('_')
    # Format: task_{task}_{model_family}_{seed}
    if len(task_parts) >= 4:
        task = task_parts[1]
        model_family = task_parts[2]
        seed = int(task_parts[3])
    else:
        # Fallback: try to extract from JSON
        task = model_family = seed = None
    
    # Fourth part is like "peft_mrlora_16_0.05_8"
    peft = None
    if len(parts) >= 4:
        peft_part = parts[3]
        if peft_part.startswith('peft_'):
            peft = peft_part.split('_')[1]
    
    # Try to read JSON for additional metadata
    exp_type = None
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        # Get type from args
        if 'args' in data and 'type' in data['args']:
            exp_type = data['args']['type']
        elif 'args' in data and isinstance(data['args'], dict) and 'type' in data['args']:
            exp_type = data['args']['type']
        # Also get model_family, task, peft, seed from args if missing
        if model_family is None and 'args' in data and 'model_family' in data['args']:
            model_family = data['args']['model_family']
        if task is None and 'args' in data and 'task' in data['args']:
            task = data['args']['task']
        if seed is None and 'args' in data and 'seed' in data['args']:
            seed = data['args']['seed']
        if peft is None and 'args' in data and 'peft' in data['args']:
            peft = data['args']['peft']
    except Exception as e:
        pass
    
    # If type not found in JSON, infer from variant
    if exp_type is None:
        exp_type = VARIANT_TO_TYPE.get(variant, -1)
    
    # For FFT experiments, set peft to 'FFT'
    if variant == 'fft':
        peft = 'FFT'
    
    return {
        'variant': variant,
        'model_family': model_family,
        'task': task,
        'peft': peft,
        'seed': seed,
        'type': exp_type,
        'file': str(json_file)
    }

def collect_completed_experiments(results_dir: str = 'results') -> pd.DataFrame:
    """
    Collect all completed experiments from results directory.
    Returns DataFrame with columns: variant, model_family, task, peft, seed, type.
    """
    completed_records = []
    
    # Find all JSON files
    json_files = list(Path(results_dir).rglob('*.json'))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            record = parse_experiment_path(json_file)
            # Skip if essential fields are missing
            if (record['model_family'] is None or record['task'] is None or 
                record['seed'] is None or record['peft'] is None):
                continue
            completed_records.append(record)
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
            continue
    
    df = pd.DataFrame(completed_records)
    
    if df.empty:
        print("No valid experiments found!")
        return df
    
    # Convert seed to int
    df['seed'] = df['seed'].astype(int)
    
    print(f"Collected {len(df)} valid experiment records")
    print(f"Variants: {df['variant'].unique().tolist()}")
    print(f"Model families: {df['model_family'].unique().tolist()}")
    print(f"Tasks: {df['task'].unique().tolist()}")
    print(f"PEFT variants: {df['peft'].unique().tolist()}")
    print(f"Seeds: {sorted(df['seed'].unique().tolist())}")
    print(f"Types: {sorted(df['type'].unique().tolist())}")
    
    return df

def generate_expected_experiments() -> pd.DataFrame:
    """
    Generate all expected experiments based on the loops in BERT_Distill_LoRA.py.
    Returns DataFrame with columns: variant, model_family, task, peft, seed, type.
    """
    expected_records = []
    
    # FFT experiments (type 0)
    for seed in SEEDS:
        for task in GLUE_TASKS:
            for model_family in MODEL_FAMILIES:
                expected_records.append({
                    'variant': 'fft',
                    'model_family': model_family,
                    'task': task,
                    'peft': 'FFT',
                    'seed': seed,
                    'type': 0
                })
    
    # Teacher LoRA experiments (type 2)
    for seed in SEEDS:
        for task in GLUE_TASKS:
            for model_family in MODEL_FAMILIES:
                for peft in PEFT_VARIANTS:
                    expected_records.append({
                        'variant': 'lora',
                        'model_family': model_family,
                        'task': task,
                        'peft': peft,
                        'seed': seed,
                        'type': 2
                    })
    
    # KD-LoRA experiments (type 1)
    for seed in SEEDS:
        for task in GLUE_TASKS:
            for model_family in MODEL_FAMILIES:
                for peft in PEFT_VARIANTS:
                    expected_records.append({
                        'variant': 'kd-lora',
                        'model_family': model_family,
                        'task': task,
                        'peft': peft,
                        'seed': seed,
                        'type': 1
                    })
    
    # Note: Type 3 (mrlora) is a special case; we treat mrlora experiments
    # with type=3 as part of the regular lora/kd-lora variants.
    # They will be matched based on variant, model_family, task, peft, seed.
    
    expected_df = pd.DataFrame(expected_records)
    print(f"Generated {len(expected_df)} expected experiment configurations")
    return expected_df

def compute_completion_rates(completed_df: pd.DataFrame, expected_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute completion rates across different dimensions.
    Returns dict with completion statistics.
    """
    # Create a unique key for each experiment
    def create_key(row):
        return (row['variant'], row['model_family'], row['task'], row['peft'], row['seed'])
    
    completed_df['key'] = completed_df.apply(create_key, axis=1)
    expected_df['key'] = expected_df.apply(create_key, axis=1)
    
    completed_keys = set(completed_df['key'].tolist())
    expected_keys = set(expected_df['key'].tolist())
    
    # Find missing experiments
    missing_keys = expected_keys - completed_keys
    missing_df = expected_df[expected_df['key'].isin(missing_keys)].copy()
    missing_df = missing_df.drop('key', axis=1)
    
    # Overall completion rate
    total_expected = len(expected_keys)
    total_completed = len(completed_keys)
    completion_rate = total_completed / total_expected * 100 if total_expected > 0 else 0
    
    # Completion by variant (training variant)
    variant_stats = {}
    for variant in ['fft', 'lora', 'kd-lora']:
        variant_expected = expected_df[expected_df['variant'] == variant]
        variant_completed = completed_df[completed_df['variant'] == variant]
        variant_rate = len(variant_completed) / len(variant_expected) * 100 if len(variant_expected) > 0 else 0
        variant_stats[variant] = {
            'expected': len(variant_expected),
            'completed': len(variant_completed),
            'rate': variant_rate
        }
    
    # Completion by model family
    model_stats = {}
    for model_family in MODEL_FAMILIES:
        model_expected = expected_df[expected_df['model_family'] == model_family]
        model_completed = completed_df[completed_df['model_family'] == model_family]
        model_rate = len(model_completed) / len(model_expected) * 100 if len(model_expected) > 0 else 0
        model_stats[model_family] = {
            'expected': len(model_expected),
            'completed': len(model_completed),
            'rate': model_rate
        }
    
    # Completion by task
    task_stats = {}
    for task in GLUE_TASKS:
        task_expected = expected_df[expected_df['task'] == task]
        task_completed = completed_df[completed_df['task'] == task]
        task_rate = len(task_completed) / len(task_expected) * 100 if len(task_expected) > 0 else 0
        task_stats[task] = {
            'expected': len(task_expected),
            'completed': len(task_completed),
            'rate': task_rate
        }
    
    # Completion by PEFT variant (excluding FFT)
    peft_stats = {}
    for peft in PEFT_VARIANTS + ['FFT']:
        peft_expected = expected_df[expected_df['peft'] == peft]
        peft_completed = completed_df[completed_df['peft'] == peft]
        peft_rate = len(peft_completed) / len(peft_expected) * 100 if len(peft_expected) > 0 else 0
        peft_stats[peft] = {
            'expected': len(peft_expected),
            'completed': len(peft_completed),
            'rate': peft_rate
        }
    
    # Completion by seed
    seed_stats = {}
    for seed in SEEDS:
        seed_expected = expected_df[expected_df['seed'] == seed]
        seed_completed = completed_df[completed_df['seed'] == seed]
        seed_rate = len(seed_completed) / len(seed_expected) * 100 if len(seed_expected) > 0 else 0
        seed_stats[seed] = {
            'expected': len(seed_expected),
            'completed': len(seed_completed),
            'rate': seed_rate
        }
    
    # Completion by type (training variant)
    type_stats = {}
    for exp_type in [0, 1, 2]:
        type_expected = expected_df[expected_df['type'] == exp_type]
        type_completed = completed_df[completed_df['type'] == exp_type]
        type_rate = len(type_completed) / len(type_expected) * 100 if len(type_expected) > 0 else 0
        type_name = {0: 'FFT', 1: 'KD-LoRA', 2: 'LoRA-only'}[exp_type]
        type_stats[type_name] = {
            'expected': len(type_expected),
            'completed': len(type_completed),
            'rate': type_rate
        }
    
    # Cross-tabulation: model_family × variant
    cross_model_variant = pd.crosstab(
        completed_df['model_family'], 
        completed_df['variant'],
        margins=True
    )
    
    # Cross-tabulation: task × variant
    cross_task_variant = pd.crosstab(
        completed_df['task'],
        completed_df['variant'],
        margins=True
    )
    
    # Cross-tabulation: peft × variant (excluding FFT)
    peft_variant_df = completed_df[completed_df['peft'] != 'FFT']
    cross_peft_variant = pd.crosstab(
        peft_variant_df['peft'],
        peft_variant_df['variant'],
        margins=True
    )
    
    return {
        'overall': {
            'expected': total_expected,
            'completed': total_completed,
            'rate': completion_rate,
            'missing_count': len(missing_keys)
        },
        'by_variant': variant_stats,
        'by_model_family': model_stats,
        'by_task': task_stats,
        'by_peft': peft_stats,
        'by_seed': seed_stats,
        'by_type': type_stats,
        'missing_experiments': missing_df,
        'cross_model_variant': cross_model_variant,
        'cross_task_variant': cross_task_variant,
        'cross_peft_variant': cross_peft_variant
    }

def print_completion_report(stats: Dict[str, Any]):
    """
    Print a comprehensive completion report.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETION REPORT")
    print("=" * 80)
    
    overall = stats['overall']
    print(f"\nOVERALL COMPLETION:")
    print(f"  Expected experiments: {overall['expected']}")
    print(f"  Completed experiments: {overall['completed']}")
    print(f"  Completion rate: {overall['rate']:.2f}%")
    print(f"  Missing experiments: {overall['missing_count']}")
    
    print("\n" + "-" * 80)
    print("COMPLETION BY TRAINING VARIANT (directory):")
    for variant, data in stats['by_variant'].items():
        print(f"  {variant:10s}: {data['completed']:4d} / {data['expected']:4d} = {data['rate']:6.2f}%")
    
    print("\n" + "-" * 80)
    print("COMPLETION BY MODEL FAMILY:")
    for model, data in stats['by_model_family'].items():
        print(f"  {model:10s}: {data['completed']:4d} / {data['expected']:4d} = {data['rate']:6.2f}%")
    
    print("\n" + "-" * 80)
    print("COMPLETION BY TASK:")
    for task, data in stats['by_task'].items():
        print(f"  {task:10s}: {data['completed']:4d} / {data['expected']:4d} = {data['rate']:6.2f}%")
    
    print("\n" + "-" * 80)
    print("COMPLETION BY PEFT VARIANT:")
    for peft, data in stats['by_peft'].items():
        print(f"  {peft:10s}: {data['completed']:4d} / {data['expected']:4d} = {data['rate']:6.2f}%")
    
    print("\n" + "-" * 80)
    print("COMPLETION BY SEED:")
    for seed, data in stats['by_seed'].items():
        print(f"  Seed {seed:4d}: {data['completed']:4d} / {data['expected']:4d} = {data['rate']:6.2f}%")
    
    print("\n" + "-" * 80)
    print("COMPLETION BY TRAINING TYPE:")
    for type_name, data in stats['by_type'].items():
        print(f"  {type_name:10s}: {data['completed']:4d} / {data['expected']:4d} = {data['rate']:6.2f}%")
    
    print("\n" + "-" * 80)
    print("CROSS-TABULATION: Model Family × Variant (completed counts):")
    print(stats['cross_model_variant'].to_string())
    
    print("\n" + "-" * 80)
    print("CROSS-TABULATION: Task × Variant (completed counts):")
    print(stats['cross_task_variant'].to_string())
    
    print("\n" + "-" * 80)
    print("CROSS-TABULATION: PEFT Variant × Variant (completed counts, excludes FFT):")
    print(stats['cross_peft_variant'].to_string())
    
    # Save missing experiments to CSV
    missing_df = stats['missing_experiments']
    if not missing_df.empty:
        missing_path = 'missing_experiments.csv'
        missing_df.to_csv(missing_path, index=False)
        print(f"\nMissing experiments saved to {missing_path}")
        print(f"Total missing: {len(missing_df)}")
        
        # Show some examples of missing experiments
        print("\nExamples of missing experiments (first 10):")
        print(missing_df.head(10).to_string(index=False))
    else:
        print("\nNo missing experiments! All expected experiments have been completed.")
    
    print("\n" + "=" * 80)
    print("REPORT COMPLETE")
    print("=" * 80)

def save_detailed_completion_table(completed_df: pd.DataFrame, expected_df: pd.DataFrame):
    """
    Create and save a detailed completion table for each combination.
    """
    # Create a grid of all expected combinations
    grid_records = []
    
    for variant in ['fft', 'lora', 'kd-lora']:
        variant_df = expected_df[expected_df['variant'] == variant]
        for model_family in MODEL_FAMILIES:
            for task in GLUE_TASKS:
                for seed in SEEDS:
                    if variant == 'fft':
                        peft_list = ['FFT']
                    else:
                        peft_list = PEFT_VARIANTS
                    
                    for peft in peft_list:
                        # Check if this experiment exists
                        exists = not completed_df[
                            (completed_df['variant'] == variant) &
                            (completed_df['model_family'] == model_family) &
                            (completed_df['task'] == task) &
                            (completed_df['peft'] == peft) &
                            (completed_df['seed'] == seed)
                        ].empty
                        
                        grid_records.append({
                            'variant': variant,
                            'model_family': model_family,
                            'task': task,
                            'peft': peft,
                            'seed': seed,
                            'completed': 1 if exists else 0
                        })
    
    grid_df = pd.DataFrame(grid_records)
    
    # Save detailed grid
    grid_path = 'completion_grid.csv'
    grid_df.to_csv(grid_path, index=False)
    print(f"Detailed completion grid saved to {grid_path}")
    
    # Create pivot tables for visualization
    # Overall completion by model_family × task × variant
    pivot = grid_df.pivot_table(
        index=['model_family', 'task'],
        columns='variant',
        values='completed',
        aggfunc='mean'
    )
    pivot_path = 'completion_pivot.csv'
    pivot.to_csv(pivot_path)
    print(f"Completion pivot table saved to {pivot_path}")
    
    return grid_df

def plot_completion_heatmap(grid_df: pd.DataFrame, output_dir: str = '.'):
    """
    Generate heatmap visualizations of completion rates.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib or Seaborn not available. Skipping heatmap generation.")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Heatmap: completion rate by model_family × task for each variant
    for variant in ['fft', 'lora', 'kd-lora']:
        variant_df = grid_df[grid_df['variant'] == variant]
        if variant_df.empty:
            continue
        
        # Aggregate over seeds and PEFT variants (for FFT, peft is just 'FFT')
        if variant == 'fft':
            pivot = variant_df.pivot_table(
                index='model_family',
                columns='task',
                values='completed',
                aggfunc='mean'
            )
        else:
            pivot = variant_df.pivot_table(
                index='model_family',
                columns='task',
                values='completed',
                aggfunc='mean'
            )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', vmin=0, vmax=1,
                    cbar_kws={'label': 'Completion Rate'}, linewidths=0.5)
        plt.title(f'Completion Rate: {variant} (Model Family × Task)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/completion_heatmap_{variant}.png', dpi=150)
        plt.close()
    
    # 2. Heatmap: completion rate by PEFT variant × task for LoRA variants
    for variant in ['lora', 'kd-lora']:
        variant_df = grid_df[grid_df['variant'] == variant]
        if variant_df.empty:
            continue
        
        # Aggregate over seeds and model families
        pivot = variant_df.pivot_table(
            index='peft',
            columns='task',
            values='completed',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', vmin=0, vmax=1,
                    cbar_kws={'label': 'Completion Rate'}, linewidths=0.5)
        plt.title(f'Completion Rate: {variant} (PEFT Variant × Task)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/completion_peft_task_{variant}.png', dpi=150)
        plt.close()
    
    # 3. Bar plot: overall completion by dimension
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Completion by variant
    variant_completion = grid_df.groupby('variant')['completed'].mean().reset_index()
    axes[0,0].bar(variant_completion['variant'], variant_completion['completed'] * 100, color='steelblue')
    axes[0,0].set_xlabel('Training Variant')
    axes[0,0].set_ylabel('Completion Rate (%)')
    axes[0,0].set_title('Completion by Training Variant')
    axes[0,0].grid(True, alpha=0.3, axis='y')
    
    # Completion by model family
    model_completion = grid_df.groupby('model_family')['completed'].mean().reset_index()
    axes[0,1].bar(model_completion['model_family'], model_completion['completed'] * 100, color='coral')
    axes[0,1].set_xlabel('Model Family')
    axes[0,1].set_ylabel('Completion Rate (%)')
    axes[0,1].set_title('Completion by Model Family')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # Completion by task
    task_completion = grid_df.groupby('task')['completed'].mean().reset_index()
    axes[1,0].bar(task_completion['task'], task_completion['completed'] * 100, color='mediumseagreen')
    axes[1,0].set_xlabel('Task')
    axes[1,0].set_ylabel('Completion Rate (%)')
    axes[1,0].set_title('Completion by Task')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3, axis='y')
    
    # Completion by seed
    seed_completion = grid_df.groupby('seed')['completed'].mean().reset_index()
    axes[1,1].bar(seed_completion['seed'].astype(str), seed_completion['completed'] * 100, color='goldenrod')
    axes[1,1].set_xlabel('Seed')
    axes[1,1].set_ylabel('Completion Rate (%)')
    axes[1,1].set_title('Completion by Seed')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/completion_summary_bars.png', dpi=150)
    plt.close()
    
    print(f"Heatmap visualizations saved to {output_dir}/")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment completion rates for KD-LoRA.')
    parser.add_argument('--plot', action='store_true', help='Generate heatmap visualizations')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for plots and CSV files')
    args = parser.parse_args()
    
    print("=" * 80)
    print("KD-LoRA Experiment Completion Analysis")
    print("=" * 80)
    
    # Step 1: Collect completed experiments
    print("\n1. Collecting completed experiments from results directory...")
    completed_df = collect_completed_experiments()
    
    if completed_df.empty:
        print("No completed experiments found. Exiting.")
        return
    
    # Step 2: Generate expected experiments
    print("\n2. Generating expected experiment configurations...")
    expected_df = generate_expected_experiments()
    
    # Step 3: Compute completion statistics
    print("\n3. Computing completion rates...")
    stats = compute_completion_rates(completed_df, expected_df)
    
    # Step 4: Print report
    print_completion_report(stats)
    
    # Step 5: Save detailed completion tables
    print("\n4. Saving detailed completion tables...")
    grid_df = save_detailed_completion_table(completed_df, expected_df)
    
    # Step 6: Generate visualizations if requested
    if args.plot:
        print("\n5. Generating visualizations...")
        plot_completion_heatmap(grid_df, args.output_dir)
    
    # Additional analysis: check for duplicates
    duplicate_keys = completed_df[completed_df.duplicated(subset=['variant', 'model_family', 'task', 'peft', 'seed'])]
    if not duplicate_keys.empty:
        print(f"\nWARNING: Found {len(duplicate_keys)} duplicate experiments!")
        dup_path = f'{args.output_dir}/duplicate_experiments.csv'
        duplicate_keys.to_csv(dup_path, index=False)
        print(f"Duplicate experiments saved to {dup_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()