#!/usr/bin/env python3
"""
Visualization functions for KD-LoRA experimental results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Model families and strategies
MODEL_FAMILIES = ['bert', 'roberta', 'deberta']
STRATEGIES = ['FFT', 'LoRA', 'KD-LoRA']

def plot_heatmap(table_df: pd.DataFrame, output_path: str = 'table_i_heatmap.png'):
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
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {output_path}")

def plot_strategy_comparison_by_model(table_df: pd.DataFrame, output_path: str = 'strategy_comparison_by_model.png'):
    """Generate bar plots comparing strategies for each model family."""
    # Remove Average row for plots
    plot_df = table_df.drop('Average', errors='ignore')
    
    # Create subplots for each model family
    fig, axes = plt.subplots(1, len(MODEL_FAMILIES), figsize=(18, 6), sharey=True)
    if len(MODEL_FAMILIES) == 1:
        axes = [axes]
    
    for idx, model in enumerate(MODEL_FAMILIES):
        ax = axes[idx]
        
        # Prepare data for this model: tasks x strategies
        tasks = []
        strategy_data = {s: [] for s in STRATEGIES}
        
        for task in plot_df.index:
            # Check if at least one strategy has data for this task
            has_data = False
            for strategy in STRATEGIES:
                val = plot_df.loc[task, (model, strategy)]
                if not pd.isna(val):
                    has_data = True
            
            if has_data:
                tasks.append(task)
                for strategy in STRATEGIES:
                    val = plot_df.loc[task, (model, strategy)]
                    strategy_data[strategy].append(val if not pd.isna(val) else 0)
        
        if not tasks:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model.capitalize()} (No Data)')
            continue
        
        # Create bar positions
        x = np.arange(len(tasks))
        width = 0.25
        
        # Plot bars for each strategy
        bars = []
        for i, strategy in enumerate(STRATEGIES):
            offset = (i - 1) * width  # Center FFT, left LoRA, right KD-LoRA
            values = strategy_data[strategy]
            bars.append(ax.bar(x + offset, values, width, label=strategy, alpha=0.8))
            
            # Add value labels on top of bars (only for non-zero values)
            for bar, val in zip(bars[-1], values):
                if val > 0:
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
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Strategy comparison by model saved to {output_path}")

def plot_model_comparison_by_task(table_df: pd.DataFrame, output_path: str = 'model_comparison_by_task.png'):
    """Generate bar plots comparing models for each task."""
    # Remove Average row for plots
    plot_df = table_df.drop('Average', errors='ignore')
    
    # Group tasks by availability (tasks that have data for at least one model and strategy)
    tasks_with_data = []
    for task in plot_df.index:
        has_data = False
        for model in MODEL_FAMILIES:
            for strategy in STRATEGIES:
                if not pd.isna(plot_df.loc[task, (model, strategy)]):
                    has_data = True
                    break
            if has_data:
                break
        if has_data:
            tasks_with_data.append(task)
    
    if not tasks_with_data:
        print("No data available for plotting")
        return
    
    # Create subplots: arrange in grid
    n_tasks = len(tasks_with_data)
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    for idx, task in enumerate(tasks_with_data):
        ax = axes_flat[idx]
        
        # Prepare data for this task: models x strategies
        model_data = {}
        for model in MODEL_FAMILIES:
            model_vals = []
            for strategy in STRATEGIES:
                val = plot_df.loc[task, (model, strategy)]
                model_vals.append(val if not pd.isna(val) else 0)
            model_data[model] = model_vals
        
        # Create bar positions
        x = np.arange(len(STRATEGIES))
        width = 0.25
        
        # Plot bars for each model
        bars = []
        for i, model in enumerate(MODEL_FAMILIES):
            offset = (i - 1) * width
            values = model_data[model]
            bars.append(ax.bar(x + offset, values, width, label=model, alpha=0.8))
            
            # Add value labels
            for bar, val in zip(bars[-1], values):
                if val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Metric Score')
        ax.set_title(f'Task: {task}')
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGIES)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(tasks_with_data), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.suptitle('Model Comparison Across Strategies for Each Task', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Model comparison by task saved to {output_path}")

def plot_performance_delta(table_df: pd.DataFrame, output_path: str = 'performance_delta.png'):
    """Plot performance delta between FFT and LoRA, FFT and KD-LoRA."""
    # Remove Average row
    plot_df = table_df.drop('Average', errors='ignore')
    
    # Calculate deltas
    delta_data = []
    for task in plot_df.index:
        for model in MODEL_FAMILIES:
            fft_val = plot_df.loc[task, (model, 'FFT')] if (model, 'FFT') in plot_df.columns else np.nan
            lora_val = plot_df.loc[task, (model, 'LoRA')] if (model, 'LoRA') in plot_df.columns else np.nan
            kd_lora_val = plot_df.loc[task, (model, 'KD-LoRA')] if (model, 'KD-LoRA') in plot_df.columns else np.nan
            
            if not pd.isna(fft_val) and not pd.isna(lora_val):
                delta_data.append({
                    'Task': task,
                    'Model': model,
                    'Delta_Type': 'FFT - LoRA',
                    'Delta': fft_val - lora_val
                })
            
            if not pd.isna(fft_val) and not pd.isna(kd_lora_val):
                delta_data.append({
                    'Task': task,
                    'Model': model,
                    'Delta_Type': 'FFT - KD-LoRA',
                    'Delta': fft_val - kd_lora_val
                })
    
    if not delta_data:
        print("No delta data available for plotting")
        return
    
    delta_df = pd.DataFrame(delta_data)
    
    # Create grouped bar plot
    plt.figure(figsize=(12, 6))
    
    # Pivot for easier plotting
    pivot_df = delta_df.pivot_table(
        index=['Task', 'Model'],
        columns='Delta_Type',
        values='Delta'
    ).reset_index()
    
    # Plot
    x = np.arange(len(pivot_df))
    width = 0.35
    
    # Check which delta types exist
    delta_types = [col for col in ['FFT - LoRA', 'FFT - KD-LoRA'] if col in pivot_df.columns]
    
    bars = []
    for i, delta_type in enumerate(delta_types):
        offset = (i - 0.5) * width
        values = pivot_df[delta_type].fillna(0).values
        bars.append(plt.bar(x + offset, values, width, label=delta_type, alpha=0.8))
    
    plt.xlabel('Task-Model Combination')
    plt.ylabel('Performance Delta (FFT - Other)')
    plt.title('Performance Delta: FFT vs LoRA and KD-LoRA')
    plt.xticks(x, [f'{row.Task}\n{row.Model}' for _, row in pivot_df.iterrows()], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Performance delta plot saved to {output_path}")

def plot_average_performance(table_df: pd.DataFrame, output_path: str = 'average_performance.png'):
    """Plot average performance across all tasks for each model and strategy."""
    # Extract average row
    if 'Average' not in table_df.index:
        print("No average row in table")
        return
    
    avg_row = table_df.loc['Average']
    
    # Prepare data
    data = []
    for model in MODEL_FAMILIES:
        for strategy in STRATEGIES:
            val = avg_row[(model, strategy)] if (model, strategy) in avg_row.index else np.nan
            if not pd.isna(val):
                data.append({
                    'Model': model,
                    'Strategy': strategy,
                    'Average Score': val
                })
    
    if not data:
        print("No average data available")
        return
    
    avg_df = pd.DataFrame(data)
    
    # Pivot for grouped bar plot
    pivot_df = avg_df.pivot(index='Model', columns='Strategy', values='Average Score')
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(MODEL_FAMILIES))
    width = 0.25
    
    # Plot bars for each strategy
    for i, strategy in enumerate(STRATEGIES):
        if strategy in pivot_df.columns:
            offset = (i - 1) * width
            values = pivot_df[strategy].values
            bars = plt.bar(x + offset, values, width, label=strategy, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Model Family')
    plt.ylabel('Average Metric Score')
    plt.title('Average Performance Across All GLUE Tasks')
    plt.xticks(x, [m.capitalize() for m in MODEL_FAMILIES])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Average performance plot saved to {output_path}")

def generate_all_visualizations(table_df: pd.DataFrame, prefix: str = ''):
    """Generate all visualizations."""
    if prefix and not prefix.endswith('_'):
        prefix = prefix + '_'
    
    print("Generating visualizations...")
    
    # Heatmap
    plot_heatmap(table_df, f'{prefix}table_i_heatmap.png')
    
    # Strategy comparison by model
    plot_strategy_comparison_by_model(table_df, f'{prefix}strategy_comparison_by_model.png')
    
    # Model comparison by task
    plot_model_comparison_by_task(table_df, f'{prefix}model_comparison_by_task.png')
    
    # Performance delta
    plot_performance_delta(table_df, f'{prefix}performance_delta.png')
    
    # Average performance
    plot_average_performance(table_df, f'{prefix}average_performance.png')
    
    print("All visualizations generated!")

if __name__ == '__main__':
    # Example usage: load table from CSV and generate visualizations
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        try:
            table_df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
            generate_all_visualizations(table_df)
        except Exception as e:
            print(f"Error loading CSV: {e}")
    else:
        print("Usage: python visualize_results.py <table_csv_path>")
        print("Example: python visualize_results.py table_i_results.csv")