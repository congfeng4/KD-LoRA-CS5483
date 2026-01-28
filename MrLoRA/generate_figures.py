#!/usr/bin/env python3
"""
Generate figures for MR‑LoRA manuscript:
1. Sensitivity bar chart (task‑wise performance drop)
2. Trade‑off scatter plot (GLUE score vs parameter count)
3. Schematic diagram of MR‑LoRA adapter.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def generate_sensitivity():
    """Create bar chart of task‑wise performance drops (LoRA vs MR‑LoRA)."""
    df = pd.read_csv('sensitivity_data.csv')
    # Ensure tasks are in a sensible order
    task_order = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'qnli', 'rte', 'wnli']
    df['task'] = pd.Categorical(df['task'], categories=task_order, ordered=True)
    df = df.sort_values('task')
    
    tasks = df['task'].tolist()
    drops_lora = df['drop_lora'].tolist()
    drops_mrlora = df['drop_mrlora'].tolist()
    
    x = np.arange(len(tasks))
    width = 0.35
    
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, drops_lora, width, label='LoRA', color='#1f77b4')
    bars2 = ax.bar(x + width/2, drops_mrlora, width, label='MR‑LoRA', color='#ff7f0e')
    
    ax.set_xlabel('GLUE Task')
    ax.set_ylabel('Performance Drop (FFT → Student)')
    ax.set_title('Task‑wise Sensitivity to Distillation')
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/sensitivity.png', dpi=300)
    print('Saved sensitivity.png')

def generate_tradeoff():
    """Create scatter plot of GLUE score vs trainable parameters."""
    df = pd.read_csv('tradeoff_data.csv')
    # Map method names to display labels
    method_labels = {
        'fft': 'FFT',
        'lora': 'LoRA',
        'mrlora': 'MR‑LoRA',
        'mrlora-rs': 'MR‑LoRA‑RS',
        'adalora': 'AdaLoRA',
        'dora': 'DoRA',
        'olora': 'OLoRA',
        'rslora': 'RS‑LoRA'
    }
    df['method_label'] = df['method'].map(lambda x: method_labels.get(x, x))
    
    # Color map for methods
    unique_methods = df['method_label'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_methods)))
    method_to_color = dict(zip(unique_methods, colors))
    
    # Marker map for model families
    family_markers = {
        'bert': 'o',
        'roberta': 's',
        'deberta': '^'
    }
    
    fig, ax = plt.subplots()
    
    for (model_family, method_label), group in df.groupby(['model_family', 'method_label']):
        ax.scatter(group['param_count_m'], group['glue_score'],
                   label=f'{model_family} {method_label}',
                   color=method_to_color[method_label],
                   marker=family_markers[model_family],
                   s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Trainable Parameters (M)')
    ax.set_ylabel('GLUE Score')
    ax.set_title('Efficiency–Accuracy Trade‑off')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend: separate legend for methods and families?
    # Simpler: just one legend, but may be crowded. We'll create two legends.
    from matplotlib.lines import Line2D
    method_handles = []
    for method, color in method_to_color.items():
        method_handles.append(Line2D([0], [0], marker='o', color='w', label=method,
                                     markerfacecolor=color, markersize=10))
    family_handles = []
    for family, marker in family_markers.items():
        family_handles.append(Line2D([0], [0], marker=marker, color='w', label=family,
                                     markerfacecolor='gray', markersize=10))
    
    # Place method legend on top right, family legend on top left
    ax.legend(handles=method_handles, title='Method', loc='upper left', bbox_to_anchor=(1.02, 1))
    ax2 = ax.twinx()
    ax2.set_ylabel('')  # empty
    ax2.legend(handles=family_handles, title='Model Family', loc='upper left', bbox_to_anchor=(1.02, 0.7))
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('figures/tradeoff.png', dpi=300, bbox_inches='tight')
    print('Saved tradeoff.png')

def generate_schematic():
    """Draw a simple block diagram of MR‑LoRA adapter (unchanged from original)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    # Draw a rectangle for linear layer
    ax.add_patch(Rectangle((0.1, 0.3), 0.2, 0.4, fill=None, edgecolor='black'))
    ax.text(0.2, 0.7, '$W_0$', ha='center', va='center')
    # Draw multiple blocks
    for i in range(3):
        x = 0.4 + i*0.15
        ax.add_patch(Rectangle((x, 0.3), 0.1, 0.4, fill=None, edgecolor='blue'))
        ax.text(x+0.05, 0.7, f'$B_{i+1}A_{i+1}$', ha='center', va='center', fontsize=8)
        # weight lambda
        ax.text(x+0.05, 0.2, f'$\\lambda_{i+1}$', ha='center', va='center')
    # Sum symbol
    ax.text(0.85, 0.5, '$+$', ha='center', va='center', fontsize=20)
    # Output arrow
    ax.arrow(0.9, 0.5, 0.1, 0, head_width=0.05, head_length=0.02, fc='black')
    ax.text(0.95, 0.55, '$\\Delta W$', ha='center', va='bottom')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('figures/mrlora_schematic.png', dpi=300)
    print('Saved mrlora_schematic.png')

def main():
    os.makedirs('figures', exist_ok=True)
    generate_sensitivity()
    generate_tradeoff()
    generate_schematic()
    print('All figures generated.')

if __name__ == '__main__':
    main()