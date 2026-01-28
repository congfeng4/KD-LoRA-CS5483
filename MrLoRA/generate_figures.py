#!/usr/bin/env python3
"""
Generate sensitivity figure for appendix.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load GLUE scores
df = pd.read_csv('../table_i_results.csv', header=[0,1])
# Drop the 'Average' row
df = df[df[('Model Family', 'Strategy')] != 'Average']
# Get task names (first column)
tasks = df.iloc[:, 0].tolist()
# For each task, compute average drop across models
drops = []
for task in tasks:
    # Get scores for this task across all model families and strategies
    # We'll compute average drop from FFT to KD-LoRA
    # For simplicity, let's compute per model family then average
    task_drops = []
    for model in ['bert', 'roberta', 'deberta']:
        fft_score = df.loc[df.iloc[:, 0] == task, (model, 'FFT')].values[0]
        kd_score = df.loc[df.iloc[:, 0] == task, (model, 'KD-LoRA')].values[0]
        drop = fft_score - kd_score
        task_drops.append(drop)
    drops.append(np.mean(task_drops))

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(tasks)), drops)
plt.xticks(range(len(tasks)), tasks, rotation=45, ha='right')
plt.ylabel('Average performance drop (FFT → KD‑LoRA)')
plt.title('Task‑wise sensitivity to distillation')
plt.tight_layout()
plt.savefig('figures/sensitivity.png', dpi=300)
print('Saved sensitivity.png')

# Also create schematic (simple diagram)
# Draw a simple block diagram using matplotlib
fig, ax = plt.subplots(figsize=(8, 4))
# Draw a rectangle for linear layer
ax.add_patch(plt.Rectangle((0.1, 0.3), 0.2, 0.4, fill=None, edgecolor='black'))
ax.text(0.2, 0.7, '$W_0$', ha='center', va='center')
# Draw multiple blocks
for i in range(3):
    x = 0.4 + i*0.15
    ax.add_patch(plt.Rectangle((x, 0.3), 0.1, 0.4, fill=None, edgecolor='blue'))
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

print('Figures generated.')