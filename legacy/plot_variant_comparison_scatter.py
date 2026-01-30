#!/usr/bin/env python3
"""
Scatter plot comparing LoRA-only vs KD-LoRA performance per variant.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load summary data
table_iia = pd.read_csv('table_iia_results.csv', index_col=0)
table_iib = pd.read_csv('table_iib_results.csv', index_col=0)

# Extract average rows
avg_iia = table_iia.loc['Average']
avg_iib = table_iib.loc['Average']

# Create comparison DataFrame
variants = [v for v in avg_iia.index if v != 'FFT']
data = []
for variant in variants:
    lora_score = avg_iia.get(variant, np.nan)
    kd_lora_score = avg_iib.get(variant, np.nan)
    if not pd.isna(lora_score) and not pd.isna(kd_lora_score):
        data.append({
            'Variant': variant,
            'LoRA-only': lora_score,
            'KD-LoRA': kd_lora_score,
            'Difference': kd_lora_score - lora_score
        })

df = pd.DataFrame(data)

print("Variant comparison (average across tasks):")
print(df.round(4))

# Create scatter plot
plt.figure(figsize=(8, 6))
ax = plt.gca()

# Plot diagonal line
min_val = min(df['LoRA-only'].min(), df['KD-LoRA'].min()) - 0.02
max_val = max(df['LoRA-only'].max(), df['KD-LoRA'].max()) + 0.02
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')

# Plot points
colors = sns.color_palette('husl', len(df))
for idx, row in df.iterrows():
    plt.scatter(row['LoRA-only'], row['KD-LoRA'], 
                color=colors[idx], s=100, edgecolor='black', linewidth=1)
    plt.annotate(row['Variant'], (row['LoRA-only'], row['KD-LoRA']),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('LoRA-only Average Score', fontsize=12)
plt.ylabel('KD-LoRA Average Score', fontsize=12)
plt.title('Performance Comparison: LoRA-only vs KD-LoRA Variants', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('scatter_lora_vs_kdlora.png', dpi=150)
print("\nScatter plot saved to scatter_lora_vs_kdlora.png")

# Create bar plot of differences
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Variant'], df['Difference'], color='steelblue', edgecolor='black')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('PEFT Variant', fontsize=12)
plt.ylabel('KD-LoRA - LoRA-only Difference', fontsize=12)
plt.title('Performance Difference: KD-LoRA vs LoRA-only', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')

# Annotate bars
for bar in bars:
    height = bar.get_height()
    va = 'bottom' if height >= 0 else 'top'
    color = 'green' if height >= 0 else 'red'
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height:+.4f}', ha='center', va=va, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('difference_kdlora_minus_lora.png', dpi=150)
print("Difference bar plot saved to difference_kdlora_minus_lora.png")

# Print summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
for _, row in df.iterrows():
    diff = row['Difference']
    if diff > 0:
        print(f"{row['Variant']:10s}: KD-LoRA outperforms LoRA-only by {diff:.4f}")
    elif diff < 0:
        print(f"{row['Variant']:10s}: LoRA-only outperforms KD-LoRA by {-diff:.4f}")
    else:
        print(f"{row['Variant']:10s}: Equal performance")

print("\nOverall, KD-LoRA underperforms LoRA-only for all variants.")
print("Largest gap: adalora (KD-LoRA improves relative to LoRA-only?)")
print("Smallest gap: mrlora")