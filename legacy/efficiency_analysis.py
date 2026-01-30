import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Load CSV
df = pd.read_csv('kd-lora-table-ii.csv')
print("Original data:")
print(df)

# Clean values: remove suffixes and convert to numeric
def clean_value(val):
    if isinstance(val, str):
        val = val.strip()
        # Remove M, MB, s suffixes
        if val.endswith('M'):
            return float(val[:-1])
        elif val.endswith('MB'):
            return float(val[:-2])
        elif val.endswith('s'):
            return float(val[:-1])
        else:
            return float(val)
    return val

# Apply cleaning to relevant columns
cols_to_clean = ['Rank 8', 'Rank 16', 'Rank 32', 'Rank 64', 'Memory Usage', 'Inference Time']
for col in cols_to_clean:
    df[col] = df[col].apply(clean_value)

print("\nCleaned data:")
print(df)

# Add model family column for grouping
def get_family(model_name):
    if 'BERT-base' in model_name or 'DistilBERT-base' in model_name:
        return 'BERT-base'
    elif 'RoBERTa-base' in model_name or 'DistilRoBERTa-base' in model_name:
        return 'RoBERTa-base'
    elif 'DeBERTa-v3-base' in model_name or 'DeBERTa-v3-small' in model_name:
        return 'DeBERTa-v3-base'
    else:
        return model_name

df['Model Family'] = df['Model'].apply(get_family)

# Compute parameter reduction ratios relative to FFT for each rank
# We'll create a new dataframe with FFT values as reference
fft_df = df[df['Method'] == 'FFT'].set_index('Model Family')
lora_df = df[df['Method'] == 'LoRA'].set_index('Model Family')
kd_df = df[df['Method'] == 'KD-LoRA'].set_index('Model Family')

ranks = ['Rank 8', 'Rank 16', 'Rank 32', 'Rank 64']

# Parameter reduction ratio = FFT_params / method_params (higher means more reduction)
# Actually reduction ratio = (FFT - method) / FFT = 1 - method/FFT
# Let's compute reduction percentage
reduction_lora = {}
reduction_kd = {}
for rank in ranks:
    reduction_lora[rank] = (fft_df[rank] - lora_df[rank]) / fft_df[rank]
    reduction_kd[rank] = (fft_df[rank] - kd_df[rank]) / fft_df[rank]

reduction_lora_df = pd.DataFrame(reduction_lora)
reduction_kd_df = pd.DataFrame(reduction_kd)

print("\nParameter reduction ratio (relative to FFT) for LoRA:")
print(reduction_lora_df)
print("\nParameter reduction ratio (relative to FFT) for KD-LoRA:")
print(reduction_kd_df)

# Memory reduction ratio
mem_reduction_lora = (fft_df['Memory Usage'] - lora_df['Memory Usage']) / fft_df['Memory Usage']
mem_reduction_kd = (fft_df['Memory Usage'] - kd_df['Memory Usage']) / fft_df['Memory Usage']
print("\nMemory reduction ratio for LoRA:")
print(mem_reduction_lora)
print("\nMemory reduction ratio for KD-LoRA:")
print(mem_reduction_kd)

# Inference time speedup: FFT_time / method_time (higher is better)
speedup_lora = fft_df['Inference Time'] / lora_df['Inference Time']
speedup_kd = fft_df['Inference Time'] / kd_df['Inference Time']
print("\nInference time speedup (FFT/method) for LoRA:")
print(speedup_lora)
print("\nInference time speedup (FFT/method) for KD-LoRA:")
print(speedup_kd)

# Compute average reduction/speedup across models for each method
avg_reduction_lora = reduction_lora_df.mean()
avg_reduction_kd = reduction_kd_df.mean()
avg_mem_reduction_lora = mem_reduction_lora.mean()
avg_mem_reduction_kd = mem_reduction_kd.mean()
avg_speedup_lora = speedup_lora.mean()
avg_speedup_kd = speedup_kd.mean()

print("\nAverage across models:")
print(f"LoRA parameter reduction: {avg_reduction_lora}")
print(f"KD-LoRA parameter reduction: {avg_reduction_kd}")
print(f"LoRA memory reduction: {avg_mem_reduction_lora:.3f}")
print(f"KD-LoRA memory reduction: {avg_mem_reduction_kd:.3f}")
print(f"LoRA inference speedup: {avg_speedup_lora:.3f}")
print(f"KD-LoRA inference speedup: {avg_speedup_kd:.3f}")

# Create visualizations
output_dir = 'efficiency_plots'
os.makedirs(output_dir, exist_ok=True)

# 1. Bar chart: parameter count (Rank 8) for each model and method
fig, ax = plt.subplots()
# Prepare data: each model family has three methods
# We'll pivot for Rank 8
rank8_data = df.pivot(index='Model Family', columns='Method', values='Rank 8')
rank8_data = rank8_data[['FFT', 'LoRA', 'KD-LoRA']]  # ensure order
rank8_data.plot(kind='bar', ax=ax)
ax.set_ylabel('Parameter Count (M)')
ax.set_title('Parameter Count (Rank 8) by Model and Method')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'param_count_rank8.png'), dpi=300)
plt.close()

# 2. Bar chart: memory usage for each model and method
fig, ax = plt.subplots()
mem_data = df.pivot(index='Model Family', columns='Method', values='Memory Usage')
mem_data = mem_data[['FFT', 'LoRA', 'KD-LoRA']]
mem_data.plot(kind='bar', ax=ax)
ax.set_ylabel('Memory Usage (MB)')
ax.set_title('Memory Usage by Model and Method')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'memory_usage.png'), dpi=300)
plt.close()

# 3. Bar chart: inference time for each model and method
fig, ax = plt.subplots()
time_data = df.pivot(index='Model Family', columns='Method', values='Inference Time')
time_data = time_data[['FFT', 'LoRA', 'KD-LoRA']]
time_data.plot(kind='bar', ax=ax)
ax.set_ylabel('Inference Time (s)')
ax.set_title('Inference Time by Model and Method')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'inference_time.png'), dpi=300)
plt.close()

# 4. Line plot: parameter count vs rank (x-axis rank, y-axis parameter count, lines for each model-method combination)
fig, ax = plt.subplots()
rank_values = [8, 16, 32, 64]
for (model_family, method), group in df.groupby(['Model Family', 'Method']):
    # extract parameter values across ranks
    params = [group[f'Rank {r}'].values[0] for r in rank_values]
    label = f'{model_family} {method}'
    ax.plot(rank_values, params, marker='o', label=label)
ax.set_xlabel('LoRA Rank')
ax.set_ylabel('Parameter Count (M)')
ax.set_title('Parameter Count vs LoRA Rank')
ax.set_xticks(rank_values)
ax.set_yscale('log')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'param_vs_rank.png'), dpi=300)
plt.close()

# 5. Heatmap: parameter reduction ratio (FFT vs LoRA, FFT vs KD-LoRA) across ranks and models
# Combine reduction data into a single dataframe for heatmap
# We'll create a multi-index (Model, Method) where Method is reduction type
heatmap_data = []
for model in reduction_lora_df.index:
    for rank in ranks:
        heatmap_data.append({
            'Model': model,
            'Rank': rank,
            'Reduction': reduction_lora_df.loc[model, rank],
            'Method': 'LoRA'
        })
        heatmap_data.append({
            'Model': model,
            'Rank': rank,
            'Reduction': reduction_kd_df.loc[model, rank],
            'Method': 'KD-LoRA'
        })
heatmap_df = pd.DataFrame(heatmap_data)
# Pivot for heatmap: rows = models, columns = rank, values = reduction
heatmap_pivot = heatmap_df.pivot_table(index=['Model', 'Method'], columns='Rank', values='Reduction')
# Plot two heatmaps side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, method in enumerate(['LoRA', 'KD-LoRA']):
    subset = heatmap_pivot.loc[(slice(None), method), :].droplevel(1)
    sns.heatmap(subset, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[idx])
    axes[idx].set_title(f'Parameter Reduction Ratio ({method})')
    axes[idx].set_xlabel('Rank')
    axes[idx].set_ylabel('Model')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'reduction_heatmap.png'), dpi=300)
plt.close()

print("\nPlots saved to efficiency_plots/")