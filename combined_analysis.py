#!/usr/bin/env python3
"""
Combine GLUE performance (Table I) with efficiency metrics (Table II)
to analyze accuracy-efficiency trade-offs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ---------- Load GLUE scores (Table I) ----------
glue_path = "table_i_results.csv"
if not os.path.exists(glue_path):
    raise FileNotFoundError(f"GLUE scores not found: {glue_path}")

# The CSV has two header rows: Model Family and Strategy
glue_df = pd.read_csv(glue_path, header=[0, 1])
# The last row is "Average"
glue_avg = glue_df.iloc[-1]  # Series with MultiIndex columns
# Drop the "Task" column (first element)
glue_avg = glue_avg.iloc[1:]  # skip first column (Task names)

# Convert to DataFrame for easier manipulation
# The MultiIndex columns are (Model Family, Strategy)
# We'll create a tidy DataFrame
records = []
for (model_family, strategy), score in glue_avg.items():
    records.append({
        'Model Family': model_family,
        'Strategy': strategy,
        'GLUE Score': score
    })
glue_scores = pd.DataFrame(records)

# Map to match efficiency table naming
model_map = {
    'bert': 'BERT-base',
    'roberta': 'RoBERTa-base', 
    'deberta': 'DeBERTa-v3-base'
}
glue_scores['Model'] = glue_scores['Model Family'].map(model_map)
# For KD-LoRA, the model is distilled variant
def get_model_name(row):
    if row['Strategy'] == 'KD-LoRA':
        if row['Model Family'] == 'bert':
            return 'DistilBERT-base'
        elif row['Model Family'] == 'roberta':
            return 'DistilRoBERTa-base'
        elif row['Model Family'] == 'deberta':
            return 'DeBERTa-v3-small'
    else:
        return row['Model']
glue_scores['Model'] = glue_scores.apply(get_model_name, axis=1)

# ---------- Load efficiency data (Table II) ----------
eff_path = "kd-lora-table-ii.csv"
if not os.path.exists(eff_path):
    raise FileNotFoundError(f"Efficiency data not found: {eff_path}")

eff_df = pd.read_csv(eff_path, sep=',')
# Clean columns
eff_df.columns = [col.strip() for col in eff_df.columns]
# Clean values: remove suffixes M, MB, s
def clean_value(x):
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('M'):
            return float(x.replace('M', ''))
        elif x.endswith('MB'):
            return float(x.replace('MB', ''))
        elif x.endswith('s'):
            return float(x.replace('s', ''))
        else:
            return float(x)
    else:
        return x

for col in eff_df.columns:
    if col.startswith('Rank') or col in ['Memory Usage', 'Inference Time']:
        eff_df[col] = eff_df[col].apply(clean_value)

# For each model-method, we'll use Rank 8 parameters
eff_df['Parameters (M)'] = eff_df['Rank 8']

# ---------- Merge ----------
# Merge on Model and Method (Strategy)
combined = pd.merge(
    glue_scores, 
    eff_df[['Model', 'Method', 'Parameters (M)', 'Memory Usage', 'Inference Time']],
    left_on=['Model', 'Strategy'],
    right_on=['Model', 'Method'],
    how='inner'
)
combined = combined.drop(columns='Method')
print("Combined dataset:")
print(combined)

# ---------- Compute efficiency-accuracy metrics ----------
combined['Score per Param (per M)'] = combined['GLUE Score'] / combined['Parameters (M)']
combined['Score per Memory (per GB)'] = combined['GLUE Score'] / (combined['Memory Usage'] / 1024)  # MB to GB
combined['Score per Time (per s)'] = combined['GLUE Score'] / combined['Inference Time']

print("\nEfficiency-Accuracy Metrics:")
print(combined[['Model', 'Strategy', 'GLUE Score', 'Parameters (M)', 'Score per Param (per M)']])

# ---------- Visualizations ----------
output_dir = "combined_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Scatter: Parameters vs GLUE Score
plt.figure()
for strategy in combined['Strategy'].unique():
    subset = combined[combined['Strategy'] == strategy]
    plt.scatter(subset['Parameters (M)'], subset['GLUE Score'], 
                label=strategy, s=100)
plt.xlabel('Trainable Parameters (M)')
plt.ylabel('GLUE Score')
plt.title('GLUE Score vs Parameter Count (Rank 8)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'scatter_params_vs_score.png'), dpi=300)
plt.close()

# 2. Scatter: Inference Time vs GLUE Score
plt.figure()
for strategy in combined['Strategy'].unique():
    subset = combined[combined['Strategy'] == strategy]
    plt.scatter(subset['Inference Time'], subset['GLUE Score'], 
                label=strategy, s=100)
plt.xlabel('Inference Time (s)')
plt.ylabel('GLUE Score')
plt.title('GLUE Score vs Inference Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'scatter_time_vs_score.png'), dpi=300)
plt.close()

# 3. Bar chart: Score per Parameter
plt.figure()
combined_sorted = combined.sort_values('Score per Param (per M)', ascending=False)
plt.bar(range(len(combined_sorted)), combined_sorted['Score per Param (per M)'])
plt.xticks(range(len(combined_sorted)), 
           [f"{row['Model']} ({row['Strategy']})" for _, row in combined_sorted.iterrows()], 
           rotation=45, ha='right')
plt.ylabel('GLUE Score per Million Parameters')
plt.title('Efficiency: Accuracy per Parameter')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bar_score_per_param.png'), dpi=300)
plt.close()

# 4. Heatmap: Normalized metrics
metrics_norm = combined.copy()
for col in ['GLUE Score', 'Parameters (M)', 'Memory Usage', 'Inference Time']:
    metrics_norm[f'{col} (norm)'] = metrics_norm[col] / metrics_norm[col].max()
norm_cols = [f'{col} (norm)' for col in ['GLUE Score', 'Parameters (M)', 'Memory Usage', 'Inference Time']]
heatmap_data = metrics_norm.set_index(['Model', 'Strategy'])[norm_cols]
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f')
plt.title('Normalized Metrics (higher = better for GLUE, worse for resources)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_normalized.png'), dpi=300)
plt.close()

# ---------- Generate report ----------
report_lines = []
report_lines.append("# Combined Accuracy-Efficiency Analysis")
report_lines.append("\n## Combined Dataset")
report_lines.append(combined.to_markdown(index=False))

report_lines.append("\n## Efficiency-Accuracy Metrics")
report_lines.append(combined[['Model', 'Strategy', 'GLUE Score', 'Parameters (M)', 'Score per Param (per M)',
                              'Memory Usage', 'Inference Time']].to_markdown(index=False))

report_lines.append("\n## Key Observations")
# Compute average drops
fft_scores = combined[combined['Strategy'] == 'FFT'].set_index('Model Family')['GLUE Score']
lora_scores = combined[combined['Strategy'] == 'LoRA'].set_index('Model Family')['GLUE Score']
kdlora_scores = combined[combined['Strategy'] == 'KD-LoRA'].set_index('Model Family')['GLUE Score']

avg_drop_fft_lora = (fft_scores - lora_scores).mean()
avg_drop_fft_kdlora = (fft_scores - kdlora_scores).mean()
avg_drop_lora_kdlora = (lora_scores - kdlora_scores).mean()

report_lines.append(f"- Average GLUE score drop FFT → LoRA: {avg_drop_fft_lora:.3f}")
report_lines.append(f"- Average GLUE score drop FFT → KD‑LoRA: {avg_drop_fft_kdlora:.3f}")
report_lines.append(f"- Average GLUE score drop LoRA → KD‑LoRA: {avg_drop_lora_kdlora:.3f}")

# Parameter reduction
param_fft = combined[combined['Strategy'] == 'FFT']['Parameters (M)'].mean()
param_lora = combined[combined['Strategy'] == 'LoRA']['Parameters (M)'].mean()
param_kdlora = combined[combined['Strategy'] == 'KD-LoRA']['Parameters (M)'].mean()
param_reduction_lora = (param_fft - param_lora) / param_fft * 100
param_reduction_kdlora = (param_fft - param_kdlora) / param_fft * 100

report_lines.append(f"- Parameter reduction LoRA vs FFT: {param_reduction_lora:.1f}%")
report_lines.append(f"- Parameter reduction KD‑LoRA vs FFT: {param_reduction_kdlora:.1f}%")

# Inference speedup
time_fft = combined[combined['Strategy'] == 'FFT']['Inference Time'].mean()
time_lora = combined[combined['Strategy'] == 'LoRA']['Inference Time'].mean()
time_kdlora = combined[combined['Strategy'] == 'KD-LoRA']['Inference Time'].mean()
speedup_lora = time_fft / time_lora
speedup_kdlora = time_fft / time_kdlora

report_lines.append(f"- Inference speedup LoRA vs FFT: {speedup_lora:.2f}x")
report_lines.append(f"- Inference speedup KD‑LoRA vs FFT: {speedup_kdlora:.2f}x")

report_lines.append("\n## Visualizations")
report_lines.append(f"Plots saved to `{output_dir}/`")
report_lines.append("- `scatter_params_vs_score.png`: GLUE Score vs Parameter count")
report_lines.append("- `scatter_time_vs_score.png`: GLUE Score vs Inference time")
report_lines.append("- `bar_score_per_param.png`: Score per million parameters")
report_lines.append("- `heatmap_normalized.png`: Normalized metrics heatmap")

report_lines.append("\n---")
report_lines.append(f"*Generated on {pd.Timestamp.now()}*")

report_path = "combined_analysis_report.md"
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"\nReport saved to {report_path}")
print(f"Plots saved to {output_dir}/")