import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Load CSV (comma separator ',')
df = pd.read_csv('kd-lora-table-ii.csv', sep=',')
# Clean column names (strip spaces)
df.columns = df.columns.str.strip()
# Show the loaded data
print("Loaded DataFrame shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# 3. Compute summary statistics: average score per model family per strategy (excluding the "Score" row)
# Separate tasks from score row
task_df = df[df['Task'] != 'Score'].copy()
score_row = df[df['Task'] == 'Score'].copy()

# Convert numeric columns to float (they may be strings)
numeric_cols = df.columns[1:]  # all columns except 'Task'
for col in numeric_cols:
    task_df[col] = pd.to_numeric(task_df[col], errors='coerce')
    score_row[col] = pd.to_numeric(score_row[col], errors='coerce')

# Compute average per model family per strategy across tasks
# We'll group by model family and strategy
# First, melt the dataframe to long format
long_df = task_df.melt(id_vars=['Task'], var_name='Model_Strategy', value_name='Score')

# Extract model family and strategy from column name
# Pattern: 'BERT-b/DBERT-b FFT' -> model_family='BERT-b/DBERT-b', strategy='FFT'
def split_model_strategy(col):
    # Find last space
    last_space = col.rfind(' ')
    model = col[:last_space]
    strategy = col[last_space+1:]
    return model, strategy

model_strategy = long_df['Model_Strategy'].apply(split_model_strategy)
long_df['Model_Family'] = model_strategy.apply(lambda x: x[0])
long_df['Strategy'] = model_strategy.apply(lambda x: x[1])

# Compute average per model family and strategy
avg_scores = long_df.groupby(['Model_Family', 'Strategy'])['Score'].mean().reset_index()
print("\nAverage scores per model family per strategy (excluding Score row):")
print(avg_scores)

# 4. Compute relative performance drop from FFT to LoRA, FFT to KD-LoRA, LoRA to KD-LoRA for each model family and task.
# Create pivot table with tasks as rows, model_family+strategy as columns
pivot = task_df.set_index('Task')
# We'll compute differences for each model family separately
model_families = ['BERT-b/DBERT-b', 'DeB-b/DeB-s', 'RoB-b/DRoB-b']
strategies = ['FFT', 'LoRA', 'KD-LoRA']

# Create a dictionary to store differences
diffs = {}
for model in model_families:
    fft_col = f"{model} FFT"
    lora_col = f"{model} LoRA"
    kd_lora_col = f"{model} KD-LoRA"
    
    # Ensure columns exist
    if fft_col in pivot.columns and lora_col in pivot.columns:
        diffs[f'{model}_FFT_to_LoRA'] = pivot[fft_col] - pivot[lora_col]
    if fft_col in pivot.columns and kd_lora_col in pivot.columns:
        diffs[f'{model}_FFT_to_KD_LoRA'] = pivot[fft_col] - pivot[kd_lora_col]
    if lora_col in pivot.columns and kd_lora_col in pivot.columns:
        diffs[f'{model}_LoRA_to_KD_LoRA'] = pivot[lora_col] - pivot[kd_lora_col]

diff_df = pd.DataFrame(diffs)
print("\nRelative performance drops (positive means first better than second):")
print(diff_df)

# 5. Identify tasks where KD-LoRA performs better than LoRA or even FFT.
# KD-LoRA > LoRA: negative drop in LoRA_to_KD_LoRA (since LoRA - KD-LoRA)
# KD-LoRA > FFT: negative drop in FFT_to_KD_LoRA
kd_lora_better_lora = {}
kd_lora_better_fft = {}
for model in model_families:
    col = f'{model}_LoRA_to_KD_LoRA'
    if col in diff_df.columns:
        better_tasks = diff_df.index[diff_df[col] < 0].tolist()
        kd_lora_better_lora[model] = better_tasks
    col2 = f'{model}_FFT_to_KD_LoRA'
    if col2 in diff_df.columns:
        better_tasks = diff_df.index[diff_df[col2] < 0].tolist()
        kd_lora_better_fft[model] = better_tasks

print("\nTasks where KD-LoRA performs better than LoRA:")
for model, tasks in kd_lora_better_lora.items():
    print(f"{model}: {tasks}")

print("\nTasks where KD-LoRA performs better than FFT:")
for model, tasks in kd_lora_better_fft.items():
    print(f"{model}: {tasks}")

# 6. Identify tasks where LoRA underperforms significantly.
# Define significant as drop > 2% from FFT? Let's compute average drop and identify outliers.
significant_threshold = 2.0  # percentage points
lora_underperform = {}
for model in model_families:
    col = f'{model}_FFT_to_LoRA'
    if col in diff_df.columns:
        underperform_tasks = diff_df.index[diff_df[col] > significant_threshold].tolist()
        lora_underperform[model] = underperform_tasks

print(f"\nTasks where LoRA underperforms FFT by >{significant_threshold}%:")
for model, tasks in lora_underperform.items():
    print(f"{model}: {tasks}")

# 7. Compute overall average scores across all tasks for each model and strategy (the "Score" row is already provided, but compute to verify)
computed_avg = task_df[numeric_cols].mean()
print("\nComputed average scores across tasks (verification):")
print(computed_avg)
print("\nProvided Score row:")
print(score_row[numeric_cols].iloc[0])

# 8. Create visualizations
# Ensure output directory
output_dir = 'analysis_plots'
os.makedirs(output_dir, exist_ok=True)

# Bar chart comparing average scores per model family per strategy.
plt.figure()
bar_data = avg_scores.pivot(index='Model_Family', columns='Strategy', values='Score')
bar_data.plot(kind='bar', rot=0)
plt.title('Average GLUE Score per Model Family and Strategy')
plt.ylabel('Average Accuracy (%)')
plt.xlabel('Model Family')
plt.legend(title='Strategy')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bar_avg_scores.png'), dpi=300)
plt.close()

# Heatmap of performance across tasks and strategies for each model family.
for model in model_families:
    # Filter columns for this model
    model_cols = [col for col in pivot.columns if model in col]
    model_data = pivot[model_cols].copy()
    # Sort strategies: FFT, LoRA, KD-LoRA
    strategy_order = ['FFT', 'LoRA', 'KD-LoRA']
    # Reorder columns
    ordered_cols = []
    for s in strategy_order:
        for col in model_cols:
            if s in col:
                ordered_cols.append(col)
    model_data = model_data[ordered_cols]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(model_data, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Accuracy (%)'})
    plt.title(f'Performance Heatmap for {model}')
    plt.xlabel('Strategy')
    plt.ylabel('Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{model.replace("/", "_")}.png'), dpi=300)
    plt.close()

# Line plot showing performance trends across tasks (x-axis tasks, y-axis score, colored by strategy).
# We'll create separate plot for each model family
for model in model_families:
    plt.figure()
    tasks = task_df['Task'].tolist()
    for strategy in strategies:
        col = f"{model} {strategy}"
        if col in task_df.columns:
            scores = task_df[col].values
            plt.plot(tasks, scores, marker='o', label=strategy)
    plt.title(f'GLUE Performance across Tasks for {model}')
    plt.xlabel('Task')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Strategy')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'line_trends_{model.replace("/", "_")}.png'), dpi=300)
    plt.close()

print(f"\nPlots saved to '{output_dir}' directory.")

# 9. Generate concise textual summary
print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)

# Which model family performs best overall?
# Use the provided Score row for overall average
overall_scores = score_row[numeric_cols].iloc[0]
# Group by model family (average across strategies)
model_family_scores = {}
for model in model_families:
    model_cols = [col for col in overall_scores.index if model in col]
    model_family_scores[model] = overall_scores[model_cols].mean()

best_model = max(model_family_scores, key=model_family_scores.get)
print(f"Best performing model family: {best_model} (average across strategies: {model_family_scores[best_model]:.1f}%)")

# Which strategy performs best overall?
strategy_scores = {}
for strategy in strategies:
    strategy_cols = [col for col in overall_scores.index if strategy in col]
    strategy_scores[strategy] = overall_scores[strategy_cols].mean()

best_strategy = max(strategy_scores, key=strategy_scores.get)
print(f"Best performing strategy: {best_strategy} (average across models: {strategy_scores[best_strategy]:.1f}%)")

# How much performance drop occurs with KD-LoRA compared to FFT (average across tasks and models)?
# Compute average drop across all model families and tasks
fft_kd_drops = []
for model in model_families:
    col = f'{model}_FFT_to_KD_LoRA'
    if col in diff_df.columns:
        fft_kd_drops.extend(diff_df[col].values)
avg_drop = np.mean(fft_kd_drops)
print(f"Average performance drop from FFT to KD-LoRA: {avg_drop:.2f} percentage points")

# Which tasks are most robust to distillation? Which are most sensitive?
# Robust: smallest drop from FFT to KD-LoRA across models
# Compute average drop per task across models
task_drops = {}
for task in task_df['Task']:
    drops = []
    for model in model_families:
        col = f'{model}_FFT_to_KD_LoRA'
        if col in diff_df.columns:
            drops.append(diff_df.loc[task, col])
    task_drops[task] = np.mean(drops)

# Sort by drop magnitude (absolute)
robust_tasks = sorted(task_drops.items(), key=lambda x: abs(x[1]))[:3]
sensitive_tasks = sorted(task_drops.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
print("\nMost robust tasks to distillation (smallest drop from FFT to KD-LoRA):")
for task, drop in robust_tasks:
    print(f"  {task}: {drop:.2f}% drop")
print("\nMost sensitive tasks to distillation (largest drop from FFT to KD-LoRA):")
for task, drop in sensitive_tasks:
    print(f"  {task}: {drop:.2f}% drop")

# Any surprising results (e.g., KD-LoRA outperforming LoRA).
print("\nSurprising results (KD-LoRA outperforming LoRA):")
for model, tasks in kd_lora_better_lora.items():
    if tasks:
        print(f"  {model}: {tasks}")
print("\nSurprising results (KD-LoRA outperforming FFT):")
for model, tasks in kd_lora_better_fft.items():
    if tasks:
        print(f"  {model}: {tasks}")

print("\n" + "="*60)
print("Analysis complete.")
print("="*60)