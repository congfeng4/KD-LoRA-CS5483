#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys

# Load paper table
paper_df = pd.read_csv('src/kd-lora-table-I.csv')
print("Paper table:")
print(paper_df.head())

# Map paper columns to (model_family, variant)
# Columns: Task, BERT-b/DBERT-b FFT, BERT-b/DBERT-b LoRA, BERT-b/DBERT-b KD-LoRA, DeB-b/DeB-s FFT, ...
# We'll parse column names
columns = paper_df.columns.tolist()
print("Columns:", columns)

# Mapping from paper model family to our model_family keys
model_map = {
    'BERT-b/DBERT-b': 'bert',
    'DeB-b/DeB-s': 'deberta',
    'RoB-b/DRoB-b': 'roberta'
}
variant_map = {'FFT': 'fft', 'LoRA': 'lora', 'KD-LoRA': 'kd-lora'}

# Transform paper table to long format
paper_records = []
for _, row in paper_df.iterrows():
    task = row['Task']
    for col in columns[1:]:
        # col format like "BERT-b/DBERT-b FFT"
        parts = col.rsplit(' ', 1)
        if len(parts) != 2:
            continue
        model_str, variant_str = parts
        model_family = model_map.get(model_str)
        variant = variant_map.get(variant_str)
        if model_family is None or variant is None:
            continue
        value = row[col]
        # Some values might be strings with commas? They are numeric.
        try:
            value_float = float(value)
        except:
            continue
        paper_records.append({
            'task': task,
            'model_family': model_family,
            'variant': variant,
            'paper_metric': value_float
        })
paper_long = pd.DataFrame(paper_records)
print("\nPaper long format sample:")
print(paper_long.head())

# Map our task names to paper task names
task_map = {
    'cola': 'CoLA',
    'mnli': 'MNLI_m',  # matched (fallback)
    'mnli_matched': 'MNLI_m',
    'mnli_mismatched': 'MNLI_mm',
    'mrpc': 'MRPC',
    'qqp': 'QQP',
    'qnli': 'QNLI',
    'rte': 'RTE',
    'sst2': 'SST-2',
    'stsb': 'STS-B',
    'wnli': 'WNLI'
}

# Load our median results (average across seeds)
our_df = pd.read_csv('test_median3.csv')
# Convert metric to percentage (multiply by 100)
our_df['metric_pct'] = our_df['median_metric'] * 100
# Map task names
our_df['task_paper'] = our_df['task'].map(task_map)
# Average across seeds
our_avg = our_df.groupby(['task_paper', 'model_family', 'variant']).agg({'metric_pct': 'mean'}).reset_index()
our_avg = our_avg.rename(columns={'metric_pct': 'our_metric', 'task_paper': 'task'})
print("\nOur average across seeds:")
print(our_avg.head())

# Merge
merged = pd.merge(our_avg, paper_long, on=['task', 'model_family', 'variant'], how='inner')
if merged.empty:
    print("No matching tasks found. Check task naming.")
    sys.exit(1)

merged['diff'] = merged['our_metric'] - merged['paper_metric']
merged['abs_diff'] = merged['diff'].abs()
print("\nComparison:")
print(merged.to_string())

# Summary statistics
print("\nSummary of differences:")
print(f"Number of comparisons: {len(merged)}")
print(f"Mean absolute difference: {merged['abs_diff'].mean():.2f}")
print(f"Max absolute difference: {merged['abs_diff'].max():.2f}")
print(f"Min absolute difference: {merged['abs_diff'].min():.2f}")

# Breakdown by variant
for variant in ['fft', 'lora', 'kd-lora']:
    sub = merged[merged['variant'] == variant]
    if not sub.empty:
        print(f"\nVariant {variant}:")
        print(f"  Mean abs diff: {sub['abs_diff'].mean():.2f}")
        print(f"  Max abs diff: {sub['abs_diff'].max():.2f}")
        print(f"  Min abs diff: {sub['abs_diff'].min():.2f}")
        # List large differences > 2.0
        large = sub[sub['abs_diff'] > 2.0]
        if not large.empty:
            print(f"  Large differences (>2.0):")
            for _, row in large.iterrows():
                print(f"    {row['task']} {row['model_family']}: ours {row['our_metric']:.1f} paper {row['paper_metric']:.1f} diff {row['diff']:.1f}")

# Note: MNLI mismatched not included (paper has separate column MNLI_mm). We only have matched accuracy.
# If we want mismatched, need to modify collection.