#!/usr/bin/env python3
"""
Analyze BERT data availability for Table II.
"""

import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('bert_metrics.csv', skiprows=1)  # Skip "Found 200 files" line

print("Data shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Check variant distribution
print("\n=== Variant Distribution ===")
print(df['variant'].value_counts())

# Check peft distribution per variant
print("\n=== PEFT Distribution per Variant ===")
print(pd.crosstab(df['variant'], df['peft']))

# Check task distribution per variant
print("\n=== Task Distribution per Variant ===")
print(pd.crosstab(df['variant'], df['task']))

# Check type distribution (0=FFT, 1=KD-LoRA, 2=LoRA)
print("\n=== Type Distribution ===")
print(df['type'].value_counts())
print("\nType mapping:")
print("0: FFT, 1: KD-LoRA, 2: LoRA")

# Check seeds per variant-task combination
print("\n=== Seed Coverage ===")
for variant in df['variant'].unique():
    variant_df = df[df['variant'] == variant]
    print(f"\n{variant}:")
    for task in variant_df['task'].unique():
        task_df = variant_df[variant_df['task'] == task]
        seeds = sorted(task_df['seed'].unique())
        print(f"  {task}: {len(seeds)} seeds - {seeds}")

# Check metric keys
print("\n=== Metric Keys by Task ===")
for task in df['task'].unique():
    task_df = df[df['task'] == task]
    metrics = task_df['metric_key'].unique()
    print(f"{task}: {metrics}")

print("\n=== Summary for Table II ===")
print("\nFFT (type 0):")
fft_df = df[df['type'] == 0]
print(f"  Tasks: {sorted(fft_df['task'].unique())}")
print(f"  Seeds: {sorted(fft_df['seed'].unique())}")

print("\nLoRA (type 2):")
lora_df = df[df['type'] == 2]
print(f"  PEFT variants: {sorted(lora_df['peft'].unique())}")
print(f"  Tasks: {sorted(lora_df['task'].unique())}")
print(f"  Seeds: {sorted(lora_df['seed'].unique())}")

print("\nKD-LoRA (type 1):")
kd_lora_df = df[df['type'] == 1]
print(f"  PEFT variants: {sorted(kd_lora_df['peft'].unique())}")
print(f"  Tasks: {sorted(kd_lora_df['task'].unique())}")
print(f"  Seeds: {sorted(kd_lora_df['seed'].unique())}")

# Check for missing combinations
print("\n=== Missing Combinations ===")
all_tasks = sorted(df['task'].unique())
all_pefts = sorted(df['peft'].unique())

print("\nLoRA (type 2) missing:")
for task in all_tasks:
    for peft in all_pefts:
        if peft == 'lora':  # FFT uses 'lora' as peft but type=0
            continue
        subset = df[(df['type'] == 2) & (df['task'] == task) & (df['peft'] == peft)]
        if subset.empty:
            print(f"  {task} - {peft}")

print("\nKD-LoRA (type 1) missing:")
for task in all_tasks:
    for peft in all_pefts:
        if peft == 'lora':  # FFT uses 'lora' as peft but type=0
            continue
        subset = df[(df['type'] == 1) & (df['task'] == task) & (df['peft'] == peft)]
        if subset.empty:
            print(f"  {task} - {peft}")