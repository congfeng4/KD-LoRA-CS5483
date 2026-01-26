#!/usr/bin/env python3
"""
Analyze seed variability for Table II experiments.
Compute standard deviations across seeds for each (task, strategy, variant).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Reuse data collection functions from create_table_ii.py
# (or import directly)
import sys
sys.path.append('.')
from create_table_ii import collect_all_data

print("=" * 80)
print("SEED VARIABILITY ANALYSIS FOR TABLE II")
print("=" * 80)

# Collect raw data
print("\nCollecting data from results directory...")
raw_df = collect_all_data()

if raw_df.empty:
    print("No data found!")
    sys.exit(1)

print(f"Collected {len(raw_df)} data records")
print(f"Unique seeds found: {sorted(raw_df['Seed'].unique())}")

# Group by task, strategy, variant and compute statistics
group_cols = ['Task', 'Strategy', 'PEFT Variant']
grouped = raw_df.groupby(group_cols)

stats_list = []
for name, group in grouped:
    task, strategy, variant = name
    values = group['Metric Value'].tolist()
    seeds = group['Seed'].tolist()
    
    stats_list.append({
        'Task': task,
        'Strategy': strategy,
        'PEFT Variant': variant,
        'Mean': np.mean(values),
        'Std': np.std(values),
        'Min': np.min(values),
        'Max': np.max(values),
        'Count': len(values),
        'Seeds': seeds
    })

stats_df = pd.DataFrame(stats_list)

# Separate tables for LoRA-only and KD-LoRA
print("\n1. SEED VARIABILITY FOR LoRA-only STRATEGY")
print("-" * 40)
lora_stats = stats_df[stats_df['Strategy'] == 'LoRA']
if not lora_stats.empty:
    # Pivot to show std per task and variant
    pivot_std = lora_stats.pivot_table(
        index='Task',
        columns='PEFT Variant',
        values='Std',
        aggfunc='first'
    )
    pivot_count = lora_stats.pivot_table(
        index='Task',
        columns='PEFT Variant',
        values='Count',
        aggfunc='first'
    )
    
    print("\nStandard deviations across seeds:")
    print(pivot_std.round(4))
    
    print("\nNumber of seeds per cell:")
    print(pivot_count)
    
    # Save to CSV
    pivot_std.to_csv('seed_std_lora_only.csv')
    print(f"\nStandard deviations saved to seed_std_lora_only.csv")
else:
    print("No LoRA-only data")

print("\n2. SEED VARIABILITY FOR KD-LoRA STRATEGY")
print("-" * 40)
kd_lora_stats = stats_df[stats_df['Strategy'] == 'KD-LoRA']
if not kd_lora_stats.empty:
    pivot_std = kd_lora_stats.pivot_table(
        index='Task',
        columns='PEFT Variant',
        values='Std',
        aggfunc='first'
    )
    pivot_count = kd_lora_stats.pivot_table(
        index='Task',
        columns='PEFT Variant',
        values='Count',
        aggfunc='first'
    )
    
    print("\nStandard deviations across seeds:")
    print(pivot_std.round(4))
    
    print("\nNumber of seeds per cell:")
    print(pivot_count)
    
    # Save to CSV
    pivot_std.to_csv('seed_std_kd_lora.csv')
    print(f"\nStandard deviations saved to seed_std_kd_lora.csv")
else:
    print("No KD-LoRA data")

# Summary statistics
print("\n3. OVERVIEW OF SEED AVAILABILITY")
print("-" * 40)
summary = stats_df.groupby(['Strategy', 'PEFT Variant']).agg({
    'Count': ['mean', 'min', 'max']
}).round(2)
print(summary)

# Identify experiments with multiple seeds
multi_seed = stats_df[stats_df['Count'] > 1]
if not multi_seed.empty:
    print(f"\nExperiments with multiple seeds: {len(multi_seed)}")
    # Compute average std across variants
    avg_std_by_variant = multi_seed.groupby(['Strategy', 'PEFT Variant'])['Std'].mean().reset_index()
    print("\nAverage standard deviation (across tasks) for variants with multiple seeds:")
    print(avg_std_by_variant.round(4))
else:
    print("\nNo experiments with multiple seeds")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)