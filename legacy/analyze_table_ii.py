#!/usr/bin/env python3
"""
Analyze Table II results for KD-LoRA paper.
Compute performance degradation relative to FFT baseline, rank variants, and identify best performers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load generated tables
table_iia = pd.read_csv('table_iia_results.csv', index_col=0)
table_iib = pd.read_csv('table_iib_results.csv', index_col=0)

print("=" * 80)
print("ANALYSIS OF TABLE II RESULTS")
print("=" * 80)

# Extract average rows (last row)
avg_iia = table_iia.loc['Average']
avg_iib = table_iib.loc['Average']

print("\n1. AVERAGE PERFORMANCE ACROSS GLUE TASKS")
print("-" * 40)
print("Table IIa (LoRA-only strategy):")
for variant in avg_iia.index:
    print(f"  {variant:10s}: {avg_iia[variant]:.4f}")

print("\nTable IIb (KD-LoRA strategy):")
for variant in avg_iib.index:
    print(f"  {variant:10s}: {avg_iib[variant]:.4f}")

# Compute relative performance drop compared to FFT
def compute_relative_drop(avg_series):
    """Compute percentage drop relative to FFT baseline."""
    if 'FFT' not in avg_series.index:
        return pd.Series()
    fft_score = avg_series['FFT']
    rel_drop = {}
    for variant in avg_series.index:
        if variant == 'FFT':
            continue
        variant_score = avg_series[variant]
        if pd.isna(variant_score):
            rel_drop[variant] = np.nan
        else:
            rel_drop[variant] = (fft_score - variant_score) / fft_score * 100
    return pd.Series(rel_drop)

rel_drop_iia = compute_relative_drop(avg_iia)
rel_drop_iib = compute_relative_drop(avg_iib)

print("\n2. RELATIVE PERFORMANCE DROP vs FFT BASELINE (%)")
print("-" * 40)
print("Table IIa (LoRA-only):")
for variant, drop in rel_drop_iia.items():
    if not pd.isna(drop):
        print(f"  {variant:10s}: {drop:.2f}%")

print("\nTable IIb (KD-LoRA):")
for variant, drop in rel_drop_iib.items():
    if not pd.isna(drop):
        print(f"  {variant:10s}: {drop:.2f}%")

# Rank variants by performance (excluding FFT)
def rank_variants(avg_series):
    """Rank variants by average score (descending)."""
    variants = [v for v in avg_series.index if v != 'FFT' and not pd.isna(avg_series[v])]
    sorted_variants = sorted(variants, key=lambda v: avg_series[v], reverse=True)
    return sorted_variants

rank_iia = rank_variants(avg_iia)
rank_iib = rank_variants(avg_iib)

print("\n3. VARIANT RANKING (best to worst)")
print("-" * 40)
print("Table IIa (LoRA-only):")
for i, variant in enumerate(rank_iia, 1):
    print(f"  {i}. {variant:10s}: {avg_iia[variant]:.4f}")

print("\nTable IIb (KD-LoRA):")
for i, variant in enumerate(rank_iib, 1):
    print(f"  {i}. {variant:10s}: {avg_iib[variant]:.4f}")

# Identify tasks where variants outperform FFT
def find_outperforming_tasks(table_df):
    """Find tasks where any variant outperforms FFT."""
    outperforming = {}
    if 'FFT' not in table_df.columns:
        return outperforming
    
    # Exclude 'Average' row
    task_rows = table_df.index[table_df.index != 'Average']
    
    for task in task_rows:
        fft_score = table_df.loc[task, 'FFT']
        if pd.isna(fft_score):
            continue
        
        for variant in table_df.columns:
            if variant == 'FFT':
                continue
            variant_score = table_df.loc[task, variant]
            if not pd.isna(variant_score) and variant_score > fft_score:
                if task not in outperforming:
                    outperforming[task] = []
                outperforming[task].append((variant, variant_score - fft_score))
    
    return outperforming

outperform_iia = find_outperforming_tasks(table_iia)
outperform_iib = find_outperforming_tasks(table_iib)

print("\n4. TASKS WHERE VARIANTS OUTPERFORM FFT BASELINE")
print("-" * 40)
print("Table IIa (LoRA-only):")
if outperform_iia:
    for task, variants in outperform_iia.items():
        print(f"  {task}:")
        for variant, margin in variants:
            print(f"    {variant}: +{margin:.4f}")
else:
    print("  No variants outperform FFT baseline")

print("\nTable IIb (KD-LoRA):")
if outperform_iib:
    for task, variants in outperform_iib.items():
        print(f"  {task}:")
        for variant, margin in variants:
            print(f"    {variant}: +{margin:.4f}")
else:
    print("  No variants outperform FFT baseline")

# Compute per-task relative performance
print("\n5. PER-TASK RELATIVE PERFORMANCE (top 3 variants per task)")
print("-" * 40)

def top_variants_per_task(table_df, n=3):
    """Return top n variants per task (excluding FFT)."""
    top_results = {}
    # Exclude 'Average' row
    task_rows = table_df.index[table_df.index != 'Average']
    
    for task in task_rows:
        row = table_df.loc[task]
        # Filter out FFT and NaN
        variants = [v for v in row.index if v != 'FFT' and not pd.isna(row[v])]
        sorted_variants = sorted(variants, key=lambda v: row[v], reverse=True)
        top_results[task] = sorted_variants[:n]
    return top_results

top_iia = top_variants_per_task(table_iia)
top_iib = top_variants_per_task(table_iib)

print("Table IIa (LoRA-only):")
for task, variants in top_iia.items():
    print(f"  {task:15s}: {', '.join(variants)}")

print("\nTable IIb (KD-LoRA):")
for task, variants in top_iib.items():
    print(f"  {task:15s}: {', '.join(variants)}")

# Generate summary statistics
print("\n6. SUMMARY STATISTICS")
print("-" * 40)
print(f"Table IIa - Number of tasks: {len(table_iia) - 1}")
print(f"Table IIb - Number of tasks: {len(table_iib) - 1}")

# Count missing values per variant
def count_missing(table_df):
    """Count missing values per variant (excluding 'Average' row)."""
    task_rows = table_df.index[table_df.index != 'Average']
    missing_counts = {}
    for variant in table_df.columns:
        missing = sum(pd.isna(table_df.loc[task_rows, variant]))
        missing_counts[variant] = missing
    return missing_counts

missing_iia = count_missing(table_iia)
missing_iib = count_missing(table_iib)

print("\nMissing data points per variant (excluding 'Average' row):")
print("Table IIa:")
for variant, count in missing_iia.items():
    if count > 0:
        print(f"  {variant:10s}: {count} missing")

print("\nTable IIb:")
for variant, count in missing_iib.items():
    if count > 0:
        print(f"  {variant:10s}: {count} missing")

# Create visualization: relative performance drop
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Table IIa drop
variants_iia = [v for v in rel_drop_iia.index if not pd.isna(rel_drop_iia[v])]
drops_iia = [rel_drop_iia[v] for v in variants_iia]
colors_iia = ['red' if drop > 5 else 'orange' if drop > 2 else 'green' for drop in drops_iia]
ax1.bar(variants_iia, drops_iia, color=colors_iia)
ax1.set_xlabel('PEFT Variant')
ax1.set_ylabel('Performance Drop vs FFT (%)')
ax1.set_title('Table IIa: LoRA-only Strategy')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.tick_params(axis='x', rotation=45)

# Table IIb drop
variants_iib = [v for v in rel_drop_iib.index if not pd.isna(rel_drop_iib[v])]
drops_iib = [rel_drop_iib[v] for v in variants_iib]
colors_iib = ['red' if drop > 5 else 'orange' if drop > 2 else 'green' for drop in drops_iib]
ax2.bar(variants_iib, drops_iib, color=colors_iib)
ax2.set_xlabel('PEFT Variant')
ax2.set_ylabel('Performance Drop vs FFT (%)')
ax2.set_title('Table IIb: KD-LoRA Strategy')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('table_ii_performance_drop.png', dpi=150)
print(f"\nPerformance drop visualization saved to table_ii_performance_drop.png")

# Create summary table
summary_data = []
for variant in set(list(avg_iia.index) + list(avg_iib.index)):
    if variant == 'FFT':
        continue
    row = {'Variant': variant}
    if variant in avg_iia:
        row['LoRA-only Score'] = avg_iia[variant]
        row['LoRA-only Drop (%)'] = rel_drop_iia.get(variant, np.nan)
    else:
        row['LoRA-only Score'] = np.nan
        row['LoRA-only Drop (%)'] = np.nan
    
    if variant in avg_iib:
        row['KD-LoRA Score'] = avg_iib[variant]
        row['KD-LoRA Drop (%)'] = rel_drop_iib.get(variant, np.nan)
    else:
        row['KD-LoRA Score'] = np.nan
        row['KD-LoRA Drop (%)'] = np.nan
    
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('LoRA-only Score', ascending=False)
summary_path = 'table_ii_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"Summary table saved to {summary_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)