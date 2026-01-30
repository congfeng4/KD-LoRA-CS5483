#!/usr/bin/env python3
import pandas as pd
import numpy as np

print("=== Available Data Analysis ===")

# 1. Check what per-task data exists
print("\n1. Per-task data files:")
print("table_i_mrlora.csv - FFT, Teacher MR-LoRA, Student MR-LoRA")
print("table_i_detailed.csv - FFT, LoRA (Teacher), KD-LoRA (Student LoRA)")

mrlora_df = pd.read_csv('table_i_mrlora.csv')
detailed_df = pd.read_csv('table_i_detailed.csv')

print(f"\ntable_i_mrlora.csv columns: {list(mrlora_df.columns)}")
print(f"table_i_detailed.csv columns: {list(detailed_df.columns)}")

# 2. Check glue_averages.csv for other variants
glue_df = pd.read_csv('glue_averages.csv')
print(f"\n2. GLUE averages file has {len(glue_df)} rows")
print("Methods in glue_averages.csv:", glue_df['method'].unique())

# 3. Check tradeoff_data.csv for parameter counts
tradeoff_df = pd.read_csv('tradeoff_data.csv')
print(f"\n3. Tradeoff data file has {len(tradeoff_df)} rows")
print("Methods in tradeoff_data.csv:", tradeoff_df['method'].unique())

# 4. Check for student data for other variants
print("\n4. Looking for student variants...")
# From table_i_detailed.csv, KD-LoRA is student LoRA
# From table_i_mrlora.csv, Student MR-LoRA is student MR-LoRA
# Are there student versions of AdaLoRA, DoRA, OLoRA, etc.?
print("Student data found:")
print("  - KD-LoRA (Student LoRA) in table_i_detailed.csv")
print("  - Student MR-LoRA in table_i_mrlora.csv")
print("\nNo student data found for: AdaLoRA, DoRA, OLoRA, RS-LoRA, MR-LoRA-RS")

# 5. Check table_i_glue_scores.csv
scores_df = pd.read_csv('table_i_glue_scores.csv')
print(f"\n5. table_i_glue_scores.csv columns: {list(scores_df.columns)}")

# 6. What data do we actually have for each variant?
print("\n6. Data availability by variant:")
variants = ['lora', 'mrlora', 'adalora', 'dora', 'olora', 'rslora', 'mrlora-rs']

for variant in variants:
    print(f"\n{variant}:")
    # Teacher GLUE
    if variant in glue_df['method'].values:
        teacher_glue = glue_df[glue_df['method'] == variant]
        print(f"  Teacher GLUE: {len(teacher_glue)} model families")
    
    # Teacher params
    if variant in tradeoff_df['method'].values:
        teacher_params = tradeoff_df[tradeoff_df['method'] == variant]
        print(f"  Teacher params: {len(teacher_params)} model families")
    
    # Student data?
    if variant == 'lora':
        print("  Student: KD-LoRA in table_i_detailed.csv")
    elif variant == 'mrlora':
        print("  Student: Student MR-LoRA in table_i_mrlora.csv")
    else:
        print("  Student: No data found")