#!/usr/bin/env python3
"""
Simple analysis of BERT data.
"""

import pandas as pd
import numpy as np

# Load the CSV file
try:
    df = pd.read_csv('bert_metrics.csv')
except:
    # Try skipping first row if there's an issue
    df = pd.read_csv('bert_metrics.csv', skiprows=1)

print("Data shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check unique values
print("\n=== Unique values ===")
print("Variant:", df['variant'].unique())
print("Task:", df['task'].unique())
print("PEFT:", df['peft'].unique())
print("Type:", df['type'].unique())

# Count by type
print("\n=== Count by type ===")
print(df['type'].value_counts())

# For type 0 (FFT)
print("\n=== FFT (type 0) ===")
fft = df[df['type'] == 0]
print(f"Rows: {len(fft)}")
print(f"Tasks: {sorted(fft['task'].unique())}")
print(f"Seeds: {sorted(fft['seed'].unique())}")

# For type 2 (LoRA)
print("\n=== LoRA (type 2) ===")
lora = df[df['type'] == 2]
print(f"Rows: {len(lora)}")
print(f"PEFT variants: {sorted(lora['peft'].unique())}")
print(f"Tasks: {sorted(lora['task'].unique())}")
print(f"Seeds: {sorted(lora['seed'].unique())}")

# For type 1 (KD-LoRA)
print("\n=== KD-LoRA (type 1) ===")
kd_lora = df[df['type'] == 1]
print(f"Rows: {len(kd_lora)}")
print(f"PEFT variants: {sorted(kd_lora['peft'].unique())}")
print(f"Tasks: {sorted(kd_lora['task'].unique())}")
print(f"Seeds: {sorted(kd_lora['seed'].unique())}")

# Check MNLI handling
print("\n=== MNLI Data ===")
mnli_data = df[df['task'] == 'mnli']
print(f"MNLI rows: {len(mnli_data)}")
print("Metric keys for MNLI:", mnli_data['metric_key'].unique())

# Check for mismatched accuracy
mismatched = df[df['metric_key'] == 'mismatched_accuracy']
print(f"\nMismatched accuracy rows: {len(mismatched)}")
print("Tasks with mismatched_accuracy:", mismatched['task'].unique())