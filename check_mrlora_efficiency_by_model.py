import pandas as pd

# Load efficiency metrics
df = pd.read_csv('efficiency_metrics_all.csv')

# Filter for mrlora
mrlora_df = df[df['peft'] == 'mrlora'].copy()

print("MRLORA EFFICIENCY METRICS BY MODEL FAMILY")
print("=" * 60)

# Group by model family and variant (strategy)
grouped = mrlora_df.groupby(['model_family', 'variant']).agg({
    'metric_value': ['mean', 'std', 'count'],
    'train_time': 'mean',
    'trainable_params_count': 'mean',
    'avg_memory_allocated_mb': 'mean',
    'train_samples_per_second': 'mean',
    'total_flos': 'mean'
}).round(3)

# Display results
for (model_family, variant), group_data in grouped.iterrows():
    print(f"\n{model_family:10s} - {variant:10s}:")
    print(f"  Performance: {group_data[('metric_value', 'mean')]:.4f} (±{group_data[('metric_value', 'std')]:.4f}, n={group_data[('metric_value', 'count')]:.0f})")
    print(f"  Train Time: {group_data[('train_time', 'mean')]:.1f}s")
    print(f"  Parameters: {group_data[('trainable_params_count', 'mean')]:.3f}M")
    print(f"  Memory: {group_data[('avg_memory_allocated_mb', 'mean')]:.1f}MB")
    print(f"  Throughput: {group_data[('train_samples_per_second', 'mean')]:.1f} samples/s")
    print(f"  FLOPs: {float(group_data[('total_flos', 'mean')]):.2e}")

print("\n\nCOMPARISON ACROSS MODEL FAMILIES")
print("=" * 60)

# Compare average across model families
for variant in ['lora', 'kd-lora']:
    print(f"\n{variant:10s} strategy:")
    variant_data = mrlora_df[mrlora_df['variant'] == variant]
    if not variant_data.empty:
        for model in variant_data['model_family'].unique():
            model_data = variant_data[variant_data['model_family'] == model]
            print(f"  {model:10s}: {model_data['metric_value'].mean():.4f} (n={len(model_data)})")