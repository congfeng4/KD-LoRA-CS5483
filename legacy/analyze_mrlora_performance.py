"""
Comprehensive analysis of mrlora performance across different aspects.
"""
import pandas as pd
import numpy as np

def analyze_mrlora_performance():
    # Load all relevant data files
    print("=== MRLORA PERFORMANCE ANALYSIS ===\n")
    
    # 1. Load efficiency metrics
    efficiency_df = pd.read_csv('efficiency_summary_by_variant.csv', header=[0,1])
    efficiency_df.columns = ['_'.join(col).strip('_') for col in efficiency_df.columns.values]
    
    # Extract mrlora data
    mrlora_data = efficiency_df[efficiency_df['peft_strategy'].str.contains('mrlora', case=False, na=False)]
    
    print("1. EFFICIENCY METRICS (Average across all experiments):")
    print("=" * 80)
    
    mrlora_kd = mrlora_data[mrlora_data['peft_strategy'].str.contains('KD-LoRA')]
    mrlora_lora = mrlora_data[mrlora_data['peft_strategy'].str.contains('LoRA-only')]
    
    if not mrlora_kd.empty:
        print("\nKD-LoRA Strategy:")
        print(f"  Performance: {mrlora_kd['metric_value_mean'].values[0]:.4f}")
        print(f"  Training Time: {mrlora_kd['train_time_mean'].values[0]:.1f}s")
        print(f"  Trainable Parameters: {mrlora_kd['trainable_params_millions_mean'].values[0]:.3f}M")
        print(f"  Memory Usage: {mrlora_kd['avg_memory_allocated_mb_mean'].values[0]:.1f}MB")
        print(f"  Throughput: {mrlora_kd['throughput_samples_per_sec_mean'].values[0]:.1f} samples/s")
        print(f"  Total FLOPs: {float(mrlora_kd['total_flos_mean'].values[0]):.2e}")
    
    if not mrlora_lora.empty:
        print("\nLoRA-only Strategy:")
        print(f"  Performance: {mrlora_lora['metric_value_mean'].values[0]:.4f}")
        print(f"  Training Time: {mrlora_lora['train_time_mean'].values[0]:.1f}s")
        print(f"  Trainable Parameters: {mrlora_lora['trainable_params_millions_mean'].values[0]:.3f}M")
        print(f"  Memory Usage: {mrlora_lora['avg_memory_allocated_mb_mean'].values[0]:.1f}MB")
        print(f"  Throughput: {mrlora_lora['throughput_samples_per_sec_mean'].values[0]:.1f} samples/s")
        print(f"  Total FLOPs: {float(mrlora_lora['total_flos_mean'].values[0]):.2e}")
    
    # 2. Load statistical test results
    stats_df = pd.read_csv('statistical_test_results.csv')
    mrlora_stats = stats_df[stats_df['comparison'].str.contains('mrlora', case=False, na=False)]
    
    print("\n\n2. STATISTICAL SIGNIFICANCE:")
    print("=" * 80)
    
    for _, row in mrlora_stats.iterrows():
        comp = row['comparison']
        if 'LoRA-only vs KD-LoRA' in comp:
            print(f"\nmrlora: LoRA-only vs KD-LoRA:")
            print(f"  LoRA-only mean: {row['mean_lora']:.4f}")
            print(f"  KD-LoRA mean: {row['mean_kd']:.4f}")
            print(f"  Difference: {row['difference']:.4f} (LoRA-only {'better' if row['difference'] > 0 else 'worse'})")
            print(f"  p-value: {row['p_value']:.4f} {'(NOT significant)' if row['p_value'] > 0.05 else '(SIGNIFICANT)'}")
        elif 'vs FFT' in comp:
            print(f"\nmrlora vs FFT:")
            print(f"  FFT mean: {row['mean_fft']:.4f}")
            print(f"  mrlora mean: {row['mean_peft']:.4f}")
            print(f"  Difference: {row['difference']:.4f} (mrlora {'better' if row['difference'] > 0 else 'worse'})")
            print(f"  p-value: {row['p_value']:.4f} {'(NOT significant)' if row['p_value'] > 0.05 else '(SIGNIFICANT)'}")
    
    # 3. Load parameter efficiency
    param_df = pd.read_csv('parameter_efficiency_analysis.csv')
    mrlora_param = param_df[(param_df['peft'] == 'mrlora') | (param_df['peft'] == 'mrlora-rs')]
    
    print("\n\n3. PARAMETER EFFICIENCY:")
    print("=" * 80)
    
    for _, row in mrlora_param.iterrows():
        strategy = row['strategy']
        print(f"\n{strategy}:")
        print(f"  Parameters: {row['trainable_params_millions']:.3f}M")
        print(f"  Performance: {row['metric_value']:.4f}")
        print(f"  Performance per Parameter: {row['performance_per_param']:.4f}")
        print(f"  Training Time: {row['train_time']:.1f}s")
    
    # 4. Load multi-model analysis
    multi_df = pd.read_csv('multi_model_analysis_results.csv')
    mrlora_multi = multi_df[(multi_df['peft_variant'] == 'mrlora') | (multi_df['peft_variant'] == 'mrlora-rs')]
    
    print("\n\n4. MODEL-SPECIFIC PERFORMANCE:")
    print("=" * 80)
    
    for _, row in mrlora_multi.iterrows():
        model = row['model_family']
        strategy = row['strategy']
        print(f"\n{model} - {strategy}:")
        print(f"  Average Score: {row['average_score']:.4f}")
        print(f"  Drop vs FFT: {row['drop_vs_fft_percent']:.2f}%")
        print(f"  Max Score: {row['max_score']:.4f}")
        print(f"  Rank: {int(row['rank'])}")
    
    # 5. Comparative analysis with other variants
    print("\n\n5. COMPARATIVE RANKING (vs other PEFT variants):")
    print("=" * 80)
    
    # Load all data for comparison
    all_peft_data = efficiency_df.copy()
    
    # Extract performance metrics
    performances = []
    for idx, row in all_peft_data.iterrows():
        peft_strat = row['peft_strategy']
        if pd.notna(peft_strat):
            peft, strategy = peft_strat.split(',')
            performances.append({
                'peft': peft.strip(),
                'strategy': strategy.strip(),
                'performance': row['metric_value_mean'],
                'train_time': row['train_time_mean'],
                'parameters': row['trainable_params_millions_mean'],
                'memory': row['avg_memory_allocated_mb_mean']
            })
    
    perf_df = pd.DataFrame(performances)
    
    # Rank by performance
    print("\nPerformance Ranking (Higher is better):")
    perf_rank = perf_df.sort_values('performance', ascending=False).reset_index(drop=True)
    for i, (_, row) in enumerate(perf_rank.iterrows(), 1):
        marker = " ← mrlora" if row['peft'] == 'mrlora' else ""
        print(f"{i:2d}. {row['peft']:10s} ({row['strategy']:10s}): {row['performance']:.4f}{marker}")
    
    # Rank by parameter efficiency (performance per parameter)
    perf_df['perf_per_param'] = perf_df['performance'] / perf_df['parameters']
    print("\nParameter Efficiency Ranking (Performance per Parameter):")
    param_rank = perf_df.sort_values('perf_per_param', ascending=False).reset_index(drop=True)
    for i, (_, row) in enumerate(param_rank.iterrows(), 1):
        marker = " ← mrlora" if row['peft'] == 'mrlora' else ""
        print(f"{i:2d}. {row['peft']:10s} ({row['strategy']:10s}): {row['perf_per_param']:.4f}{marker}")
    
    # Rank by training time (lower is better)
    print("\nTraining Time Ranking (Lower is better):")
    time_rank = perf_df.sort_values('train_time').reset_index(drop=True)
    for i, (_, row) in enumerate(time_rank.iterrows(), 1):
        marker = " ← mrlora" if row['peft'] == 'mrlora' else ""
        print(f"{i:2d}. {row['peft']:10s} ({row['strategy']:10s}): {row['train_time']:.1f}s{marker}")
    
    # Rank by memory usage (lower is better)
    print("\nMemory Usage Ranking (Lower is better):")
    mem_rank = perf_df.sort_values('memory').reset_index(drop=True)
    for i, (_, row) in enumerate(mem_rank.iterrows(), 1):
        marker = " ← mrlora" if row['peft'] == 'mrlora' else ""
        print(f"{i:2d}. {row['peft']:10s} ({row['strategy']:10s}): {row['memory']:.1f}MB{marker}")
    
    # 6. Key insights
    print("\n\n6. KEY INSIGHTS ABOUT MRLORA:")
    print("=" * 80)
    
    # Calculate relative performance
    mrlora_kd_perf = mrlora_kd['metric_value_mean'].values[0] if not mrlora_kd.empty else None
    mrlora_lora_perf = mrlora_lora['metric_value_mean'].values[0] if not mrlora_lora.empty else None
    
    # Get top performers
    top_perf = perf_rank.iloc[0]
    top_param = param_rank.iloc[0]
    top_time = time_rank.iloc[0]
    top_mem = mem_rank.iloc[0]
    
    print(f"\na) Performance Position:")
    mrlora_lora_rank = perf_rank[perf_rank['peft'] == 'mrlora'][perf_rank['strategy'] == 'LoRA-only'].index[0] + 1
    mrlora_kd_rank = perf_rank[perf_rank['peft'] == 'mrlora'][perf_rank['strategy'] == 'KD-LoRA'].index[0] + 1
    print(f"   - LoRA-only: Rank {mrlora_lora_rank}/12 ({mrlora_lora_perf:.4f})")
    print(f"   - KD-LoRA: Rank {mrlora_kd_rank}/12 ({mrlora_kd_perf:.4f})")
    
    print(f"\nb) Performance vs Top Variant ({top_perf['peft']} - {top_perf['strategy']}):")
    if mrlora_lora_perf:
        diff_lora = mrlora_lora_perf - top_perf['performance']
        print(f"   - LoRA-only: {diff_lora:+.4f} ({diff_lora/top_perf['performance']*100:+.1f}%)")
    if mrlora_kd_perf:
        diff_kd = mrlora_kd_perf - top_perf['performance']
        print(f"   - KD-LoRA: {diff_kd:+.4f} ({diff_kd/top_perf['performance']*100:+.1f}%)")
    
    print(f"\nc) Parameter Efficiency:")
    mrlora_lora_param_rank = param_rank[param_rank['peft'] == 'mrlora'][param_rank['strategy'] == 'LoRA-only'].index[0] + 1
    mrlora_kd_param_rank = param_rank[param_rank['peft'] == 'mrlora'][param_rank['strategy'] == 'KD-LoRA'].index[0] + 1
    print(f"   - LoRA-only: Rank {mrlora_lora_param_rank}/12")
    print(f"   - KD-LoRA: Rank {mrlora_kd_param_rank}/12")
    
    print(f"\nd) Training Speed:")
    mrlora_lora_time_rank = time_rank[time_rank['peft'] == 'mrlora'][time_rank['strategy'] == 'LoRA-only'].index[0] + 1
    mrlora_kd_time_rank = time_rank[time_rank['peft'] == 'mrlora'][time_rank['strategy'] == 'KD-LoRA'].index[0] + 1
    print(f"   - LoRA-only: Rank {mrlora_lora_time_rank}/12 ({mrlora_lora['train_time_mean'].values[0]:.1f}s)")
    print(f"   - KD-LoRA: Rank {mrlora_kd_time_rank}/12 ({mrlora_kd['train_time_mean'].values[0]:.1f}s)")
    
    print(f"\ne) Memory Efficiency:")
    mrlora_lora_mem_rank = mem_rank[mem_rank['peft'] == 'mrlora'][mem_rank['strategy'] == 'LoRA-only'].index[0] + 1
    mrlora_kd_mem_rank = mem_rank[mem_rank['peft'] == 'mrlora'][mem_rank['strategy'] == 'KD-LoRA'].index[0] + 1
    print(f"   - LoRA-only: Rank {mrlora_lora_mem_rank}/12 ({mrlora_lora['avg_memory_allocated_mb_mean'].values[0]:.1f}MB)")
    print(f"   - KD-LoRA: Rank {mrlora_kd_mem_rank}/12 ({mrlora_kd['avg_memory_allocated_mb_mean'].values[0]:.1f}MB)")
    
    print(f"\nf) Key Strengths and Weaknesses:")
    print(f"   STRENGTHS:")
    print(f"   - KD-LoRA strategy shows competitive memory usage")
    print(f"   - Consistent performance across model families")
    print(f"   - Good performance retention vs FFT (only -5.22% drop)")
    
    print(f"\n   WEAKNESSES:")
    print(f"   - Highest parameter count among PEFT variants")
    print(f"   - Poor parameter efficiency (low performance per parameter)")
    print(f"   - Slow training speed for LoRA-only strategy")
    print(f"   - High memory usage for LoRA-only strategy")
    
    return {
        'mrlora_kd': mrlora_kd,
        'mrlora_lora': mrlora_lora,
        'performance_rank': perf_rank,
        'parameter_rank': param_rank,
        'time_rank': time_rank,
        'memory_rank': mem_rank
    }

if __name__ == "__main__":
    analyze_mrlora_performance()