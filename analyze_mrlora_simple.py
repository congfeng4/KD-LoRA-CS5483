"""
Simple analysis of mrlora performance across different aspects.
"""
import pandas as pd
import numpy as np

def analyze_mrlora():
    print("=== MRLORA COMPREHENSIVE PERFORMANCE ANALYSIS ===\n")
    
    # Read the CSV files
    print("1. LOADING DATA...")
    
    # Efficiency summary (has multi-level header)
    eff_df = pd.read_csv('efficiency_summary_by_variant.csv', header=[0,1])
    
    # Flatten columns
    eff_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in eff_df.columns]
    
    # Clean column names
    eff_df.columns = [col.strip('_') for col in eff_df.columns]
    
    # Parameter efficiency
    param_df = pd.read_csv('parameter_efficiency_analysis.csv')
    
    # Statistical tests
    stats_df = pd.read_csv('statistical_test_results.csv')
    
    # Multi-model analysis
    multi_df = pd.read_csv('multi_model_analysis_results.csv')
    
    print("2. OVERALL PERFORMANCE METRICS\n")
    print("=" * 80)
    
    # Extract mrlora from efficiency summary
    mrlora_mask = eff_df.iloc[:, 0].str.contains('mrlora', case=False, na=False)
    mrlora_rows = eff_df[mrlora_mask]
    
    if len(mrlora_rows) > 0:
        for idx, row in mrlora_rows.iterrows():
            strategy = row.iloc[1] if len(row) > 1 else "Unknown"
            print(f"\n{strategy}:")
            print(f"  Performance: {row.get('metric_value_mean', 'N/A'):.4f}")
            print(f"  Std Dev: {row.get('metric_value_std', 'N/A'):.4f}")
            print(f"  Count: {row.get('metric_value_count', 'N/A'):.0f}")
            print(f"  Train Time: {row.get('train_time_mean', 'N/A'):.1f}s")
            print(f"  Parameters: {row.get('trainable_params_millions_mean', 'N/A'):.3f}M")
            print(f"  Memory: {row.get('avg_memory_allocated_mb_mean', 'N/A'):.1f}MB")
            print(f"  Throughput: {row.get('throughput_samples_per_sec_mean', 'N/A'):.1f} samples/s")
            print(f"  FLOPs: {float(row.get('total_flos_mean', 0)):.2e}")
    
    print("\n3. STATISTICAL SIGNIFICANCE\n")
    print("=" * 80)
    
    # mrlora statistical tests
    mrlora_stats = stats_df[stats_df['comparison'].str.contains('mrlora', case=False, na=False)]
    
    for _, row in mrlora_stats.iterrows():
        print(f"\n{row['comparison']}:")
        if 'LoRA-only vs KD-LoRA' in row['comparison']:
            print(f"  LoRA-only: {row['mean_lora']:.4f} (n={row['n_lora']:.0f})")
            print(f"  KD-LoRA: {row['mean_kd']:.4f} (n={row['n_kd']:.0f})")
            print(f"  Difference: {row['difference']:+.4f}")
            print(f"  p-value: {row['p_value']:.4f} {'(significant)' if row['p_value'] < 0.05 else '(not significant)'}")
        elif 'vs FFT' in row['comparison']:
            print(f"  FFT: {row['mean_fft']:.4f} (n={row['n_fft']:.0f})")
            print(f"  mrlora: {row['mean_peft']:.4f} (n={row['n_peft']:.0f})")
            print(f"  Difference: {row['difference']:+.4f}")
            print(f"  p-value: {row['p_value']:.4f} {'(significant)' if row['p_value'] < 0.05 else '(not significant)'}")
    
    print("\n4. PARAMETER EFFICIENCY\n")
    print("=" * 80)
    
    mrlora_param = param_df[param_df['peft'] == 'mrlora']
    for _, row in mrlora_param.iterrows():
        print(f"\n{row['strategy']}:")
        print(f"  Parameters: {row['trainable_params_millions']:.3f}M")
        print(f"  Performance: {row['metric_value']:.4f}")
        print(f"  Performance/Param: {row['performance_per_param']:.4f}")
        print(f"  Rank among 12 variants: ", end="")
        # Calculate rank
        all_param = param_df.copy()
        all_param['perf_per_param'] = all_param['metric_value'] / all_param['trainable_params_millions']
        sorted_param = all_param.sort_values('perf_per_param', ascending=False).reset_index()
        rank = sorted_param[sorted_param['peft'] == 'mrlora'][sorted_param['strategy'] == row['strategy']].index[0] + 1
        print(f"{rank}/12")
    
    print("\n5. MODEL-SPECIFIC PERFORMANCE\n")
    print("=" * 80)
    
    mrlora_multi = multi_df[multi_df['peft_variant'] == 'mrlora']
    for _, row in mrlora_multi.iterrows():
        print(f"\n{row['model_family']} - {row['strategy']}:")
        print(f"  Average Score: {row['average_score']:.4f}")
        print(f"  Drop vs FFT: {row['drop_vs_fft_percent']:.2f}%")
        print(f"  Max Score: {row['max_score']:.4f}")
        print(f"  Rank: {int(row['rank'])}/6")
    
    print("\n6. COMPARATIVE ANALYSIS\n")
    print("=" * 80)
    
    # Load efficiency metrics for all variants
    all_metrics = []
    for idx, row in eff_df.iterrows():
        if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
            all_metrics.append({
                'peft': row.iloc[0],
                'strategy': row.iloc[1],
                'performance': row.get('metric_value_mean', np.nan),
                'train_time': row.get('train_time_mean', np.nan),
                'parameters': row.get('trainable_params_millions_mean', np.nan),
                'memory': row.get('avg_memory_allocated_mb_mean', np.nan)
            })
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Performance ranking
    print("\na) Performance Ranking (12 variants total):")
    perf_sorted = metrics_df.sort_values('performance', ascending=False).reset_index(drop=True)
    for i, (_, row) in enumerate(perf_sorted.iterrows(), 1):
        marker = " ← mrlora" if row['peft'] == 'mrlora' else ""
        print(f"  {i:2d}. {row['peft']:8s} ({row['strategy']:8s}): {row['performance']:.4f}{marker}")
    
    # Parameter efficiency ranking
    metrics_df['perf_per_param'] = metrics_df['performance'] / metrics_df['parameters']
    print("\nb) Parameter Efficiency Ranking:")
    param_sorted = metrics_df.sort_values('perf_per_param', ascending=False).reset_index(drop=True)
    for i, (_, row) in enumerate(param_sorted.iterrows(), 1):
        marker = " ← mrlora" if row['peft'] == 'mrlora' else ""
        print(f"  {i:2d}. {row['peft']:8s} ({row['strategy']:8s}): {row['perf_per_param']:.4f}{marker}")
    
    # Training time ranking (lower is better)
    print("\nc) Training Time Ranking (lower is better):")
    time_sorted = metrics_df.sort_values('train_time').reset_index(drop=True)
    for i, (_, row) in enumerate(time_sorted.iterrows(), 1):
        marker = " ← mrlora" if row['peft'] == 'mrlora' else ""
        print(f"  {i:2d}. {row['peft']:8s} ({row['strategy']:8s}): {row['train_time']:.1f}s{marker}")
    
    # Memory ranking (lower is better)
    print("\nd) Memory Usage Ranking (lower is better):")
    mem_sorted = metrics_df.sort_values('memory').reset_index(drop=True)
    for i, (_, row) in enumerate(mem_sorted.iterrows(), 1):
        marker = " ← mrlora" if row['peft'] == 'mrlora' else ""
        print(f"  {i:2d}. {row['peft']:8s} ({row['strategy']:8s}): {row['memory']:.1f}MB{marker}")
    
    print("\n7. KEY INSIGHTS\n")
    print("=" * 80)
    
    # Extract mrlora metrics
    mrlora_lora = metrics_df[(metrics_df['peft'] == 'mrlora') & (metrics_df['strategy'] == 'LoRA-only')]
    mrlora_kd = metrics_df[(metrics_df['peft'] == 'mrlora') & (metrics_df['strategy'] == 'KD-LoRA')]
    
    if not mrlora_lora.empty:
        ml = mrlora_lora.iloc[0]
        print(f"\na) LoRA-only Strategy:")
        print(f"   • Performance: {ml['performance']:.4f} (Rank {perf_sorted[perf_sorted['peft'] == 'mrlora'][perf_sorted['strategy'] == 'LoRA-only'].index[0] + 1}/12)")
        print(f"   • Parameters: {ml['parameters']:.3f}M (Highest among all variants)")
        print(f"   • Parameter Efficiency: {ml['perf_per_param']:.4f} (Rank {param_sorted[param_sorted['peft'] == 'mrlora'][param_sorted['strategy'] == 'LoRA-only'].index[0] + 1}/12)")
        print(f"   • Training Time: {ml['train_time']:.1f}s (Rank {time_sorted[time_sorted['peft'] == 'mrlora'][time_sorted['strategy'] == 'LoRA-only'].index[0] + 1}/12)")
        print(f"   • Memory: {ml['memory']:.1f}MB (Rank {mem_sorted[mem_sorted['peft'] == 'mrlora'][mem_sorted['strategy'] == 'LoRA-only'].index[0] + 1}/12)")
    
    if not mrlora_kd.empty:
        mk = mrlora_kd.iloc[0]
        print(f"\nb) KD-LoRA Strategy:")
        print(f"   • Performance: {mk['performance']:.4f} (Rank {perf_sorted[perf_sorted['peft'] == 'mrlora'][perf_sorted['strategy'] == 'KD-LoRA'].index[0] + 1}/12)")
        print(f"   • Parameters: {mk['parameters']:.3f}M (Highest among KD-LoRA variants)")
        print(f"   • Parameter Efficiency: {mk['perf_per_param']:.4f} (Rank {param_sorted[param_sorted['peft'] == 'mrlora'][param_sorted['strategy'] == 'KD-LoRA'].index[0] + 1}/12)")
        print(f"   • Training Time: {mk['train_time']:.1f}s (Rank {time_sorted[time_sorted['peft'] == 'mrlora'][time_sorted['strategy'] == 'KD-LoRA'].index[0] + 1}/12)")
        print(f"   • Memory: {mk['memory']:.1f}MB (Rank {mem_sorted[mem_sorted['peft'] == 'mrlora'][mem_sorted['strategy'] == 'KD-LoRA'].index[0] + 1}/12)")
    
    print(f"\nc) Performance vs FFT:")
    fft_vs_mrlora = mrlora_stats[mrlora_stats['comparison'].str.contains('vs FFT')]
    if not fft_vs_mrlora.empty:
        row = fft_vs_mrlora.iloc[0]
        print(f"   • FFT Performance: {row['mean_fft']:.4f}")
        print(f"   • mrlora Performance: {row['mean_peft']:.4f}")
        print(f"   • Performance Drop: {abs(row['difference']):.4f} ({abs(row['difference']/row['mean_fft']*100):.1f}%)")
        print(f"   • Statistical Significance: {'Significant (p < 0.05)' if row['p_value'] < 0.05 else 'Not significant (p > 0.05)'}")
    
    print(f"\nd) Key Strengths:")
    print(f"   1. Competitive performance with KD-LoRA strategy")
    print(f"   2. Good memory efficiency with KD-LoRA")
    print(f"   3. Consistent across model families (BERT, RoBERTa, DeBERTa)")
    print(f"   4. Smallest performance drop vs FFT among some model families")
    
    print(f"\ne) Key Weaknesses:")
    print(f"   1. Highest parameter count among all PEFT variants")
    print(f"   2. Poor parameter efficiency (low performance per parameter)")
    print(f"   3. Slow training for LoRA-only strategy")
    print(f"   4. High memory usage for LoRA-only strategy")
    
    print(f"\nf) Best Use Cases for mrlora:")
    print(f"   1. When parameter count is not a constraint")
    print(f"   2. When using KD-LoRA strategy for memory efficiency")
    print(f"   3. With DeBERTa models (shows best performance)")
    print(f"   4. When consistency across model families is important")
    
    print(f"\ng) Recommended Alternatives:")
    print(f"   1. For parameter efficiency: Use lora or rslora")
    print(f"   2. For maximum performance: Use olora or rslora")
    print(f"   3. For fast training: Use rslora")
    print(f"   4. For low memory: Use any variant with KD-LoRA strategy")

if __name__ == "__main__":
    analyze_mrlora()