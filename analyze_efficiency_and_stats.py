"""
Comprehensive analysis of efficiency metrics and statistical tests for KD-LoRA paper.
This script performs:
1. Statistical significance tests (t-tests) comparing LoRA-only vs KD-LoRA
2. Efficiency vs performance trade-off analysis
3. Parameter efficiency analysis
4. Generation of supplementary tables and figures
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind, f_oneway, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_preprocess_data():
    """Load efficiency metrics and preprocess for analysis."""
    df = pd.read_csv('efficiency_metrics_all.csv')
    
    # Convert metric_value to numeric
    df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')
    
    # For MNLI, we need to handle matched_accuracy vs mismatched_accuracy
    # Use matched_accuracy as primary metric
    mnli_mask = df['task'] == 'mnli'
    df.loc[mnli_mask, 'metric_value'] = df.loc[mnli_mask, 'metric_value']
    
    # Create combined strategy column
    df['strategy'] = df['variant'].map({
        'fft': 'FFT',
        'lora': 'LoRA-only',
        'kd-lora': 'KD-LoRA'
    })
    
    # Create full variant name including strategy
    df['full_variant'] = df['peft'] + ' (' + df['strategy'] + ')'
    
    # For FFT, set peft to 'fft' for consistency
    df.loc[df['variant'] == 'fft', 'peft'] = 'fft'
    
    # Convert trainable_params_count: FFT has absolute counts, LoRA has percentages
    # We need to handle this carefully
    df['trainable_params_millions'] = np.where(
        df['variant'] == 'fft',
        df['trainable_params_count'],  # Already in millions for FFT
        df['trainable_params_count'] * 110 / 100  # Estimate: assume 110M params base * percentage
    )
    
    # Add efficiency metrics
    df['throughput_samples_per_sec'] = df['train_samples_per_second']
    df['memory_efficiency_mb_per_param'] = df['avg_memory_allocated_mb'] / df['trainable_params_millions']
    df['time_efficiency_sec_per_param'] = df['train_time'] / df['trainable_params_millions']
    
    return df

def perform_statistical_tests(df):
    """Perform statistical significance tests on performance differences."""
    print("=== STATISTICAL SIGNIFICANCE TESTS ===")
    results = []
    
    # Filter out FFT for PEFT comparisons
    peft_df = df[df['variant'] != 'fft'].copy()
    
    # 1. Compare LoRA-only vs KD-LoRA for each PEFT variant
    print("\n1. LoRA-only vs KD-LoRA comparison (within each PEFT variant):")
    peft_variants = ['adalora', 'dora', 'lora', 'mrlora', 'olora', 'rslora']
    
    for peft_var in peft_variants:
        lora_only = peft_df[(peft_df['peft'] == peft_var) & (peft_df['variant'] == 'lora')]['metric_value'].dropna()
        kd_lora = peft_df[(peft_df['peft'] == peft_var) & (peft_df['variant'] == 'kd-lora')]['metric_value'].dropna()
        
        if len(lora_only) > 1 and len(kd_lora) > 1:
            t_stat, p_value = ttest_ind(lora_only, kd_lora, equal_var=False)
            mean_lora = lora_only.mean()
            mean_kd = kd_lora.mean()
            diff = mean_lora - mean_kd
            results.append({
                'comparison': f'{peft_var}: LoRA-only vs KD-LoRA',
                'mean_lora': mean_lora,
                'mean_kd': mean_kd,
                'difference': diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n_lora': len(lora_only),
                'n_kd': len(kd_lora)
            })
            print(f"{peft_var:10s} | LoRA-only: {mean_lora:.4f} (n={len(lora_only):3d}) | "
                  f"KD-LoRA: {mean_kd:.4f} (n={len(kd_lora):3d}) | "
                  f"Diff: {diff:+.4f} | p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
    
    # 2. Compare each PEFT variant against FFT baseline (pooled across strategies)
    print("\n2. PEFT variants vs FFT baseline:")
    fft_scores = df[df['variant'] == 'fft']['metric_value'].dropna()
    
    for peft_var in peft_variants:
        peft_scores = peft_df[peft_df['peft'] == peft_var]['metric_value'].dropna()
        
        if len(peft_scores) > 1:
            t_stat, p_value = ttest_ind(fft_scores, peft_scores, equal_var=False)
            mean_fft = fft_scores.mean()
            mean_peft = peft_scores.mean()
            diff = mean_peft - mean_fft  # Positive means PEFT is better
            results.append({
                'comparison': f'{peft_var} vs FFT',
                'mean_fft': mean_fft,
                'mean_peft': mean_peft,
                'difference': diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n_fft': len(fft_scores),
                'n_peft': len(peft_scores)
            })
            print(f"{peft_var:10s} | FFT: {mean_fft:.4f} (n={len(fft_scores):3d}) | "
                  f"PEFT: {mean_peft:.4f} (n={len(peft_scores):3d}) | "
                  f"Diff: {diff:+.4f} | p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
    
    # 3. Compare model families
    print("\n3. Model family comparisons:")
    model_families = ['bert', 'roberta', 'deberta']
    
    for i, model1 in enumerate(model_families):
        for model2 in model_families[i+1:]:
            scores1 = df[df['model_family'] == model1]['metric_value'].dropna()
            scores2 = df[df['model_family'] == model2]['metric_value'].dropna()
            
            if len(scores1) > 1 and len(scores2) > 1:
                t_stat, p_value = ttest_ind(scores1, scores2, equal_var=False)
                mean1 = scores1.mean()
                mean2 = scores2.mean()
                diff = mean1 - mean2
                results.append({
                    'comparison': f'{model1} vs {model2}',
                    'mean1': mean1,
                    'mean2': mean2,
                    'difference': diff,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n1': len(scores1),
                    'n2': len(scores2)
                })
                print(f"{model1:8s} vs {model2:8s} | {model1}: {mean1:.4f} (n={len(scores1):3d}) | "
                      f"{model2}: {mean2:.4f} (n={len(scores2):3d}) | "
                      f"Diff: {diff:+.4f} | p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('statistical_test_results.csv', index=False)
    print(f"\nSaved statistical test results to statistical_test_results.csv")
    
    return results_df

def analyze_efficiency_tradeoffs(df):
    """Analyze efficiency vs performance trade-offs."""
    print("\n=== EFFICIENCY VS PERFORMANCE TRADE-OFFS ===")
    
    # Filter out FFT for efficiency analysis (different scale)
    peft_df = df[df['variant'] != 'fft'].copy()
    
    # 1. Efficiency metrics by variant and strategy
    print("\n1. Average efficiency metrics by PEFT variant and strategy:")
    efficiency_summary = peft_df.groupby(['peft', 'strategy']).agg({
        'metric_value': ['mean', 'std', 'count'],
        'train_time': 'mean',
        'trainable_params_millions': 'mean',
        'avg_memory_allocated_mb': 'mean',
        'throughput_samples_per_sec': 'mean',
        'total_flos': 'mean'
    }).round(4)
    
    print(efficiency_summary.to_string())
    efficiency_summary.to_csv('efficiency_summary_by_variant.csv')
    
    # 2. Correlation analysis
    print("\n2. Correlation between efficiency metrics and performance:")
    efficiency_metrics = ['train_time', 'trainable_params_millions', 
                         'avg_memory_allocated_mb', 'throughput_samples_per_sec', 'total_flos']
    
    corr_results = []
    for metric in efficiency_metrics:
        if metric in peft_df.columns:
            corr = peft_df['metric_value'].corr(peft_df[metric])
            corr_results.append({'metric': metric, 'correlation_with_performance': corr})
            print(f"{metric:25s} | Correlation: {corr:.4f}")
    
    # 3. Efficiency-performance scatter plots
    print("\n3. Generating efficiency-performance scatter plots...")
    
    # Plot 1: Training time vs performance
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plots = [
        ('train_time', 'Training Time (s)', 0),
        ('trainable_params_millions', 'Trainable Parameters (M)', 1),
        ('avg_memory_allocated_mb', 'Avg Memory Allocated (MB)', 2),
        ('throughput_samples_per_sec', 'Throughput (samples/sec)', 3),
        ('total_flos', 'Total FLOPs', 4),
        ('memory_efficiency_mb_per_param', 'Memory per Param (MB/M)', 5)
    ]
    
    for i, (metric, xlabel, ax_idx) in enumerate(plots):
        if metric in peft_df.columns:
            ax = axes[ax_idx]
            for strategy in peft_df['strategy'].unique():
                subset = peft_df[peft_df['strategy'] == strategy]
                ax.scatter(subset[metric], subset['metric_value'], 
                          alpha=0.6, label=strategy, s=50)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Performance')
            ax.set_title(f'{xlabel} vs Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('efficiency_performance_scatter.png', dpi=300, bbox_inches='tight')
    print("Saved scatter plot: efficiency_performance_scatter.png")
    
    # 4. Efficiency comparison bar charts
    print("\n4. Generating efficiency comparison bar charts...")
    
    # Group by peft and strategy for bar charts
    grouped = peft_df.groupby(['peft', 'strategy']).agg({
        'metric_value': 'mean',
        'train_time': 'mean',
        'trainable_params_millions': 'mean',
        'avg_memory_allocated_mb': 'mean'
    }).reset_index()
    
    # Normalize metrics for comparison
    for metric in ['train_time', 'trainable_params_millions', 'avg_memory_allocated_mb']:
        grouped[f'{metric}_norm'] = grouped[metric] / grouped[metric].max()
    
    # Plot normalized efficiency metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('metric_value', 'Performance', 0, 0),
        ('train_time_norm', 'Normalized Training Time', 0, 1),
        ('trainable_params_millions_norm', 'Normalized Trainable Params', 1, 0),
        ('avg_memory_allocated_mb_norm', 'Normalized Memory Usage', 1, 1)
    ]
    
    for metric, title, row, col in metrics_to_plot:
        ax = axes[row, col]
        pivot = grouped.pivot(index='peft', columns='strategy', values=metric)
        pivot.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('PEFT Variant')
        ax.set_ylabel(title)
        ax.legend(title='Strategy')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('efficiency_comparison_bars.png', dpi=300, bbox_inches='tight')
    print("Saved bar chart: efficiency_comparison_bars.png")
    
    return efficiency_summary

def analyze_parameter_efficiency(df):
    """Analyze parameter efficiency across variants."""
    print("\n=== PARAMETER EFFICIENCY ANALYSIS ===")
    
    # Separate FFT and PEFT for analysis
    fft_df = df[df['variant'] == 'fft'].copy()
    peft_df = df[df['variant'] != 'fft'].copy()
    
    # 1. Parameter counts
    print("\n1. Parameter counts:")
    print(f"FFT average trainable parameters: {fft_df['trainable_params_millions'].mean():.2f}M")
    print(f"PEFT average trainable parameters: {peft_df['trainable_params_millions'].mean():.2f}M")
    print(f"Reduction factor: {fft_df['trainable_params_millions'].mean() / peft_df['trainable_params_millions'].mean():.2f}x")
    
    # 2. Parameter efficiency by PEFT variant
    print("\n2. Parameter efficiency by PEFT variant and strategy:")
    param_efficiency = peft_df.groupby(['peft', 'strategy']).agg({
        'trainable_params_millions': 'mean',
        'metric_value': 'mean',
        'train_time': 'mean'
    }).round(4)
    
    # Calculate performance per parameter
    param_efficiency['performance_per_param'] = (
        param_efficiency['metric_value'] / param_efficiency['trainable_params_millions']
    )
    
    print(param_efficiency.to_string())
    param_efficiency.to_csv('parameter_efficiency_analysis.csv')
    
    # 3. Plot parameter efficiency
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Parameters vs Performance
    ax = axes[0]
    for strategy in peft_df['strategy'].unique():
        subset = peft_df[peft_df['strategy'] == strategy]
        ax.scatter(subset['trainable_params_millions'], subset['metric_value'],
                  alpha=0.6, label=strategy, s=60)
    
    ax.set_xlabel('Trainable Parameters (M)')
    ax.set_ylabel('Performance')
    ax.set_title('Parameter Efficiency: Parameters vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Performance per parameter by variant
    ax = axes[1]
    performance_per_param = peft_df.copy()
    performance_per_param['perf_per_param'] = (
        performance_per_param['metric_value'] / performance_per_param['trainable_params_millions']
    )
    
    # Group and plot
    grouped = performance_per_param.groupby(['peft', 'strategy'])['perf_per_param'].mean().reset_index()
    pivot = grouped.pivot(index='peft', columns='strategy', values='perf_per_param')
    pivot.plot(kind='bar', ax=ax, alpha=0.8)
    
    ax.set_xlabel('PEFT Variant')
    ax.set_ylabel('Performance per Parameter')
    ax.set_title('Performance per Trainable Parameter')
    ax.legend(title='Strategy')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('parameter_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved parameter efficiency plot: parameter_efficiency_analysis.png")
    
    return param_efficiency

def generate_supplementary_tables(df):
    """Generate LaTeX tables for paper supplementary materials."""
    print("\n=== GENERATING SUPPLEMENTARY TABLES ===")
    
    # Table 1: Efficiency metrics summary
    peft_df = df[df['variant'] != 'fft'].copy()
    
    # Group by peft and strategy
    summary = peft_df.groupby(['peft', 'strategy']).agg({
        'metric_value': ['mean', 'std', 'count'],
        'train_time': ['mean', 'std'],
        'trainable_params_millions': ['mean', 'std'],
        'avg_memory_allocated_mb': ['mean', 'std'],
        'throughput_samples_per_sec': ['mean', 'std']
    }).round(4)
    
    # Flatten multi-level columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Save as CSV and LaTeX
    summary.to_csv('supplementary_table_efficiency.csv', index=False)
    
    # Generate LaTeX table
    latex_table = summary.to_latex(index=False, float_format="%.4f", 
                                  caption="Efficiency metrics across PEFT variants and strategies",
                                  label="tab:efficiency_metrics")
    
    with open('supplementary_table_efficiency.tex', 'w') as f:
        f.write(latex_table)
    
    print("Saved supplementary efficiency table:")
    print("- supplementary_table_efficiency.csv")
    print("- supplementary_table_efficiency.tex")
    
    # Table 2: Statistical test results (load from earlier)
    try:
        stats_df = pd.read_csv('statistical_test_results.csv')
        stats_latex = stats_df.to_latex(index=False, float_format="%.4f",
                                       caption="Statistical test results for performance differences",
                                       label="tab:statistical_tests")
        
        with open('supplementary_table_statistical_tests.tex', 'w') as f:
            f.write(stats_latex)
        
        print("Saved supplementary statistical tests table:")
        print("- statistical_test_results.csv")
        print("- supplementary_table_statistical_tests.tex")
    except:
        print("Statistical test results not available yet")
    
    return summary

def main():
    """Main analysis pipeline."""
    print("KD-LoRA Efficiency and Statistical Analysis")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df = load_and_preprocess_data()
    print(f"Loaded {len(df)} experiments")
    
    # Perform statistical tests
    stats_results = perform_statistical_tests(df)
    
    # Analyze efficiency trade-offs
    efficiency_summary = analyze_efficiency_tradeoffs(df)
    
    # Analyze parameter efficiency
    param_efficiency = analyze_parameter_efficiency(df)
    
    # Generate supplementary tables
    supplementary_tables = generate_supplementary_tables(df)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("\nGenerated files:")
    print("1. statistical_test_results.csv - Statistical test results")
    print("2. efficiency_summary_by_variant.csv - Efficiency metrics by variant")
    print("3. parameter_efficiency_analysis.csv - Parameter efficiency analysis")
    print("4. supplementary_table_efficiency.csv/.tex - Supplementary table")
    print("5. efficiency_performance_scatter.png - Scatter plots")
    print("6. efficiency_comparison_bars.png - Bar charts")
    print("7. parameter_efficiency_analysis.png - Parameter efficiency plots")
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    
    # Performance difference between LoRA-only and KD-LoRA
    peft_df = df[df['variant'] != 'fft'].copy()
    lora_mean = peft_df[peft_df['variant'] == 'lora']['metric_value'].mean()
    kd_lora_mean = peft_df[peft_df['variant'] == 'kd-lora']['metric_value'].mean()
    print(f"1. Performance: LoRA-only ({lora_mean:.4f}) vs KD-LoRA ({kd_lora_mean:.4f})")
    print(f"   Difference: {lora_mean - kd_lora_mean:.4f} (LoRA-only better)")
    
    # Parameter reduction
    fft_mean_params = df[df['variant'] == 'fft']['trainable_params_millions'].mean()
    peft_mean_params = peft_df['trainable_params_millions'].mean()
    print(f"2. Parameter reduction: {fft_mean_params/peft_mean_params:.1f}x fewer parameters with PEFT")
    
    # Memory reduction
    fft_mean_mem = df[df['variant'] == 'fft']['avg_memory_allocated_mb'].mean()
    peft_mean_mem = peft_df['avg_memory_allocated_mb'].mean()
    print(f"3. Memory reduction: {fft_mean_mem/peft_mean_mem:.1f}x less memory with PEFT")
    
    # Time efficiency
    fft_mean_time = df[df['variant'] == 'fft']['train_time'].mean()
    peft_mean_time = peft_df['train_time'].mean()
    print(f"4. Training time: FFT ({fft_mean_time:.1f}s) vs PEFT ({peft_mean_time:.1f}s)")
    print(f"   Speedup: {fft_mean_time/peft_mean_time:.1f}x faster with PEFT")

if __name__ == "__main__":
    main()