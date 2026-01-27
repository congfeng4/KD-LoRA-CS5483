#!/usr/bin/env python3
"""
Analyze Table II results for all model families (BERT, RoBERTa, DeBERTa).
Compute performance degradation relative to FFT baseline, rank variants, and compare across models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Model families and their display names
MODEL_FAMILIES = ['bert', 'roberta', 'deberta']
MODEL_DISPLAY_NAMES = {'bert': 'BERT', 'roberta': 'RoBERTa', 'deberta': 'DeBERTa-v3'}

# PEFT variants
PEFT_VARIANTS = ['adalora', 'dora', 'lora', 'mrlora', 'olora', 'rslora']

def load_model_tables():
    """Load all Table II results for each model family."""
    model_data = {}
    
    for model_family in MODEL_FAMILIES:
        model_dir = Path(f'table_ii_{model_family}')
        if not model_dir.exists():
            print(f"Warning: Directory {model_dir} not found for {model_family}")
            continue
        
        # Load Table IIa (LoRA-only) and Table IIb (KD-LoRA)
        iia_path = model_dir / 'table_iia_results.csv'
        iib_path = model_dir / 'table_iib_results.csv'
        
        if iia_path.exists():
            table_iia = pd.read_csv(iia_path, index_col=0)
        else:
            print(f"Warning: Table IIa not found for {model_family}")
            table_iia = pd.DataFrame()
        
        if iib_path.exists():
            table_iib = pd.read_csv(iib_path, index_col=0)
        else:
            print(f"Warning: Table IIb not found for {model_family}")
            table_iib = pd.DataFrame()
        
        model_data[model_family] = {
            'iia': table_iia,
            'iib': table_iib
        }
    
    return model_data

def analyze_model_performance(model_data):
    """Analyze performance for each model family."""
    print("=" * 80)
    print("MULTI-MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    all_results = []
    
    for model_family in MODEL_FAMILIES:
        if model_family not in model_data:
            continue
        
        print(f"\n{'='*60}")
        print(f"{MODEL_DISPLAY_NAMES[model_family].upper()}")
        print(f"{'='*60}")
        
        tables = model_data[model_family]
        table_iia = tables['iia']
        table_iib = tables['iib']
        
        if table_iia.empty or table_iib.empty:
            print(f"Incomplete data for {model_family}. Skipping detailed analysis.")
            continue
        
        # Extract average rows
        avg_iia = table_iia.loc['Average'] if 'Average' in table_iia.index else pd.Series()
        avg_iib = table_iib.loc['Average'] if 'Average' in table_iib.index else pd.Series()
        
        if avg_iia.empty or avg_iib.empty:
            print(f"No average data for {model_family}. Skipping.")
            continue
        
        # Compute relative performance drop compared to FFT
        def compute_relative_drop(avg_series):
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
        
        # Rank variants by performance (excluding FFT)
        def rank_variants(avg_series):
            variants = [v for v in avg_series.index if v != 'FFT' and not pd.isna(avg_series[v])]
            sorted_variants = sorted(variants, key=lambda v: avg_series[v], reverse=True)
            return sorted_variants
        
        rank_iia = rank_variants(avg_iia)
        rank_iib = rank_variants(avg_iib)
        
        print(f"\nLoRA-only Strategy (Table IIa):")
        print(f"  FFT Baseline: {avg_iia['FFT']:.4f}")
        print(f"  Best variant: {rank_iia[0]} ({avg_iia[rank_iia[0]]:.4f})")
        print(f"  Worst variant: {rank_iia[-1]} ({avg_iia[rank_iia[-1]]:.4f})")
        
        print(f"\nKD-LoRA Strategy (Table IIb):")
        print(f"  FFT Baseline: {avg_iib['FFT']:.4f}")
        print(f"  Best variant: {rank_iib[0]} ({avg_iib[rank_iib[0]]:.4f})")
        print(f"  Worst variant: {rank_iib[-1]} ({avg_iib[rank_iib[-1]]:.4f})")
        
        # Collect results for cross-model comparison
        for variant in PEFT_VARIANTS:
            if variant in avg_iia:
                all_results.append({
                    'Model Family': MODEL_DISPLAY_NAMES[model_family],
                    'Strategy': 'LoRA-only',
                    'Variant': variant,
                    'Average Score': avg_iia[variant],
                    'Relative Drop (%)': rel_drop_iia.get(variant, np.nan),
                    'FFT Baseline': avg_iia['FFT'],
                    'Rank': rank_iia.index(variant) + 1 if variant in rank_iia else np.nan
                })
            
            if variant in avg_iib:
                all_results.append({
                    'Model Family': MODEL_DISPLAY_NAMES[model_family],
                    'Strategy': 'KD-LoRA',
                    'Variant': variant,
                    'Average Score': avg_iib[variant],
                    'Relative Drop (%)': rel_drop_iib.get(variant, np.nan),
                    'FFT Baseline': avg_iib['FFT'],
                    'Rank': rank_iib.index(variant) + 1 if variant in rank_iib else np.nan
                })
    
    # Create DataFrame for all results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = 'multi_model_analysis_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nDetailed analysis results saved to {results_path}")
        return results_df
    else:
        print("\nNo results to analyze.")
        return pd.DataFrame()

def create_cross_model_comparison(results_df):
    """Create cross-model comparison visualizations and summaries."""
    if results_df.empty:
        print("No data for cross-model comparison.")
        return
    
    print("\n" + "=" * 80)
    print("CROSS-MODEL COMPARISON")
    print("=" * 80)
    
    # Pivot tables for analysis
    pivot_score = results_df.pivot_table(
        index=['Model Family', 'Variant'],
        columns='Strategy',
        values='Average Score'
    )
    
    pivot_drop = results_df.pivot_table(
        index=['Model Family', 'Variant'],
        columns='Strategy',
        values='Relative Drop (%)'
    )
    
    pivot_rank = results_df.pivot_table(
        index=['Model Family', 'Variant'],
        columns='Strategy',
        values='Rank'
    )
    
    print("\nAverage Scores by Model Family and Strategy:")
    print(pivot_score.round(4))
    
    print("\nRelative Performance Drop vs FFT (%):")
    print(pivot_drop.round(2))
    
    print("\nRanking (1=best, 6=worst):")
    print(pivot_rank)
    
    # Save pivot tables
    pivot_score.to_csv('cross_model_scores.csv')
    pivot_drop.to_csv('cross_model_drops.csv')
    pivot_rank.to_csv('cross_model_ranks.csv')
    
    # Find best performing variants per model and strategy
    print("\nBEST PERFORMING VARIANTS:")
    print("-" * 40)
    
    for strategy in ['LoRA-only', 'KD-LoRA']:
        print(f"\n{strategy}:")
        strategy_df = results_df[results_df['Strategy'] == strategy]
        for model in MODEL_DISPLAY_NAMES.values():
            model_df = strategy_df[strategy_df['Model Family'] == model]
            if not model_df.empty:
                best_row = model_df.loc[model_df['Average Score'].idxmax()]
                print(f"  {model:15s}: {best_row['Variant']:10s} ({best_row['Average Score']:.4f}), "
                      f"Drop: {best_row['Relative Drop (%)']:.2f}%")
    
    # Compare performance across models for each variant
    print("\n" + "=" * 80)
    print("VARIANT PERFORMANCE ACROSS MODELS")
    print("=" * 80)
    
    for variant in PEFT_VARIANTS:
        variant_df = results_df[results_df['Variant'] == variant]
        if variant_df.empty:
            continue
        
        print(f"\n{variant.upper()}:")
        for strategy in ['LoRA-only', 'KD-LoRA']:
            strat_df = variant_df[variant_df['Strategy'] == strategy]
            if not strat_df.empty:
                scores = strat_df.set_index('Model Family')['Average Score']
                drops = strat_df.set_index('Model Family')['Relative Drop (%)']
                print(f"  {strategy}:")
                for model in MODEL_DISPLAY_NAMES.values():
                    if model in scores.index:
                        print(f"    {model:15s}: {scores[model]:.4f} (Drop: {drops[model]:.2f}%)")

def plot_cross_model_comparison(results_df):
    """Generate visualizations for cross-model comparison."""
    if results_df.empty:
        print("No data for visualizations.")
        return
    
    try:
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Heatmap: Average scores by model family and variant
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # LoRA-only heatmap
        lora_df = results_df[results_df['Strategy'] == 'LoRA-only']
        if not lora_df.empty:
            pivot_lora = lora_df.pivot_table(
                index='Model Family',
                columns='Variant',
                values='Average Score'
            )
            im1 = axes[0,0].imshow(pivot_lora.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.9)
            axes[0,0].set_xticks(range(len(pivot_lora.columns)))
            axes[0,0].set_xticklabels(pivot_lora.columns, rotation=45)
            axes[0,0].set_yticks(range(len(pivot_lora.index)))
            axes[0,0].set_yticklabels(pivot_lora.index)
            axes[0,0].set_title('LoRA-only: Average Scores')
            plt.colorbar(im1, ax=axes[0,0])
            
            # Add text annotations
            for i in range(len(pivot_lora.index)):
                for j in range(len(pivot_lora.columns)):
                    axes[0,0].text(j, i, f'{pivot_lora.iloc[i, j]:.3f}',
                                 ha='center', va='center', color='black', fontsize=8)
        
        # KD-LoRA heatmap
        kd_lora_df = results_df[results_df['Strategy'] == 'KD-LoRA']
        if not kd_lora_df.empty:
            pivot_kd_lora = kd_lora_df.pivot_table(
                index='Model Family',
                columns='Variant',
                values='Average Score'
            )
            im2 = axes[0,1].imshow(pivot_kd_lora.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.9)
            axes[0,1].set_xticks(range(len(pivot_kd_lora.columns)))
            axes[0,1].set_xticklabels(pivot_kd_lora.columns, rotation=45)
            axes[0,1].set_yticks(range(len(pivot_kd_lora.index)))
            axes[0,1].set_yticklabels(pivot_kd_lora.index)
            axes[0,1].set_title('KD-LoRA: Average Scores')
            plt.colorbar(im2, ax=axes[0,1])
            
            # Add text annotations
            for i in range(len(pivot_kd_lora.index)):
                for j in range(len(pivot_kd_lora.columns)):
                    axes[0,1].text(j, i, f'{pivot_kd_lora.iloc[i, j]:.3f}',
                                 ha='center', va='center', color='black', fontsize=8)
        
        # 2. Bar plot: Performance drop comparison
        # Prepare data for grouped bar plot
        plot_data = []
        for _, row in results_df.iterrows():
            if not pd.isna(row['Relative Drop (%)']):
                plot_data.append({
                    'Model Family': row['Model Family'],
                    'Strategy': row['Strategy'],
                    'Variant': row['Variant'],
                    'Drop': row['Relative Drop (%)']
                })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Create grouped bar plot for each model family
            for idx, model in enumerate(MODEL_DISPLAY_NAMES.values()):
                model_df = plot_df[plot_df['Model Family'] == model]
                if model_df.empty:
                    continue
                
                # Pivot for plotting
                pivot_drop = model_df.pivot_table(
                    index='Variant',
                    columns='Strategy',
                    values='Drop'
                )
                
                # Plot
                x = np.arange(len(PEFT_VARIANTS))
                width = 0.35
                
                if 'LoRA-only' in pivot_drop.columns:
                    axes[1,0].bar(x - width/2, pivot_drop['LoRA-only'], width, 
                                 label=f'{model} - LoRA-only', alpha=0.7)
                
                if 'KD-LoRA' in pivot_drop.columns:
                    axes[1,0].bar(x + width/2, pivot_drop['KD-LoRA'], width,
                                 label=f'{model} - KD-LoRA', alpha=0.7)
            
            axes[1,0].set_xlabel('PEFT Variant')
            axes[1,0].set_ylabel('Performance Drop vs FFT (%)')
            axes[1,0].set_title('Performance Drop Comparison Across Models')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(PEFT_VARIANTS, rotation=45)
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 3. Line plot: Model comparison for each variant
        for variant_idx, variant in enumerate(PEFT_VARIANTS):
            variant_df = results_df[results_df['Variant'] == variant]
            if variant_df.empty:
                continue
            
            # Separate by strategy
            for strategy in ['LoRA-only', 'KD-LoRA']:
                strat_df = variant_df[variant_df['Strategy'] == strategy]
                if not strat_df.empty:
                    x = [MODEL_DISPLAY_NAMES[m] for m in MODEL_FAMILIES if MODEL_DISPLAY_NAMES[m] in strat_df['Model Family'].values]
                    y = [strat_df[strat_df['Model Family'] == MODEL_DISPLAY_NAMES[m]]['Average Score'].iloc[0] 
                         for m in MODEL_FAMILIES if MODEL_DISPLAY_NAMES[m] in strat_df['Model Family'].values]
                    
                    if len(x) == len(y) and len(x) > 1:
                        axes[1,1].plot(x, y, marker='o', label=f'{variant} - {strategy}')
        
        axes[1,1].set_xlabel('Model Family')
        axes[1,1].set_ylabel('Average Score')
        axes[1,1].set_title('Variant Performance Across Models')
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cross_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Cross-model comparison plot saved to cross_model_comparison.png")
        
        # 4. Create summary statistic plot
        summary_stats = results_df.groupby(['Model Family', 'Strategy'])['Relative Drop (%)'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(MODEL_DISPLAY_NAMES))
        width = 0.35
        
        for i, strategy in enumerate(['LoRA-only', 'KD-LoRA']):
            strat_df = summary_stats[summary_stats['Strategy'] == strategy]
            if not strat_df.empty:
                means = [strat_df[strat_df['Model Family'] == m]['mean'].iloc[0] 
                        if m in strat_df['Model Family'].values else 0 for m in MODEL_DISPLAY_NAMES.values()]
                stds = [strat_df[strat_df['Model Family'] == m]['std'].iloc[0] 
                       if m in strat_df['Model Family'].values else 0 for m in MODEL_DISPLAY_NAMES.values()]
                
                plt.bar(x + i*width - width/2, means, width, label=strategy, alpha=0.7, yerr=stds, capsize=5)
        
        plt.xlabel('Model Family')
        plt.ylabel('Average Performance Drop vs FFT (%)')
        plt.title('Average Performance Drop by Model Family and Strategy')
        plt.xticks(x, MODEL_DISPLAY_NAMES.values())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_strategy_drop_summary.png', dpi=150)
        plt.close()
        print("Model-strategy drop summary saved to model_strategy_drop_summary.png")
        
    except Exception as e:
        print(f"Error generating plots: {e}")

def generate_summary_report(results_df):
    """Generate a comprehensive summary report."""
    if results_df.empty:
        print("No data for summary report.")
        return
    
    def write_report(f=None):
        """Write the report to file or stdout."""
        import sys
        if f is None:
            output = sys.stdout
        else:
            output = f
        
        output.write("=" * 80 + "\n")
        output.write("SUMMARY REPORT: KEY FINDINGS\n")
        output.write("=" * 80 + "\n")
        
        # Overall trends
        output.write("\n1. OVERALL TRENDS:\n")
        
        # Compute average drop across all models
        avg_drop_lora = results_df[results_df['Strategy'] == 'LoRA-only']['Relative Drop (%)'].mean()
        avg_drop_kd = results_df[results_df['Strategy'] == 'KD-LoRA']['Relative Drop (%)'].mean()
        
        output.write(f"  • LoRA-only average drop vs FFT: {avg_drop_lora:.2f}%\n")
        output.write(f"  • KD-LoRA average drop vs FFT: {avg_drop_kd:.2f}%\n")
        output.write(f"  • KD-LoRA underperforms LoRA-only by {avg_drop_kd - avg_drop_lora:.2f} percentage points\n")
        
        # Best variants overall
        output.write("\n2. BEST PERFORMING VARIANTS OVERALL:\n")
        for strategy in ['LoRA-only', 'KD-LoRA']:
            strat_df = results_df[results_df['Strategy'] == strategy]
            if not strat_df.empty:
                best_variant = strat_df.groupby('Variant')['Average Score'].mean().idxmax()
                best_score = strat_df.groupby('Variant')['Average Score'].mean().max()
                output.write(f"  • {strategy}: {best_variant} (avg: {best_score:.4f})\n")
        
        # Model-specific insights
        output.write("\n3. MODEL-SPECIFIC INSIGHTS:\n")
        for model in MODEL_DISPLAY_NAMES.values():
            model_df = results_df[results_df['Model Family'] == model]
            if model_df.empty:
                continue
            
            # Best variant for this model
            for strategy in ['LoRA-only', 'KD-LoRA']:
                strat_df = model_df[model_df['Strategy'] == strategy]
                if not strat_df.empty:
                    best_row = strat_df.loc[strat_df['Average Score'].idxmax()]
                    worst_row = strat_df.loc[strat_df['Average Score'].idxmin()]
                    output.write(f"  • {model} - {strategy}:\n")
                    output.write(f"    - Best: {best_row['Variant']} ({best_row['Average Score']:.4f}, "
                               f"Drop: {best_row['Relative Drop (%)']:.2f}%)\n")
                    output.write(f"    - Worst: {worst_row['Variant']} ({worst_row['Average Score']:.4f}, "
                               f"Drop: {worst_row['Relative Drop (%)']:.2f}%)\n")
        
        # Consistency across models
        output.write("\n4. CONSISTENCY ACROSS MODELS:\n")
        for variant in PEFT_VARIANTS:
            variant_df = results_df[results_df['Variant'] == variant]
            if variant_df.empty:
                continue
            
            # Compute rank consistency
            lora_ranks = variant_df[variant_df['Strategy'] == 'LoRA-only']['Rank'].tolist()
            kd_ranks = variant_df[variant_df['Strategy'] == 'KD-LoRA']['Rank'].tolist()
            
            if lora_ranks:
                avg_rank_lora = np.mean(lora_ranks)
                output.write(f"  • {variant} - LoRA-only: Average rank {avg_rank_lora:.1f} across models\n")
            
            if kd_ranks:
                avg_rank_kd = np.mean(kd_ranks)
                output.write(f"  • {variant} - KD-LoRA: Average rank {avg_rank_kd:.1f} across models\n")
        
        # Recommendations
        output.write("\n5. RECOMMENDATIONS FOR PAPER:\n")
        output.write("  • Highlight best-performing variants: rslora (LoRA-only) and mrlora (KD-LoRA)\n")
        output.write("  • Discuss the performance gap between LoRA-only and KD-LoRA strategies\n")
        output.write("  • Note model-specific patterns (e.g., DeBERTa shows different ranking)\n")
        output.write("  • Include cross-model comparison in supplementary material\n")
        output.write("  • Consider statistical significance given varying completion rates\n")
    
    # Print to console
    write_report()
    
    # Save summary to file
    summary_path = 'multi_model_summary.txt'
    with open(summary_path, 'w') as f:
        write_report(f)
    print(f"\nSummary report saved to {summary_path}")

def main():
    print("=" * 80)
    print("MULTI-MODEL TABLE II ANALYSIS")
    print("Comparing performance across BERT, RoBERTa, and DeBERTa")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading model tables...")
    model_data = load_model_tables()
    
    if not model_data:
        print("No model data found. Exiting.")
        return
    
    # Analyze each model
    print("\n2. Analyzing model performance...")
    results_df = analyze_model_performance(model_data)
    
    if results_df.empty:
        print("No results to analyze. Exiting.")
        return
    
    # Cross-model comparison
    print("\n3. Creating cross-model comparison...")
    create_cross_model_comparison(results_df)
    
    # Generate visualizations
    print("\n4. Generating visualizations...")
    plot_cross_model_comparison(results_df)
    
    # Generate summary report
    print("\n5. Generating summary report...")
    generate_summary_report(results_df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()