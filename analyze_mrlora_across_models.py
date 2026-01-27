"""
Analyze mrlora performance across all three model families (BERT, RoBERTa, DeBERTa).
"""
import pandas as pd
import numpy as np

def analyze_mrlora_across_models():
    print("=== MRLORA PERFORMANCE ACROSS MODEL FAMILIES ===\n")
    print("Analyzing BERT, RoBERTa, and DeBERTa performance separately\n")
    
    # 1. Load multi-model analysis data
    multi_df = pd.read_csv('multi_model_analysis_results.csv')
    
    # Filter for mrlora
    mrlora_multi = multi_df[multi_df['Variant'] == 'mrlora'].copy()
    
    print("1. PERFORMANCE BY MODEL FAMILY AND STRATEGY\n")
    print("=" * 80)
    
    # Display performance data
    for _, row in mrlora_multi.iterrows():
        model = row['Model Family']
        strategy = row['Strategy']
        print(f"{model:12s} - {strategy:10s}:")
        print(f"  Average Score: {row['Average Score']:.4f}")
        print(f"  Drop vs FFT: {row['Relative Drop (%)']:.2f}%")
        print(f"  FFT Baseline: {row['FFT Baseline']:.4f}")
        print(f"  Rank among 6 variants: {int(row['Rank'])}/6")
        print()
    
    print("\n2. COMPARATIVE RANKING WITHIN EACH MODEL FAMILY\n")
    print("=" * 80)
    
    model_families = ['BERT', 'RoBERTa', 'DeBERTa-v3']
    strategies = ['LoRA-only', 'KD-LoRA']
    
    for model in model_families:
        print(f"\n{model}:")
        for strategy in strategies:
            model_data = multi_df[(multi_df['Model Family'] == model) & 
                                 (multi_df['Strategy'] == strategy)].copy()
            model_data_sorted = model_data.sort_values('Average Score', ascending=False).reset_index(drop=True)
            
            # Find mrlora rank
            mrlora_rank = model_data_sorted[model_data_sorted['Variant'] == 'mrlora'].index[0] + 1
            mrlora_score = model_data_sorted[model_data_sorted['Variant'] == 'mrlora']['Average Score'].values[0]
            best_score = model_data_sorted.iloc[0]['Average Score']
            best_variant = model_data_sorted.iloc[0]['Variant']
            
            print(f"  {strategy:10s}: mrlora Rank {mrlora_rank}/6")
            print(f"    Score: {mrlora_score:.4f} (Best: {best_variant} = {best_score:.4f})")
            print(f"    Difference: {mrlora_score - best_score:+.4f} ({((mrlora_score - best_score)/best_score*100):+.1f}%)")
    
    print("\n3. EFFICIENCY METRICS BY MODEL FAMILY\n")
    print("=" * 80)
    
    # Load efficiency metrics
    try:
        eff_df = pd.read_csv('efficiency_metrics_all.csv')
        
        # Filter for mrlora
        mrlora_eff = eff_df[eff_df['peft'] == 'mrlora'].copy()
        
        if not mrlora_eff.empty:
            # Group by model family and strategy
            grouped = mrlora_eff.groupby(['model_family', 'variant']).agg({
                'metric_value': ['mean', 'std', 'count'],
                'train_time': 'mean',
                'trainable_params_count': 'mean',
                'avg_memory_allocated_mb': 'mean',
                'throughput_samples_per_second': 'mean'
            }).round(3)
            
            print("Efficiency metrics for mrlora by model family:")
            
            # Flatten the grouped dataframe for display
            for (model_family, variant), group_data in grouped.iterrows():
                print(f"\n{model_family:10s} - {variant:10s}:")
                print(f"  Performance: {group_data[('metric_value', 'mean')]:.4f} (std: {group_data[('metric_value', 'std')]:.4f}, n={group_data[('metric_value', 'count')]:.0f})")
                print(f"  Train Time: {group_data[('train_time', 'mean')]:.1f}s")
                print(f"  Parameters: {group_data[('trainable_params_count', 'mean')]:.3f}M")
                print(f"  Memory: {group_data[('avg_memory_allocated_mb', 'mean')]:.1f}MB")
                print(f"  Throughput: {group_data[('throughput_samples_per_second', 'mean')]:.1f} samples/s")
        else:
            print("No efficiency metrics found for mrlora in efficiency_metrics_all.csv")
    
    except Exception as e:
        print(f"Could not load efficiency metrics: {e}")
        print("Using multi-model analysis data only.")
    
    print("\n4. MODEL-SPECIFIC PATTERNS AND INSIGHTS\n")
    print("=" * 80)
    
    # Analyze patterns across models
    for model in model_families:
        model_data = mrlora_multi[mrlora_multi['Model Family'] == model]
        if not model_data.empty:
            print(f"\n{model}:")
            
            # Extract both strategies
            lora_only = model_data[model_data['Strategy'] == 'LoRA-only']
            kd_lora = model_data[model_data['Strategy'] == 'KD-LoRA']
            
            if not lora_only.empty:
                lo_row = lora_only.iloc[0]
                print(f"  LoRA-only: {lo_row['Average Score']:.4f} (Drop: {lo_row['Relative Drop (%)']:.2f}%, Rank: {int(lo_row['Rank'])}/6)")
            
            if not kd_lora.empty:
                kd_row = kd_lora.iloc[0]
                print(f"  KD-LoRA:   {kd_row['Average Score']:.4f} (Drop: {kd_row['Relative Drop (%)']:.2f}%, Rank: {int(kd_row['Rank'])}/6)")
            
            # Calculate difference between strategies
            if not lora_only.empty and not kd_lora.empty:
                diff = lo_row['Average Score'] - kd_row['Average Score']
                print(f"  Strategy difference: {diff:+.4f} (LoRA-only {'better' if diff > 0 else 'worse'})")
    
    print("\n5. OVERALL ASSESSMENT ACROSS MODEL FAMILIES\n")
    print("=" * 80)
    
    # Calculate averages across models
    summary_data = []
    for model in model_families:
        model_data = mrlora_multi[mrlora_multi['Model Family'] == model]
        for _, row in model_data.iterrows():
            summary_data.append({
                'Model': model,
                'Strategy': row['Strategy'],
                'Score': row['Average Score'],
                'Drop': row['Relative Drop (%)'],
                'Rank': row['Rank']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Average across models
    print("\nAverage performance across all three model families:")
    for strategy in strategies:
        strat_data = summary_df[summary_df['Strategy'] == strategy]
        if not strat_data.empty:
            avg_score = strat_data['Score'].mean()
            avg_drop = strat_data['Drop'].mean()
            avg_rank = strat_data['Rank'].mean()
            print(f"\n  {strategy}:")
            print(f"    Average Score: {avg_score:.4f}")
            print(f"    Average Drop vs FFT: {avg_drop:.2f}%")
            print(f"    Average Rank: {avg_rank:.1f}/6")
    
    print("\n6. KEY FINDINGS\n")
    print("=" * 80)
    
    # Extract key insights
    print("\na) Performance Consistency:")
    for model in model_families:
        model_ranks = summary_df[summary_df['Model'] == model]['Rank'].values
        print(f"   {model:12s}: LoRA-only Rank {int(model_ranks[0])}, KD-LoRA Rank {int(model_ranks[1])}")
    
    print("\nb) Best Performance:")
    best_scores = {}
    for model in model_families:
        model_scores = summary_df[summary_df['Model'] == model]
        best_strat = model_scores.loc[model_scores['Score'].idxmax()]
        best_scores[model] = best_strat
        print(f"   {model:12s}: {best_strat['Strategy']} ({best_strat['Score']:.4f})")
    
    print("\nc) Worst Performance:")
    for model in model_families:
        model_scores = summary_df[summary_df['Model'] == model]
        worst_strat = model_scores.loc[model_scores['Score'].idxmin()]
        print(f"   {model:12s}: {worst_strat['Strategy']} ({worst_strat['Score']:.4f})")
    
    print("\nd) Performance Drop vs FFT:")
    for model in model_families:
        model_data = summary_df[summary_df['Model'] == model]
        avg_drop = model_data['Drop'].mean()
        print(f"   {model:12s}: {avg_drop:.2f}% average drop")
    
    print("\ne) Recommendations for Paper:")
    print("   1. Report mrlora performance separately for each model family")
    print("   2. Highlight where mrlora excels (e.g., BERT with KD-LoRA)")
    print("   3. Discuss consistency of rankings across model families")
    print("   4. Note any model-specific patterns (e.g., better with DeBERTa)")
    print("   5. Include efficiency metrics if available by model family")

if __name__ == "__main__":
    analyze_mrlora_across_models()