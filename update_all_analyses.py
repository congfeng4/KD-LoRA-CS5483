#!/usr/bin/env python3
"""
Master script to update all analysis outputs (PNG, CSV, TXT, MD files)
to reflect the latest data in results/ directory.
Run this script after new experiments are completed.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# List of analysis scripts to run in order
ANALYSIS_SCRIPTS = [
    # 1. Completion rate analysis (generates missing_experiments.csv)
    "analyze_completion_rate.py --plot",
    
    # 2. Efficiency and statistical analysis (core analysis)
    "analyze_efficiency_and_stats.py",
    
    # 3. Table I generation (MrLoRA only)
    "create_table_i.py",
    
    # 4. Table II generation (multi-model)
    "create_table_ii_multi_model.py",
    
    # 5. MrLoRA-specific analyses
    "analyze_mrlora_by_strategy.py",
    "analyze_mrlora_across_models.py",
    
    # 6. Additional analyses if needed
    # "analyze_results.py",  # Optional
]

# Output file patterns to check
OUTPUT_PATTERNS = [
    "*.csv",
    "*.png", 
    "*.txt",
    "*.md",
    "*.tex",
    "table_ii_*/table_iia_results.csv",
    "table_ii_*/table_iib_results.csv",
    "table_ii_*/table_iia_heatmap.png",
    "table_ii_*/table_iib_heatmap.png",
]

def run_analysis(script_cmd, script_index, total_scripts):
    """Run a single analysis script."""
    script_name = script_cmd.split()[0]
    print(f"\n{'='*80}")
    print(f"Running {script_index}/{total_scripts}: {script_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            ["python", "-u"] + script_cmd.split(),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per script
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully in {elapsed:.1f}s")
            # Print last 20 lines of output
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 20:
                print("\nLast 20 lines of output:")
                for line in output_lines[-20:]:
                    print(line)
            else:
                print(f"\nOutput:\n{result.stdout}")
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {script_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ {script_name} failed with exception: {e}")
        return False
    
    return True

def check_output_files():
    """Check which output files were created/modified."""
    print(f"\n{'='*80}")
    print("CHECKING OUTPUT FILES")
    print(f"{'='*80}")
    
    output_files = []
    for pattern in OUTPUT_PATTERNS:
        for path in Path('.').glob(pattern):
            if path.is_file():
                # Get modification time
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                output_files.append((path, mtime))
    
    if output_files:
        print(f"Found {len(output_files)} output files:")
        for path, mtime in sorted(output_files):
            print(f"  {path} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        print("No output files found matching patterns.")
    
    return output_files

def create_summary_report():
    """Create a summary report of the analysis."""
    summary = []
    summary.append("="*80)
    summary.append("ANALYSIS UPDATE SUMMARY")
    summary.append("="*80)
    summary.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Results directory: results/")
    
    # Count experiments
    try:
        import json
        json_files = list(Path('results').rglob('*.json'))
        summary.append(f"Total JSON files in results/: {len(json_files)}")
    except:
        summary.append("Could not count JSON files")
    
    # Check completion rate
    try:
        import pandas as pd
        if os.path.exists('missing_experiments.csv'):
            missing_df = pd.read_csv('missing_experiments.csv')
            summary.append(f"Missing experiments: {len(missing_df)}")
    except:
        summary.append("Could not read missing experiments")
    
    # Check efficiency metrics
    try:
        if os.path.exists('efficiency_metrics_all.csv'):
            df = pd.read_csv('efficiency_metrics_all.csv')
            summary.append(f"Total experiments analyzed: {len(df)}")
    except:
        pass
    
    summary.append("\nGenerated files:")
    
    # List key files
    key_files = [
        'missing_experiments.csv',
        'efficiency_metrics_all.csv',
        'statistical_test_results.csv',
        'table_i_results.csv',
        'model_family_comparison.png',
        'efficiency_performance_scatter.png',
        'parameter_efficiency_analysis.png',
    ]
    
    for file in key_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            summary.append(f"  ✓ {file} ({size:,} bytes)")
        else:
            summary.append(f"  ✗ {file} (missing)")
    
    # Check table directories
    table_dirs = ['table_ii_bert', 'table_ii_roberta', 'table_ii_deberta']
    for dir_name in table_dirs:
        if os.path.exists(dir_name):
            files = list(Path(dir_name).glob('*'))
            summary.append(f"  ✓ {dir_name}/ ({len(files)} files)")
        else:
            summary.append(f"  ✗ {dir_name}/ (missing)")
    
    summary.append("\n" + "="*80)
    
    # Write summary to file
    summary_path = 'analysis_update_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print('\n'.join(summary))
    print(f"\nSummary saved to {summary_path}")
    
    return summary_path

def main():
    """Main function to update all analyses."""
    print(f"{'='*80}")
    print("UPDATING ALL ANALYSES")
    print(f"{'='*80}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if results directory exists
    if not os.path.exists('results'):
        print("ERROR: results/ directory not found!")
        return 1
    
    # Check for required packages
    try:
        import pandas
        import numpy
        import matplotlib
        print("✓ Required packages (pandas, numpy, matplotlib) are available")
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install required packages: pip install pandas numpy matplotlib")
        return 1
    
    # Run all analysis scripts
    successful = 0
    total = len(ANALYSIS_SCRIPTS)
    
    for i, script_cmd in enumerate(ANALYSIS_SCRIPTS, 1):
        if run_analysis(script_cmd, i, total):
            successful += 1
    
    # Check output files
    output_files = check_output_files()
    
    # Create summary report
    summary_path = create_summary_report()
    
    # Final status
    print(f"\n{'='*80}")
    print("UPDATE COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully ran {successful}/{total} analysis scripts")
    print(f"Generated/updated {len(output_files)} output files")
    print(f"Summary report: {summary_path}")
    
    if successful == total:
        print("\n✓ All analyses updated successfully!")
        return 0
    else:
        print(f"\n⚠ {total - successful} analysis scripts failed")
        print("Check the output above for errors.")
        return 1

if __name__ == "__main__":
    sys.exit(main())