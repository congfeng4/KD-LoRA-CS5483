#!/usr/bin/env python3
"""
Create comprehensive tables for teacher and student LoRA variants.
Includes FFT baseline and per-family breakdown where available.
"""
import pandas as pd
import numpy as np

# Read data
glue_df = pd.read_csv('glue_averages.csv')
tradeoff_df = pd.read_csv('tradeoff_data.csv')
student_df = pd.read_csv('table_i_glue_scores.csv')
mrlora_per_task = pd.read_csv('table_i_mrlora.csv')
detailed_per_task = pd.read_csv('table_i_detailed.csv')

# Get FFT values for each model family
fft_glue = {}
fft_params = {}
for model_family in ['bert', 'roberta', 'deberta']:
    fft_glue[model_family] = tradeoff_df[(tradeoff_df['model_family'] == model_family) & (tradeoff_df['method'] == 'fft')]['glue_score'].values[0]
    fft_params[model_family] = tradeoff_df[(tradeoff_df['model_family'] == model_family) & (tradeoff_df['method'] == 'fft')]['param_count_m'].values[0]

# Method names mapping
method_names = {
    'fft': 'FFT',
    'lora': 'LoRA',
    'mrlora': 'MR-LoRA',
    'adalora': 'AdaLoRA',
    'dora': 'DoRA',
    'olora': 'OLoRA',
    'rslora': 'RS-LoRA',
    'mrlora-rs': 'MR-LoRA-RS'
}

# ===== TEACHER TABLE =====
print("=== TEACHER VARIANTS ===")
teacher_methods = ['fft', 'lora', 'mrlora', 'adalora', 'dora', 'olora', 'rslora', 'mrlora-rs']

teacher_data = []
for method in teacher_methods:
    row = {'Method': method_names[method]}
    
    if method == 'fft':
        # FFT baseline: 100% for everything
        row['GLUE (%FFT)'] = '100.0'
        row['Params (%FFT)'] = '100.00'
        row['BERT'] = '100.0'
        row['RoBERTa'] = '100.0'
        row['DeBERTa'] = '100.0'
    else:
        # Compute averages across families
        glue_pcts = []
        param_pcts = []
        family_pcts = {}
        
        for model_family in ['bert', 'roberta', 'deberta']:
            # Get GLUE score
            if method in glue_df[glue_df['model_family'] == model_family]['method'].values:
                glue_score = glue_df[(glue_df['model_family'] == model_family) & (glue_df['method'] == method)]['glue_score'].values[0]
                glue_pct = (glue_score / fft_glue[model_family]) * 100
                family_pcts[model_family.capitalize()] = f'{glue_pct:.1f}'
                glue_pcts.append(glue_pct)
            else:
                family_pcts[model_family.capitalize()] = '--'
            
            # Get param count
            if method in tradeoff_df[tradeoff_df['model_family'] == model_family]['method'].values:
                param_count = tradeoff_df[(tradeoff_df['model_family'] == model_family) & (tradeoff_df['method'] == method)]['param_count_m'].values[0]
                param_pct = (param_count / fft_params[model_family]) * 100
                param_pcts.append(param_pct)
        
        # Average across families
        avg_glue = np.mean(glue_pcts) if glue_pcts else 0
        avg_params = np.mean(param_pcts) if param_pcts else 0
        
        row['GLUE (%FFT)'] = f'{avg_glue:.1f}'
        row['Params (%FFT)'] = f'{avg_params:.3f}'
        row['BERT'] = family_pcts.get('Bert', '--')
        row['RoBERTa'] = family_pcts.get('Roberta', '--')
        row['DeBERTa'] = family_pcts.get('Deberta', '--')
    
    teacher_data.append(row)

# Sort by GLUE percentage (descending), but keep FFT at top
teacher_data_sorted = [teacher_data[0]]  # FFT first
teacher_data_sorted.extend(sorted(
    [row for row in teacher_data[1:] if row['GLUE (%FFT)'] != '--'],
    key=lambda x: float(x['GLUE (%FFT)']),
    reverse=True
))

# Generate LaTeX for teacher table
teacher_latex = """\\begin{table}[ht]
\\centering
\\caption{Performance and parameter efficiency of LoRA variants in \\textit{teacher} fine‑tuning. Values are expressed as percentage of full fine‑tuning (FFT). The first row shows FFT itself (100\\%).}
\\label{tab:teacher-variants-comprehensive}
\\begin{tabular}{lcccc}
\\toprule
Method & GLUE (\\% FFT) & Parameters (\\% FFT) & BERT & RoBERTa & DeBERTa \\\\
\\midrule
"""

for row in teacher_data_sorted:
    teacher_latex += f"{row['Method']:12} & {row['GLUE (%FFT)']:>6} & {row['Params (%FFT)']:>8} & {row['BERT']:>5} & {row['RoBERTa']:>8} & {row['DeBERTa']:>8} \\\\\n"

teacher_latex += """\\bottomrule
\\end{tabular}
\\end{table}"""

print(teacher_latex)

# ===== STUDENT TABLE =====
print("\n\n=== STUDENT VARIANTS ===")
# Student methods: only LoRA and MR-LoRA have data
student_methods = ['fft', 'lora', 'mrlora']

# Get student GLUE scores from table_i_glue_scores.csv
fft_glue_student = {}
for idx, row in student_df.iterrows():
    model_family = row['Model Family']
    fft_glue_student[model_family] = row['FFT']

student_data = []
for method in student_methods:
    if method == 'fft':
        row = {'Method': 'FFT', 'GLUE (%FFT)': '100.0', 'Params (%FFT)': '100.00'}
        for fam in ['bert', 'roberta', 'deberta']:
            row[fam.capitalize()] = '100.0'
    elif method == 'lora':
        # Student LoRA (KD-LoRA)
        glue_pcts = []
        param_pcts = []
        family_pcts = {}
        
        for model_family in ['bert', 'roberta', 'deberta']:
            # Get student GLUE score from table_i_glue_scores.csv
            student_score = student_df[student_df['Model Family'] == model_family]['Student LoRA'].values[0]
            glue_pct = (student_score / fft_glue_student[model_family]) * 100
            family_pcts[model_family.capitalize()] = f'{glue_pct:.1f}'
            glue_pcts.append(glue_pct)
            
            # Use LoRA teacher param percentage (similar for student)
            if 'lora' in tradeoff_df[tradeoff_df['model_family'] == model_family]['method'].values:
                param_count = tradeoff_df[(tradeoff_df['model_family'] == model_family) & (tradeoff_df['method'] == 'lora')]['param_count_m'].values[0]
                param_pct = (param_count / fft_params[model_family]) * 100
                param_pcts.append(param_pct)
        
        avg_glue = np.mean(glue_pcts)
        avg_params = np.mean(param_pcts) if param_pcts else 0.457  # default from teacher
        
        row = {
            'Method': 'LoRA',
            'GLUE (%FFT)': f'{avg_glue:.1f}',
            'Params (%FFT)': f'{avg_params:.3f}',
            'BERT': family_pcts.get('Bert', '--'),
            'RoBERTa': family_pcts.get('Roberta', '--'),
            'DeBERTa': family_pcts.get('Deberta', '--')
        }
    elif method == 'mrlora':
        # Student MR-LoRA
        glue_pcts = []
        param_pcts = []
        family_pcts = {}
        
        for model_family in ['bert', 'roberta', 'deberta']:
            # Get student GLUE score from table_i_glue_scores.csv
            student_score = student_df[student_df['Model Family'] == model_family]['Student MR‑LoRA'].values[0]
            glue_pct = (student_score / fft_glue_student[model_family]) * 100
            family_pcts[model_family.capitalize()] = f'{glue_pct:.1f}'
            glue_pcts.append(glue_pct)
            
            # Use MR-LoRA teacher param percentage
            if 'mrlora' in tradeoff_df[tradeoff_df['model_family'] == model_family]['method'].values:
                param_count = tradeoff_df[(tradeoff_df['model_family'] == model_family) & (tradeoff_df['method'] == 'mrlora')]['param_count_m'].values[0]
                param_pct = (param_count / fft_params[model_family]) * 100
                param_pcts.append(param_pct)
        
        avg_glue = np.mean(glue_pcts)
        avg_params = np.mean(param_pcts) if param_pcts else 0.563  # default from teacher
        
        row = {
            'Method': 'MR-LoRA',
            'GLUE (%FFT)': f'{avg_glue:.1f}',
            'Params (%FFT)': f'{avg_params:.3f}',
            'BERT': family_pcts.get('Bert', '--'),
            'RoBERTa': family_pcts.get('Roberta', '--'),
            'DeBERTa': family_pcts.get('Deberta', '--')
        }
    
    student_data.append(row)

# Generate LaTeX for student table
student_latex = """\\begin{table}[ht]
\\centering
\\caption{Performance and parameter efficiency of LoRA variants in \\textit{student} distillation. Values are expressed as percentage of teacher's full fine‑tuning (FFT). Only LoRA and MR‑LoRA were evaluated in the student distillation setting.}
\\label{tab:student-variants-comprehensive}
\\begin{tabular}{lcccc}
\\toprule
Method & GLUE (\\% FFT) & Parameters (\\% FFT) & BERT & RoBERTa & DeBERTa \\\\
\\midrule
"""

for row in student_data:
    student_latex += f"{row['Method']:12} & {row['GLUE (%FFT)']:>6} & {row['Params (%FFT)']:>8} & {row.get('BERT', '--'):>5} & {row.get('RoBERTa', '--'):>8} & {row.get('DeBERTa', '--'):>8} \\\\\n"

student_latex += """\\bottomrule
\\end{tabular}
\\end{table}"""

print(student_latex)

# ===== PER-TASK PERCENTAGES TABLE (for variants with per-task data) =====
print("\n\n=== PER-TASK PERCENTAGES ===")
# We have per-task data for: FFT, Teacher LoRA, Student LoRA (KD-LoRA), Teacher MR-LoRA, Student MR-LoRA
# Let's create a table showing per-task percentages for these methods

# Extract per-task data
tasks = ['CoLA', 'SST-2', 'MRPC', 'QQP', 'STS-B', 'QNLI', 'RTE', 'WNLI', 'MNLI_m', 'MNLI_mm', 'Score']

# For simplicity, let's create a table showing GLUE Score percentages for BERT only
print("\nPer-task percentages would require a very wide table.")
print("Instead, showing GLUE score percentages by model family (above) is more practical.")

# Write tables to files
with open('teacher_table_comprehensive.tex', 'w') as f:
    f.write(teacher_latex)

with open('student_table_comprehensive.tex', 'w') as f:
    f.write(student_latex)

print("\nGenerated teacher_table_comprehensive.tex and student_table_comprehensive.tex")