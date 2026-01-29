#!/usr/bin/env python3
"""
Generate LaTeX tables for teacher and student LoRA variants.
Replaces Figures 2 and 3 in the manuscript.
"""
import pandas as pd
import numpy as np

# Read data
glue_df = pd.read_csv('glue_averages.csv')
tradeoff_df = pd.read_csv('tradeoff_data.csv')
student_df = pd.read_csv('table_i_glue_scores.csv')

# Get FFT values for each model family
fft_glue = {}
fft_params = {}
for model_family in ['bert', 'roberta', 'deberta']:
    fft_glue[model_family] = tradeoff_df[(tradeoff_df['model_family'] == model_family) & (tradeoff_df['method'] == 'fft')]['glue_score'].values[0]
    fft_params[model_family] = tradeoff_df[(tradeoff_df['model_family'] == model_family) & (tradeoff_df['method'] == 'fft')]['param_count_m'].values[0]

# Methods to include for teacher table
teacher_methods = ['lora', 'mrlora', 'adalora', 'dora', 'olora', 'rslora', 'mrlora-rs']
teacher_names = {
    'lora': 'LoRA',
    'mrlora': 'MR-LoRA',
    'adalora': 'AdaLoRA',
    'dora': 'DoRA',
    'olora': 'OLoRA',
    'rslora': 'RS-LoRA',
    'mrlora-rs': 'MR-LoRA-RS'
}

# Compute teacher percentages
teacher_table = []
for method in teacher_methods:
    if method not in glue_df['method'].values:
        continue
    
    row = {'Method': teacher_names[method]}
    glue_pcts = []
    param_pcts = []
    
    for model_family in ['bert', 'roberta', 'deberta']:
        # Get GLUE score
        glue_score = glue_df[(glue_df['model_family'] == model_family) & (glue_df['method'] == method)]['glue_score'].values[0]
        glue_pct = (glue_score / fft_glue[model_family]) * 100
        glue_pcts.append(glue_pct)
        
        # Get param count
        if method in tradeoff_df[tradeoff_df['model_family'] == model_family]['method'].values:
            param_count = tradeoff_df[(tradeoff_df['model_family'] == model_family) & (tradeoff_df['method'] == method)]['param_count_m'].values[0]
            param_pct = (param_count / fft_params[model_family]) * 100
            param_pcts.append(param_pct)
    
    # Average across families
    avg_glue = np.mean(glue_pcts)
    avg_params = np.mean(param_pcts)
    
    teacher_table.append({
        'Method': teacher_names[method],
        'GLUE_pct': f'{avg_glue:.1f}',
        'Params_pct': f'{avg_params:.3f}',
        'sort_key': avg_glue
    })

# Sort by GLUE percentage (descending)
teacher_table.sort(key=lambda x: x['sort_key'], reverse=True)

# Generate LaTeX for teacher table
teacher_latex = """\\begin{table}[ht]
\\centering
\\caption{Average performance and parameter efficiency of LoRA variants in \\textit{teacher} fine‑tuning, expressed as percentage of full fine‑tuning (FFT). Values are averaged across BERT, RoBERTa, and DeBERTa model families.}
\\label{tab:teacher-variants}
\\begin{tabular}{lcc}
\\toprule
Method & GLUE (\\% FFT) & Parameters (\\% FFT) \\\\
\\midrule
"""

for row in teacher_table:
    teacher_latex += f"{row['Method']} & {row['GLUE_pct']} & {row['Params_pct']} \\\\\n"

teacher_latex += """\\bottomrule
\\end{tabular}
\\end{table}"""

# Student table (only LoRA and MR-LoRA have data)
student_methods = ['Student LoRA', 'Student MR‑LoRA']
student_short = {'Student LoRA': 'LoRA', 'Student MR‑LoRA': 'MR-LoRA'}

# Get FFT GLUE from student dataframe
fft_glue_student = {}
for idx, row in student_df.iterrows():
    model_family = row['Model Family']
    fft_glue_student[model_family] = row['FFT']

student_table = []
for method in student_methods:
    if method not in student_df.columns:
        continue
    
    row = {'Method': student_short[method]}
    glue_pcts = []
    param_pcts = []
    
    for model_family in ['bert', 'roberta', 'deberta']:
        # Get student GLUE score
        student_score = student_df[student_df['Model Family'] == model_family][method].values[0]
        glue_pct = (student_score / fft_glue_student[model_family]) * 100
        glue_pcts.append(glue_pct)
        
        # Use corresponding param count from tradeoff data
        short_method = student_short[method].lower().replace('-', '')
        if short_method == 'mrlora':
            param_method = 'mrlora'
        else:
            param_method = 'lora'
            
        if param_method in tradeoff_df[tradeoff_df['model_family'] == model_family]['method'].values:
            param_count = tradeoff_df[(tradeoff_df['model_family'] == model_family) & (tradeoff_df['method'] == param_method)]['param_count_m'].values[0]
            param_pct = (param_count / fft_params[model_family]) * 100
            param_pcts.append(param_pct)
    
    # Average across families
    avg_glue = np.mean(glue_pcts)
    avg_params = np.mean(param_pcts)
    
    student_table.append({
        'Method': student_short[method],
        'GLUE_pct': f'{avg_glue:.1f}',
        'Params_pct': f'{avg_params:.3f}',
        'sort_key': avg_glue
    })

# Sort by GLUE percentage (descending)
student_table.sort(key=lambda x: x['sort_key'], reverse=True)

# Generate LaTeX for student table
student_latex = """\\begin{table}[ht]
\\centering
\\caption{Average performance and parameter efficiency of LoRA variants in \\textit{student} distillation, expressed as percentage of teacher's full fine‑tuning (FFT). Values are averaged across BERT, RoBERTa, and DeBERTa model families.}
\\label{tab:student-variants}
\\begin{tabular}{lcc}
\\toprule
Method & GLUE (\\% FFT) & Parameters (\\% FFT) \\\\
\\midrule
"""

for row in student_table:
    student_latex += f"{row['Method']} & {row['GLUE_pct']} & {row['Params_pct']} \\\\\n"

student_latex += """\\bottomrule
\\end{tabular}
\\end{table}"""

# Write to files
with open('teacher_variants_table.tex', 'w') as f:
    f.write(teacher_latex)

with open('student_variants_table.tex', 'w') as f:
    f.write(student_latex)

print("Generated teacher_variants_table.tex and student_variants_table.tex")
print("\nTeacher table:")
print(teacher_latex)
print("\nStudent table:")
print(student_latex)