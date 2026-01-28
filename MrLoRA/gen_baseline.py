import csv
with open('table_i_detailed.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)
header = rows[0]
tasks = [row[0] for row in rows[1:-1]]
score_row = rows[-1]
# column indices: 1-3 bert, 4-6 roberta, 7-9 deberta
def fmt(val):
    return val.rstrip('0').rstrip('.') if '.' in val else val
lines = []
lines.append(r'\subsubsection{Baseline LoRA and KD‑LoRA Results}')
lines.append(r'Table~\ref{tab:baseline-lora} reports the per‑task GLUE scores for the baseline methods: full fine‑tuning (FFT), standard LoRA applied to the teacher (Teacher LoRA), and single‑rank LoRA applied to the frozen student with knowledge distillation (KD‑LoRA).')
lines.append(r'\begin{table}[ht]')
lines.append(r'\centering')
lines.append(r'\caption{Per‑task GLUE scores for baseline methods. FFT denotes full fine‑tuning of the teacher; Teacher LoRA applies standard low‑rank adapters directly to the teacher; KD‑LoRA refers to single‑rank adapters applied to the frozen student with knowledge distillation. Abbreviations and metrics as in Table~\ref{tab:glue-scores}.}')
lines.append(r'\label{tab:baseline-lora}')
lines.append(r'\begin{tabular}{lccccccccc}')
lines.append(r'\toprule')
lines.append(r'\multirow{2}{*}{Task} & \multicolumn{3}{c}{BERT‑b/DBERT‑b} & \multicolumn{3}{c}{RoB‑b/DRoB‑b} & \multicolumn{3}{c}{DeB‑b/DeB‑s} \\')
lines.append(r'\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}')
lines.append(r' & FFT & Teacher LoRA & KD‑LoRA & FFT & Teacher LoRA & KD‑LoRA & FFT & Teacher LoRA & KD‑LoRA \\')
lines.append(r'\midrule')
for i, task in enumerate(tasks):
    vals = rows[i+1][1:]
    bert = vals[0:3]
    roberta = vals[3:6]
    deberta = vals[6:9]
    # escape underscores
    task_tex = task.replace('_', '\\_')
    line = f'{task_tex:10} & {bert[0]} & {bert[1]} & {bert[2]} & {roberta[0]} & {roberta[1]} & {roberta[2]} & {deberta[0]} & {deberta[1]} & {deberta[2]} \\\\'
    lines.append(line)
lines.append(r'\midrule')
score_vals = score_row[1:]
bert = score_vals[0:3]
roberta = score_vals[3:6]
deberta = score_vals[6:9]
lines.append(f'Score     & {bert[0]} & {bert[1]} & {bert[2]} & {roberta[0]} & {roberta[1]} & {roberta[2]} & {deberta[0]} & {deberta[1]} & {deberta[2]} \\\\')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')
with open('baseline_table.tex', 'w') as f:
    f.write('\n'.join(lines))
print('Generated baseline_table.tex')
