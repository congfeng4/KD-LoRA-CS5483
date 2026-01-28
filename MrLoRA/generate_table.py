import csv
with open('table_i_mrlora.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)
header = rows[0]
tasks = [row[0] for row in rows[1:-1]]
score_row = rows[-1]
# column indices: 1-3 bert, 4-6 roberta, 7-9 deberta
def fmt(val):
    return val.rstrip('0').rstrip('.') if '.' in val else val
lines = []
lines.append(r'\begin{table}[ht]')
lines.append(r'\centering')
lines.append(r'\caption{Per‑task GLUE scores (higher is better) for each model family and fine‑tuning strategy. FFT denotes full fine‑tuning of the teacher; Teacher MR‑LoRA applies multi‑rank low‑rank adapters directly to the teacher; Student MR‑LoRA refers to multi‑rank adapters applied to the frozen student with knowledge distillation. Abbreviations: BERT‑b/DBERT‑b (BERT‑base teacher / DistilBERT‑base student), RoB‑b/DRoB‑b (RoBERTa‑base / DistilRoBERTa‑base), DeB‑b/DeB‑s (DeBERTa‑v3‑base / DeBERTa‑v3‑small). Metrics: CoLA (Matthews correlation), SST‑2 (accuracy), MRPC (average of accuracy and F1), QQP (average of accuracy and F1), STS‑B (Pearson correlation), QNLI (accuracy), RTE (accuracy), WNLI (accuracy), MNLI\_m (matched accuracy), MNLI\_mm (mismatched accuracy). Score is the average across the ten tasks.}')
lines.append(r'\label{tab:glue-scores}')
lines.append(r'\begin{tabular}{lccccccccc}')
lines.append(r'\toprule')
lines.append(r'\multirow{2}{*}{Task} & \multicolumn{3}{c}{BERT‑b/DBERT‑b} & \multicolumn{3}{c}{RoB‑b/DRoB‑b} & \multicolumn{3}{c}{DeB‑b/DeB‑s} \\')
lines.append(r'\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}')
lines.append(r' & FFT & Teacher MR‑LoRA & Student MR‑LoRA & FFT & Teacher MR‑LoRA & Student MR‑LoRA & FFT & Teacher MR‑LoRA & Student MR‑LoRA \\')
lines.append(r'\midrule')
for i, task in enumerate(tasks):
    vals = rows[i+1][1:]
    bert = vals[0:3]
    roberta = vals[3:6]
    deberta = vals[6:9]
    line = f'{task:10} & {bert[0]} & {bert[1]} & {bert[2]} & {roberta[0]} & {roberta[1]} & {roberta[2]} & {deberta[0]} & {deberta[1]} & {deberta[2]} \\\\'
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
print('\n'.join(lines))
