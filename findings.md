# Findings

## Data Sources

1. **CSV files in MrLoRA/**:
   - `teacher_per_task.csv`: raw per‑task scores for teacher variants (LoRA, MR‑LoRA, AdaLoRA, DoRA, OLoRA, RS‑LoRA, MR‑LoRA‑RS) across BERT, RoBERTa, DeBERTa.
   - `student_per_task.csv`: raw per‑task scores only for `lora_student` and `mrlora_student`.
   - `glue_averages.csv`: GLUE averages for all variants (including student distillation) – used for overall GLUE percentages.
   - `table_i_detailed.csv`: FFT, LoRA, KD‑LoRA per‑task scores (FFT baseline).
   - `table_i_mrlora.csv`: FFT, Teacher MR‑LoRA, Student MR‑LoRA per‑task scores.

2. **Generated percentage tables**:
   - `teacher_per_task_pct.tex`: already contains the same percentages as Table 2 in main.tex (except STS‑B for MR‑LoRA‑RS is '--').
   - `student_per_task_pct.tex`: matches Table 3 exactly.

3. **Results directory structure**:
   - `results/lora/task_{task}_{model}_{seed}/base_.../peft_{method}_.../metrics.json`
   - Each JSON contains evaluation metrics (`eval_matthews_correlation`, `eval_accuracy`, etc.) and configuration (`type`, `teacher_model_name`, `student_model_name`).
   - The `type` field distinguishes teacher fine‑tuning (type=2) from student distillation (type=1). Confirmed via inspection.

## Missing Cells Analysis

### Table 2 (teacher variants)
- STS‑B column for MR‑LoRA‑RS is '--'.
- Raw data missing: `teacher_per_task.csv` shows empty STS‑B for bert & roberta MR‑LoRA‑RS, and empty QQP & STS‑B for deberta MR‑LoRA‑RS.
- Therefore '--' is correct; cannot fill.

### Table 3 (student variants)
- Per‑task columns for AdaLoRA, DoRA, OLoRA, RS‑LoRA, MR‑LoRA‑RS are '--', only GLUE percentage is given.
- `student_per_task.csv` contains no rows for those methods, but metrics.json files in `results/kd‑lora/` contain the missing per‑task scores.

### Table 4 (efficiency)
- Memory reduction and inference speedup columns are '--' for all variants except LoRA and MR‑LoRA.
- Likely those measurements were only taken for LoRA and MR‑LoRA.

## Extracted Per‑Task Percentages (Student Distillation)
- Script `compute_student_missing.py` extracted scores from `results/kd‑lora/` and computed percentages relative to teacher FFT baseline.
- Results (averaged across BERT, RoBERTa, DeBERTa):
  - **AdaLoRA**: CoLA 12.7%, MRPC 84.0%, QQP 92.6%, QNLI 93.3%, RTE 79.3%, WNLI 101.1%
  - **DoRA**: CoLA 69.1%, MRPC 85.0%, QQP 93.8%, QNLI 95.2%, RTE 79.9%, WNLI 100.6%
  - **OLoRA**: CoLA 70.6%, MRPC 84.8%, QQP 94.2%, QNLI 95.4%, RTE 80.6%, WNLI 100.8%
  - **RS‑LoRA**: CoLA 75.4%, MRPC 84.5%, QQP 94.1%, QNLI 96.3%, RTE 81.6%, WNLI 103.1%
  - **MR‑LoRA‑RS**: CoLA 81.7%, MRPC 89.1%, QQP 96.7%, QNLI 97.3%, RTE 85.5%, WNLI 100.5%
- SST‑2 and STS‑B data missing for these variants; left as '--'.

## Actions Taken
- Updated `main.tex` Table 3 with the above percentages, preserving '--' for SST‑2 and STS‑B.
- Teacher variant STS‑B (MR‑LoRA‑RS) remains '--'.
- Efficiency table unchanged.

## Conclusion
The '--' cells that could be filled with real data have been filled. Remaining '--' cells correspond to missing measurements (SST‑2, STS‑B for student variants; STS‑B for teacher MR‑LoRA‑RS; efficiency metrics). The LaTeX tables now reflect the available experimental results.