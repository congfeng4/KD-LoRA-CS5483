# Progress Log

## 2026‑01‑28

### Phase 1: Identify missing cells
- Located all '--' cells in main.tex:
  - Table 2 (teacher variants): STS‑B for MR‑LoRA‑RS.
  - Table 3 (student variants): all per‑task columns for AdaLoRA, DoRA, OLoRA, RS‑LoRA, MR‑LoRA‑RS.
  - Table 4 (efficiency): memory reduction and inference speedup for all variants except LoRA and MR‑LoRA.
- Verified that generated percentage tables (`teacher_per_task_pct.tex`, `student_per_task_pct.tex`) already match the LaTeX tables.

### Phase 2: Gather required data
- Examined CSV files; confirmed missing raw data for STS‑B (MR‑LoRA‑RS) and per‑task student data for variants other than LoRA/MR‑LoRA.
- Found that metrics.json files in `results/kd‑lora/` contain student distillation results for AdaLoRA, DoRA, OLoRA, RS‑LoRA, MR‑LoRA‑RS.

### Phase 3: Compute missing values
- Wrote script `compute_student_missing.py` to extract per‑task scores and compute percentages relative to teacher FFT baseline.
- Computed percentages for CoLA, MRPC, QQP, QNLI, RTE, WNLI (SST‑2 and STS‑B data missing).
- Results:
  - AdaLoRA: CoLA 12.7%, MRPC 84.0%, QQP 92.6%, QNLI 93.3%, RTE 79.3%, WNLI 101.1%
  - DoRA: CoLA 69.1%, MRPC 85.0%, QQP 93.8%, QNLI 95.2%, RTE 79.9%, WNLI 100.6%
  - OLoRA: CoLA 70.6%, MRPC 84.8%, QQP 94.2%, QNLI 95.4%, RTE 80.6%, WNLI 100.8%
  - RS‑LoRA: CoLA 75.4%, MRPC 84.5%, QQP 94.1%, QNLI 96.3%, RTE 81.6%, WNLI 103.1%
  - MR‑LoRA‑RS: CoLA 81.7%, MRPC 89.1%, QQP 96.7%, QNLI 97.3%, RTE 85.5%, WNLI 100.5%

### Phase 4: Update LaTeX file
- Edited `main.tex` Table 3 (student variants) to fill the available percentages, leaving SST‑2 and STS‑B as '--'.
- Teacher variant STS‑B (MR‑LoRA‑RS) remains '--' (data not available).
- Efficiency table unchanged (no additional measurements).

### Phase 5: Verification
- Updated tables now reflect real data where possible.
- The LaTeX file should compile without issues.