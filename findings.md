# Findings

## Previous Task: Fill missing cells in MrLoRA/main.tex
(See above for detailed findings from that task.)

---

## New Task: Redesign experimental presentation scheme

### Current State (as of 2026‑01‑29)
**Data dimensions present in experiments:**
1. **Benchmarks**: GLUE (8 tasks: CoLA, MRPC, QQP, QNLI, RTE, WNLI, SST‑2, STS‑B)
2. **Model families**: BERT, RoBERTa, DeBERTa‑v3
3. **Training variants**:
   - Teacher fine‑tuning (`type=2`): FFT (full fine‑tuning), LoRA (standard LoRA), MR‑LoRA, AdaLoRA, DoRA, OLoRA, RS‑LoRA, MR‑LoRA‑RS
   - Student distillation (`type=1`): same set of LoRA variants applied to a smaller student model
4. **Metrics recorded**:
   - Task‑specific: accuracy (for QQP, QNLI, RTE, WNLI, SST‑2), Matthews correlation (CoLA), F1 (MRPC), Spearman correlation (STS‑B)
   - Aggregated: GLUE average (arithmetic mean of per‑task scores normalized to 0‑100 scale)
   - Efficiency: parameter count, memory reduction, inference speedup (only available for LoRA and MR‑LoRA)

**Current presentation (MrLoRA/main.tex):**
- Table 2: Teacher variants – per‑task scores as percentages of teacher FFT baseline, averaged across model families
- Table 3: Student variants – same layout as Table 2
- Table 4: Efficiency – memory reduction and inference speedup for LoRA and MR‑LoRA only
- Table 1 (?) – not yet examined

**Observations:**
- Per‑task scores are shown as percentages relative to teacher FFT baseline (averaged across BERT/RoBERTa/DeBERTa).
- The tables separate teacher and student variants, respecting the architectural difference.
- Missing efficiency data for most LoRA variants.
- SQuAD benchmark not yet run (no data).

**Key design challenge:** Need to present multi‑dimensional results in a digestible tabular form that highlights:
1. MR‑LoRA’s parameter efficiency compared to enlarging a singular LoRA matrix
2. Performance retention relative to FFT baseline
3. Comparison across model families (generality)
4. Trade‑offs between teacher and student distillation

### Design Principles (preliminary)
1. **Separate tables for teacher and student variants** – architectures differ, cannot average together.
2. **Show per‑task scores alongside averages** – readers need both detail and summary.
3. **Include parameter counts** – essential for parameter‑efficiency argument.
4. **Highlight MR‑LoRA vs. other LoRA variants** – make comparison easy.
5. **Aggregate across model families** – show generality while optionally providing per‑family breakdown in appendix.
6. **Plan for SQuAD inclusion** – table structure should accommodate another benchmark.

### Potential Table Structures
**Option A: Two‑tier presentation**
- **Top‑level table:** GLUE average (%) for all LoRA variants (teacher and student separate), plus parameter counts, memory, speedup.
- **Appendix tables:** Full per‑task breakdown for each variant.

**Option B: Integrated per‑task table**
- One large table per benchmark (GLUE, SQuAD) with rows = (variant × model family), columns = tasks + average + efficiency metrics.
- Could be overwhelming but comprehensive.

**Option C: Focused comparison tables**
- Table 1: Teacher variants – GLUE average vs. parameter count (scatter plot data)
- Table 2: Student variants – same
- Table 3: Per‑task performance of MR‑LoRA vs. best competing variant
- Table 4: Efficiency metrics for all variants where available.

**Option D: Hierarchical tables**
- Table 1: Teacher GLUE results (per‑task scores as percentages, averaged across families)
- Table 2: Student GLUE results (same)
- Table 3: Teacher SQuAD results (future)
- Table 4: Student SQuAD results (future)
- Table 5: Efficiency (parameter counts, memory, speedup) for all variants.

**Need to examine typical NLP paper conventions** – look at existing LoRA papers for inspiration.

### Immediate Task: Generate LaTeX table from results
**User request:** Create LaTeX table like referenced image, using data in `results/`. Rows = methods, columns = tasks, cells = primary metric mean across seeds. Include all tasks and methods present.

**Data exploration:**
- Metrics.json contains `eval_matthews_correlation` for CoLA, `eval_accuracy` for MNLI, etc.
- Need to map each task to its primary metric.
- Methods: lora, dora, mrlora, mrlora-rs, olora, rslora
- Tasks: cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli
- Seeds: 42, 123, 2024 (some missing combinations)
- Model families: bert, roberta, deberta (separate tables per family)

**Task-to-metric mapping:**
- cola: eval_matthews_correlation
- mnli: matched_accuracy
- mrpc: eval_f1
- qnli: eval_accuracy
- qqp: eval_f1
- rte: eval_accuracy
- sst2: eval_accuracy
- stsb: eval_spearman
- wnli: eval_accuracy

**Plan:**
1. Write Python script to traverse results, extract metric based on mapping.
2. Compute mean across seeds per (task, method, model_family).
3. Generate three LaTeX tables (one per model family) with booktabs style.
4. Format cells with 3 decimal places (e.g., 0.xxx).

### Next Steps
- Examine MrLoRA/main.tex to see exact current table layouts.
- Search for similar papers in the directory (maybe `paper/`).
- Inventory which efficiency metrics are actually available.
- Draft concrete table schemas.