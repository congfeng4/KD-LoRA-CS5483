# Task Plan: Redesign experimental presentation scheme for KD-LoRA results

## Goal
Design a clear, comprehensive presentation scheme for experimental results across multiple dimensions:
- Benchmarks: GLUE (existing), SQuAD (future)
- Tasks per benchmark (e.g., CoLA, MRPC, QQP, QNLI, RTE, WNLI, SST‑2, STS‑B for GLUE; v1.1, v2.0 for SQuAD)
- Training variants: Full Fine‑Tuning (FFT), LoRA (teacher), Student/kd‑lora (distilled)
- LoRA variants: LoRA, MR‑LoRA, AdaLoRA, DoRA, OLoRA, RS‑LoRA, MR‑LoRA‑RS
- Model families: BERT, RoBERTa, DeBERTa‑v3
- Metrics: task‑specific (accuracy, Matthews correlation, Spearman, etc.), parameter efficiency, memory reduction, inference speedup

Primary objective: Showcase that MR‑LoRA is a more parameter‑efficient way to spend parameter budget compared to enlarging a singular LoRA matrix, while maintaining most of FFT performance.

Constraints:
- LoRA and Student training settings cannot be averaged together (different architectures)
- Presentation medium is tables (LaTeX) suitable for academic paper
- Need to highlight per‑task performance as well as task‑average across all tasks and LoRA variants
- Should accommodate future SQuAD results

## Phases

### Phase 0: Generate LaTeX table from existing results (immediate request)
- Create LaTeX table with rows=methods, columns=tasks, cells=primary metric mean across seeds
- Use all tasks and methods present in `results/lora/` directory
- Deliver table.tex file for user review

### Phase 1: Understand current data and presentation
- Inventory existing results (GLUE tasks, model families, LoRA variants, metrics)
- Examine current LaTeX tables in MrLoRA/ to understand current presentation style
- Identify gaps (e.g., missing SQuAD, missing efficiency metrics for some variants)

### Phase 2: Define presentation requirements and design principles
- Determine what comparisons are most important (parameter efficiency vs. performance trade‑off)
- Decide on table granularity (per‑benchmark, per‑model‑family, aggregated across families)
- Design separate tables for teacher variants and student variants (cannot mix)
- Plan how to incorporate parameter counts, memory, speedup

### Phase 3: Propose concrete table structure(s)
- Sketch table layouts (number of tables, rows, columns)
- Define how to present per‑task scores vs. averages
- Integrate efficiency metrics (parameter budget, memory reduction, inference speedup)
- Consider visual aids (e.g., bold best results, color coding) within LaTeX constraints

### Phase 4: Validate with existing data
- Create mock‑up tables using existing GLUE data
- Ensure all relevant dimensions are captured clearly
- Check for readability and information density

### Phase 5: Produce LaTeX template and documentation
- Write LaTeX code for the proposed table(s)
- Provide instructions for filling with future SQuAD results
- Update AGENTS.md with new presentation conventions

## Progress Log
- 2026-01-29: Phase 0 added for immediate LaTeX table generation
- 2026-01-29: User requirements collected: rows=methods, columns=tasks, cells=primary metric mean across seeds

## Conclusion
Deliver a complete presentation scheme that can be used for the paper and future experiments.