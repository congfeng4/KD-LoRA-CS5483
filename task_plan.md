# Task Plan: Fill missing cells in MrLoRA/main.tex

## Goal
Replace '--' placeholders in LaTeX tables with actual data from results directory.

## Phases

### Phase 1: Identify missing cells ✓
- Located all '--' cells in main.tex tables (Tables 2, 3, 4)
- Determined which missing values can be computed from existing data

### Phase 2: Gather required data ✓
- For teacher variants: STS‑B for MR‑LoRA‑RS missing (cannot fill)
- For student variants: per‑task data for AdaLoRA, DoRA, OLoRA, RS‑LoRA, MR‑LoRA‑RS exists in results/kd‑lora/
- For efficiency table: memory reduction and inference speedup data only available for LoRA and MR‑LoRA

### Phase 3: Compute missing values ✓
- Wrote script to extract per‑task scores from metrics.json files
- Computed percentages relative to teacher FFT baseline, averaged across model families
- Results obtained for CoLA, MRPC, QQP, QNLI, RTE, WNLI; SST‑2 and STS‑B missing

### Phase 4: Update LaTeX file ✓
- Edited main.tex Table 3 (student variants) with computed percentages
- Preserved '--' for SST‑2 and STS‑B
- Teacher variant STS‑B (MR‑LoRA‑RS) remains '--'
- Efficiency table unchanged

### Phase 5: Verification ✓
- Updated tables reflect real data where possible
- LaTeX file ready for compilation

## Conclusion
Task completed. The '--' cells that could be filled with real data have been filled; remaining '--' cells correspond to missing measurements.