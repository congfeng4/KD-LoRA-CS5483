# Progress Log

## 2026‑01‑28 (Previous task: Fill missing cells in MrLoRA/main.tex)
(See above.)

---

## 2026‑01‑29 (New task: Redesign experimental presentation scheme)

### Phase 1: Understand current data and presentation
**Status:** in_progress
**Actions:**
- Read existing planning files (task_plan.md, findings.md, progress.md) from previous session.
- Listed top‑level directory structure; found `results/`, `MrLoRA/`, `analysis/`, `paper/`.
- Located metrics.json files in results directory (GLUE tasks).
- Updated task_plan.md with new goal and phases.
- Updated findings.md with current state and preliminary design principles.
- This progress entry.

**Next:** Examine MrLoRA/main.tex to see exact table layouts, and search for similar papers in `paper/` directory.

## 2026‑01‑29 (Immediate task: Generate LaTeX table from results)

**Status:** in_progress
**Actions:**
- User requirements collected: rows=methods, columns=tasks, cells=primary metric mean across seeds.
- Updated task_plan.md with Phase 0.
- Updated findings.md with immediate task details.
- Created todo list.
- Explored data structure: mapped tasks to primary metrics, identified model families, seeds, methods.
- Decided separate tables per model family (bert, roberta, deberta).
- Wrote aggregation script `scripts/aggregate_metrics.py`; collected 650 data points, computed means.
- Generated pivot CSV files for each model family.
- Wrote LaTeX generation script `scripts/generate_latex.py`; produced three LaTeX tables with booktabs, captions, labels, bold best values, 3 decimal places.

**Next:** Validate tables with user and adjust formatting if needed.