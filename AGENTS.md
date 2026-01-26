# Agent Guidelines for KD-LoRA

This document provides guidelines for AI agents working on the KD-LoRA codebase. It covers build/lint/test commands, code style, and project conventions.

## Project Overview

KD-LoRA is a research repository implementing hybrid fine‑tuning with LoRA and knowledge distillation for large language models. The codebase is written in Python and uses PyTorch, Hugging Face Transformers, and PEFT.

## Build and Installation

### Dependencies
Install via pip with the exact versions listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Environment
- Python 3.9+ recommended (compatible with PyTorch 2.4.1 and Transformers 4.46.3)
- Use Conda environments (see `run_*.sh` scripts for examples)
- CUDA‑enabled GPU required for training

### Running Experiments
The main entry points are scripts in `src/`:

- **BERT.py** – vanilla fine‑tuning
- **BERT_LoRA.py** – LoRA fine‑tuning
- **BERT_KD_LoRA.py** – knowledge‑distilled LoRA
- **BERT_Distill_LoRA.py** – full distillation pipeline
- **BERT_MrLoRA.py** – multi‑rank LoRA

Example launch command (using accelerate for distributed training):

```bash
accelerate launch src/BERT_Distill_LoRA.py -t 2 --from_disk 1
```

Shell wrappers are provided (`run_lora.sh`, `run_mrlora.sh`, etc.) that activate the Conda environment and set `CUDA_VISIBLE_DEVICES`.

## Linting and Formatting

**No formal linting or formatting configuration is present in the repository.**  
Agents should follow the existing code style (described below). If you introduce new linting/formatting tools, please add configuration files (`.flake8`, `pyproject.toml`, `.pre‑commit‑config.yaml`) and update this section.

Suggested tools (not currently enforced):
- **Black** for code formatting (line length 88)
- **isort** for import sorting (Black‑compatible profile)
- **flake8** for static analysis (ignore E203, W503)
- **mypy** for type checking (optional)

## Testing

**No test suite is currently defined.**  
The project is research‑oriented and relies on manual validation via GLUE benchmark notebooks (`bench‑glue.ipynb`, `summarize.ipynb`).

If you add unit tests:
- Place them in a `tests/` directory mirroring the `src/` structure
- Use `pytest` as the test runner
- Follow the naming convention `test_*.py`
- Mock external dependencies (HF datasets, models) where possible

## Code Style Guidelines

### Imports
Order imports as follows, with a blank line between groups:

1. Standard library (`import os`, `import argparse`)
2. Third‑party packages (`import torch`, `from transformers import …`)
3. Local modules (`from utils import *`, `from mrlora import …`)

Use absolute imports within `src/`. Avoid relative imports (`from .utils import …`) unless necessary.

### Naming Conventions
- **Variables & functions:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private members:** prefix with `_` (e.g., `_internal_helper`)

### Docstrings
Use triple‑double‑quoted docstrings for public classes and functions. Follow the existing style: a one‑line summary, optionally followed by a longer description.

Example:
```python
def load_glue_dataset(dataset_path, task, from_disk=True):
    """
    Load a GLUE dataset from disk or Hugging Face Hub.
    """
```

### Error Handling
- Use `try`‑`except` blocks for recoverable errors (e.g., file I/O)
- Let fatal errors propagate (e.g., missing required arguments)
- Log warnings with `warnings.warn()` or print to stderr

### Type Hints
Type hints are **not consistently used** in the existing code. When adding new functions, consider adding type annotations for clarity, but do not retrofit old code unless the change is minimal.

Example of acceptable addition:
```python
def get_trainable_param_count(model: torch.nn.Module) -> float:
    …
```

### Indentation and Line Length
- Use 4 spaces per indentation level (no tabs)
- Line length is not strictly enforced; typical lines are 80‑100 characters
- Break long lines after operators, with continuation lines indented one extra level

### Strings
- Use double quotes for user‑facing strings, single quotes for internal identifiers
- Prefer f‑strings over `format()` or `%` formatting

### Configuration
Configuration is often passed via `argparse` or wrapped in an `Addict` dict (from the `addict` package). Follow the pattern in `BERT_Distill_LoRA.py`:

```python
args = Addict(kwargs)
self.args = args
```

### Logging vs. Print
The codebase uses `print()` statements for progress and debugging. Agents may continue this pattern; if introducing more structured logging, use the `logging` module with `INFO` level.

## File and Directory Structure

```
KD-LoRA/
├── src/                    # Main source code
│   ├── BERT.py
│   ├── BERT_LoRA.py
│   ├── BERT_KD_LoRA.py
│   ├── BERT_Distill_LoRA.py
│   ├── BERT_MrLoRA.py
│   ├── utils.py           # Shared utilities
│   ├── mrlora/            # Multi‑rank LoRA implementation
│   └── peft/              # Local copy of PEFT (custom modifications)
├── dataset/               # GLUE datasets (git‑ignored)
├── models/                # Downloaded model weights (git‑ignored)
├── results/               # Experiment outputs (git‑ignored)
├── requirements.txt       # Python dependencies
├── run_*.sh              # Launch scripts
└── *.ipynb               # Jupyter notebooks for analysis
```

## Git Practices

- Commit messages should be concise and describe the *why* rather than the *what*
- Do not commit files listed in `.gitignore` (datasets, models, checkpoints, `.DS_Store`, etc.)
- Use feature branches for substantial changes; merge via pull requests

## Cursor / Copilot Rules

No `.cursorrules` or `.github/copilot‑instructions.md` files are present. If you create them, link them here.

## Additional Notes

- The `peft/` directory is a local copy of the Hugging Face PEFT library with custom modifications. Edit with caution; changes may need to be upstreamed.
- The `mrlora/` directory implements a novel multi‑rank LoRA variant. Respect its architecture when making changes.
- Experiment outputs (checkpoints, logs, metrics) are saved under `results/`. Use the `--dir_name` argument to control the output location.

---

*This document is intended for AI agents assisting with the KD‑LoRA project. Update it as the project evolves.*