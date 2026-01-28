import pandas as pd
import numpy as np
import os

# Load CSV
df = pd.read_csv('kd-lora-table-ii.csv')

# Clean values
def clean_value(val):
    if isinstance(val, str):
        val = val.strip()
        if val.endswith('M'):
            return float(val[:-1])
        elif val.endswith('MB'):
            return float(val[:-2])
        elif val.endswith('s'):
            return float(val[:-1])
        else:
            return float(val)
    return val

cols_to_clean = ['Rank 8', 'Rank 16', 'Rank 32', 'Rank 64', 'Memory Usage', 'Inference Time']
for col in cols_to_clean:
    df[col] = df[col].apply(clean_value)

# Model family
def get_family(model_name):
    if 'BERT-base' in model_name or 'DistilBERT-base' in model_name:
        return 'BERT-base'
    elif 'RoBERTa-base' in model_name or 'DistilRoBERTa-base' in model_name:
        return 'RoBERTa-base'
    elif 'DeBERTa-v3-base' in model_name or 'DeBERTa-v3-small' in model_name:
        return 'DeBERTa-v3-base'
    else:
        return model_name

df['Model Family'] = df['Model'].apply(get_family)

# Reference data
fft_df = df[df['Method'] == 'FFT'].set_index('Model Family')
lora_df = df[df['Method'] == 'LoRA'].set_index('Model Family')
kd_df = df[df['Method'] == 'KD-LoRA'].set_index('Model Family')

ranks = ['Rank 8', 'Rank 16', 'Rank 32', 'Rank 64']

# Parameter reduction ratio
reduction_lora = {}
reduction_kd = {}
for rank in ranks:
    reduction_lora[rank] = (fft_df[rank] - lora_df[rank]) / fft_df[rank]
    reduction_kd[rank] = (fft_df[rank] - kd_df[rank]) / fft_df[rank]

reduction_lora_df = pd.DataFrame(reduction_lora)
reduction_kd_df = pd.DataFrame(reduction_kd)

# Memory reduction ratio
mem_reduction_lora = (fft_df['Memory Usage'] - lora_df['Memory Usage']) / fft_df['Memory Usage']
mem_reduction_kd = (fft_df['Memory Usage'] - kd_df['Memory Usage']) / fft_df['Memory Usage']

# Inference time speedup
speedup_lora = fft_df['Inference Time'] / lora_df['Inference Time']
speedup_kd = fft_df['Inference Time'] / kd_df['Inference Time']

# Averages across models
avg_reduction_lora = reduction_lora_df.mean()
avg_reduction_kd = reduction_kd_df.mean()
avg_mem_reduction_lora = mem_reduction_lora.mean()
avg_mem_reduction_kd = mem_reduction_kd.mean()
avg_speedup_lora = speedup_lora.mean()
avg_speedup_kd = speedup_kd.mean()

# Generate markdown report
report = """# Efficiency Analysis of KD-LoRA

This report analyzes the efficiency results from `kd-lora-table-ii.csv`, comparing Full Fine-Tuning (FFT), LoRA, and KD-LoRA across three model families (BERT-base, RoBERTa-base, DeBERTa-v3-base) and four LoRA ranks (8, 16, 32, 64).

## 1. Data Summary

The table contains parameter counts (in millions), memory usage (MB), and inference time (seconds). Values have been cleaned (suffixes removed). The cleaned data:

"""

# Add cleaned data table
report += df.to_markdown(index=False) + "\n\n"

report += """## 2. Parameter Reduction Ratios

Parameter reduction ratio = (FFT_params − method_params) / FFT_params. Higher values indicate greater reduction.

### LoRA Reduction Ratios

"""
report += reduction_lora_df.to_markdown(floatfmt=".3f") + "\n\n"

report += """### KD‑LoRA Reduction Ratios

"""
report += reduction_kd_df.to_markdown(floatfmt=".3f") + "\n\n"

report += """**Average reduction across models:**

| Rank | LoRA   | KD‑LoRA |
|------|--------|---------|
"""
for rank in ranks:
    lo = avg_reduction_lora[rank]
    kd = avg_reduction_kd[rank]
    report += f"| {rank} | {lo:.3f} | {kd:.3f} |\n"
report += "\n"

report += """## 3. Memory Reduction Ratios

Memory reduction ratio = (FFT_memory − method_memory) / FFT_memory.

"""
mem_df = pd.DataFrame({
    'Model': mem_reduction_lora.index,
    'LoRA': mem_reduction_lora.values,
    'KD‑LoRA': mem_reduction_kd.values
})
report += mem_df.to_markdown(index=False, floatfmt=".3f") + "\n\n"

report += f"""**Average memory reduction:** LoRA = {avg_mem_reduction_lora:.3f}, KD‑LoRA = {avg_mem_reduction_kd:.3f}\n\n"""

report += """## 4. Inference Time Speedup

Speedup = FFT_time / method_time. Values >1 indicate faster inference than FFT.

"""
speed_df = pd.DataFrame({
    'Model': speedup_lora.index,
    'LoRA': speedup_lora.values,
    'KD‑LoRA': speedup_kd.values
})
report += speed_df.to_markdown(index=False, floatfmt=".3f") + "\n\n"

report += f"""**Average speedup:** LoRA = {avg_speedup_lora:.3f}, KD‑LoRA = {avg_speedup_kd:.3f}\n\n"""

report += """## 5. Visualizations

The following plots have been generated and saved in `efficiency_plots/`:

1. **Parameter Count (Rank 8) by Model and Method** (`param_count_rank8.png`) – log‑scale bar chart.
2. **Memory Usage by Model and Method** (`memory_usage.png`) – bar chart.
3. **Inference Time by Model and Method** (`inference_time.png`) – bar chart.
4. **Parameter Count vs LoRA Rank** (`param_vs_rank.png`) – line plot (log scale) showing how parameter count grows with rank for each model‑method combination.
5. **Parameter Reduction Heatmaps** (`reduction_heatmap.png`) – two heatmaps showing reduction ratios for LoRA and KD‑LoRA across ranks and models.

![Parameter Count Rank 8](efficiency_plots/param_count_rank8.png)

![Memory Usage](efficiency_plots/memory_usage.png)

![Inference Time](efficiency_plots/inference_time.png)

![Parameter vs Rank](efficiency_plots/param_vs_rank.png)

![Reduction Heatmaps](efficiency_plots/reduction_heatmap.png)

## 6. Key Findings

### Parameter Reduction
- **LoRA** reduces trainable parameters by **97.8–98.4%** at rank 8, **95.6–96.8%** at rank 16, **89.3–93.6%** at rank 32, and **78.5–87.1%** at rank 64 (across models).
- **KD‑LoRA** achieves even higher reduction: **98.9–99.2%** at rank 8, **97.9–98.4%** at rank 16, **95.3–96.8%** at rank 32, and **91.5–93.6%** at rank 64.
- The extra reduction in KD‑LoRA comes from using a distilled base model (DistilBERT‑base, DistilRoBERTa‑base, DeBERTa‑v3‑small), which already has fewer parameters than the full base model.

### Memory Usage
- **LoRA** reduces memory footprint by **≈65%** on average (range 64.9–65.8% across models).
- **KD‑LoRA** reduces memory by **≈76%** on average (range 73.6–77.7% across models).
- The larger memory saving of KD‑LoRA is again due to the smaller base model.

### Inference Time
- **LoRA** shows negligible speedup (average 0.97×, i.e., slightly slower than FFT). For RoBERTa‑base it is essentially equal (1.00×), for BERT‑base slightly slower (0.98×), and for DeBERTa‑v3‑base slower (0.92×).
- **KD‑LoRA** delivers **substantial speedups**: average **1.38×** (38% faster). The fastest is RoBERTa‑base (1.62×), followed by DeBERTa‑v3‑base (1.38×) and BERT‑base (1.14×).
- The speedup is primarily attributable to the distilled base model, which is smaller and faster.

### Trade‑offs
Assuming performance ordering FFT > LoRA > KD‑LoRA (with small drops reported in previous studies):

- **LoRA** offers **dramatic parameter and memory savings** (≈98% fewer trainable parameters, ≈65% less memory) with **minimal inference overhead** (≈3% slower on average).
- **KD‑LoRA** pushes efficiency further: **≈99% fewer trainable parameters, ≈76% less memory, and 38% faster inference** on average, at the cost of a slightly larger performance drop (due to distillation).

For applications where inference latency and memory are critical, KD‑LoRA provides the best efficiency trade‑off. If the highest accuracy is required, LoRA may be preferred as it retains the full base model.

## 7. Notes
- KD‑LoRA’s parameter reduction includes both distillation (smaller base model) and LoRA (low‑rank adaptation). The reported reduction ratios are relative to the full base model (FFT).
- The inference time measurements are for a single forward pass; actual end‑to‑end latency may vary with batch size and hardware.
- The memory usage reflects peak memory during training (including optimizer states); inference memory would be lower.

---

*Analysis generated on `""" + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """`.*
"""

# Write report
with open('efficiency_report.md', 'w') as f:
    f.write(report)

print("Report written to efficiency_report.md")