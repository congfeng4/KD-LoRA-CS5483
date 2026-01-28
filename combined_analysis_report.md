# Combined Accuracy-Efficiency Analysis

## Combined Dataset
| Model Family   | Strategy   |   GLUE Score | Model              |   Parameters (M) |   Memory Usage |   Inference Time |   Score per Param (per M) |   Score per Memory (per GB) |   Score per Time (per s) |
|:---------------|:-----------|-------------:|:-------------------|-----------------:|---------------:|-----------------:|--------------------------:|----------------------------:|-------------------------:|
| bert           | FFT        |     0.781095 | BERT-base          |            110   |         1332   |             6.1  |                0.00710086 |                    0.600481 |                0.128048  |
| bert           | LoRA       |     0.77221  | BERT-base          |              2.9 |          463.5 |             6.22 |                0.266279   |                    1.70603  |                0.124149  |
| bert           | KD-LoRA    |     0.73799  | DistilBERT-base    |              1.2 |          296.8 |             5.36 |                0.614991   |                    2.54616  |                0.137685  |
| roberta        | FFT        |     0.80791  | RoBERTa-base       |            125   |         1515.9 |             7.21 |                0.00646328 |                    0.545748 |                0.112054  |
| roberta        | LoRA       |     0.790636 | RoBERTa-base       |              2.9 |          531.9 |             7.19 |                0.272633   |                    1.52211  |                0.109963  |
| roberta        | KD-LoRA    |     0.745548 | DistilRoBERTa-base |              1.5 |          358.3 |             4.44 |                0.497032   |                    2.13073  |                0.167916  |
| deberta        | FFT        |     0.84107  | DeBERTa-v3-base    |            183   |         2234.5 |            14.37 |                0.00459601 |                    0.385435 |                0.0585296 |
| deberta        | LoRA       |     0.80981  | DeBERTa-v3-base    |              2.9 |          763.4 |            15.62 |                0.279245   |                    1.08625  |                0.0518444 |
| deberta        | KD-LoRA    |     0.769511 | DeBERTa-v3-small   |              1.5 |          590.3 |            10.38 |                0.513007   |                    1.33488  |                0.074134  |

## Efficiency-Accuracy Metrics
| Model              | Strategy   |   GLUE Score |   Parameters (M) |   Score per Param (per M) |   Memory Usage |   Inference Time |
|:-------------------|:-----------|-------------:|-----------------:|--------------------------:|---------------:|-----------------:|
| BERT-base          | FFT        |     0.781095 |            110   |                0.00710086 |         1332   |             6.1  |
| BERT-base          | LoRA       |     0.77221  |              2.9 |                0.266279   |          463.5 |             6.22 |
| DistilBERT-base    | KD-LoRA    |     0.73799  |              1.2 |                0.614991   |          296.8 |             5.36 |
| RoBERTa-base       | FFT        |     0.80791  |            125   |                0.00646328 |         1515.9 |             7.21 |
| RoBERTa-base       | LoRA       |     0.790636 |              2.9 |                0.272633   |          531.9 |             7.19 |
| DistilRoBERTa-base | KD-LoRA    |     0.745548 |              1.5 |                0.497032   |          358.3 |             4.44 |
| DeBERTa-v3-base    | FFT        |     0.84107  |            183   |                0.00459601 |         2234.5 |            14.37 |
| DeBERTa-v3-base    | LoRA       |     0.80981  |              2.9 |                0.279245   |          763.4 |            15.62 |
| DeBERTa-v3-small   | KD-LoRA    |     0.769511 |              1.5 |                0.513007   |          590.3 |            10.38 |

## Key Observations
- Average GLUE score drop FFT → LoRA: 0.019
- Average GLUE score drop FFT → KD‑LoRA: 0.059
- Average GLUE score drop LoRA → KD‑LoRA: 0.040
- Parameter reduction LoRA vs FFT: 97.9%
- Parameter reduction KD‑LoRA vs FFT: 99.0%
- Inference speedup LoRA vs FFT: 0.95x
- Inference speedup KD‑LoRA vs FFT: 1.37x

## Visualizations
Plots saved to `combined_plots/`
- `scatter_params_vs_score.png`: GLUE Score vs Parameter count
- `scatter_time_vs_score.png`: GLUE Score vs Inference time
- `bar_score_per_param.png`: Score per million parameters
- `heatmap_normalized.png`: Normalized metrics heatmap

---
*Generated on 2026-01-28 09:29:14.070812*