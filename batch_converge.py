# %%
TASK_METRIC = {
    "cola": ["eval_matthews_correlation"],
    "mnli": ["matched_accuracy", "mismatched_accuracy"],
    "mrpc": ["eval_accuracy", "eval_f1"],
    "qnli": ["eval_accuracy"],
    "qqp": ["eval_accuracy", "eval_f1"],
    "rte": ["eval_accuracy"],
    "sst2": ["eval_accuracy"],
    "stsb": ["eval_pearson", "eval_spearman"],
    "wnli": ["eval_accuracy"],
}

METRIC_NAME_MAP = {
    'eval_matthews_correlation': 'Mcc',
    'matched_accuracy': 'm',
    'mismatched_accuracy': 'mm',
    'eval_accuracy': 'Acc',
    'eval_f1': 'F1',
    'eval_pearson': 'Corr_p',
    'eval_spearman': 'Corr_s',
}

TASK_NAME_MAP = {
    'mnli': 'MNLI',
    'sst2': 'SST-2',
    'cola': 'CoLA',
    'qqp': 'QQP',
    'qnli': 'QNLI',
    'rte': 'RTE',
    'mrpc': 'MRPC',
    'stsb': 'STS-B',
}

FAMILY_NAME_MAP = {
    'bert': 'BERT-b',
    'roberta': 'RoB-b',
    'deberta': 'DeB-b',
}

METHOD_NAME_MAP = {
    'lora': 'LoRA',
    'olora': 'OLoRA',
    'dora': 'DoRA',
    'mrlora': 'MR-LoRA',
    'adalora': 'AdaLoRA',
    'mrlora-rs': 'MR-LoRA-RS',
    'rslora': 'RS-LoRA'
}
VARIANT_NAME_MAP = {
    'fft': 'FFT',
    'lora': 'LoRA-Finetuning',
    'kd-lora': 'KD-LoRA-Finetuning'
}

REMOVE_PEFT = ['mrlora-rs']

# %%
import json
import pandas as pd
import numpy as np
from pathlib import Path
from dictor import dictor
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import  NA

def extract_experiment_data(json_file):
    # variant = Path(json_file).relative_to('./converge').parts[0]

    with open(json_file, 'r') as f:
        data = json.load(f)

    # data['variant'] = variant
    # with open(json_file, 'w') as f:
    #     json.dump(data, f, indent=4)

    # Extract metadata
    model_family = dictor(data, 'args.model_family')
    peft_method = dictor(data, 'args.peft')
    print(peft_method)
    task = dictor(data, 'args.task')

    # Get training-specific metrics
    trainable_params = dictor(data, 'train.trainable_params_count', NA)
    train_runtime = dictor(data, 'train.train_time', NA)

    # Calculate Average GPU Memory (Allocated)
    memory_list = dictor(data, 'train.memory_allocated', [])
    avg_memory = np.mean(memory_list) if memory_list else NA

    rank = dictor(data, 'args.rank')
    
    # Get metrics
    # Some tasks use eval_accuracy, others eval_matthews_correlation
    for key in TASK_METRIC[task]:

        eval_runtime_history = []
        for item in data['log_history']:
            if key in item:
                eval_runtime_history.append(round(item[key], 2))

        if eval_runtime_history:
            yield {
                "family": model_family,
                "peft": peft_method,
                "task": task,
                "variant": data['variant'],
                "history": eval_runtime_history,
                "value": data[key], # Final metric value.
                "metric": key,
                "rank": rank, # total rank.
                'seed': dictor(data, 'args.seed'),
            }


def aggregate_experiment_results(root_dir):
    """
    Finds all .json files under a directory recursively, extracts data,
    and concatenates them into one large DataFrame.
    """
    root_path = Path(root_dir)
    # Recursively find all JSON files
    json_files = list(root_path.rglob("metrics.json"))

    if not json_files:
        print(f"No JSON files found in {root_dir}")
        return pd.DataFrame()

    all_dfs = []
    for f in json_files:
        try:
            rows = extract_experiment_data(f)
            all_dfs.extend(rows)
        except Exception as e:
            print(f"Failed to extract data from {f}")
            raise e

    if not all_dfs:
        print("No valid data extracted from found files.")
        return pd.DataFrame()

    # Concatenate all individual DataFrames by row
    final_df = pd.DataFrame.from_records(all_dfs)

    return final_df

df = aggregate_experiment_results('./results/')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


# --- Keep your existing MAP dictionaries and extraction logic ---

def get_method_name(row):
    """
    Refined to match the labels in your target image: FFT, LoRA, KD-LoRA
    """
    v = str(row['variant']).lower()
    if v == 'fft':
        return 'FFT'
    elif v == 'lora':
        return 'LoRA'
    elif v == 'kd-lora':
        return 'KD-LoRA'
    return v


def plot_converge_function(df, TASK, MODEL_FAMILY, SEED, METRIC, PEFT):
    # 1. Process Data for Plotting
    plot_rows = []
    for _, row in df.iterrows():
        # Ensure we only plot the specific variants we care about
        if row['Method'] not in ['FFT', 'LoRA', 'KD-LoRA']:
            continue

        for i, val in enumerate(row['history']):
            plot_rows.append({
                'Method': row['Method'],
                'Steps': i*200,
                'Metric': val
            })

    if not plot_rows:
        return

    history_df = pd.DataFrame(plot_rows)

    # 2. Styling (Matching the reference image)
    sns.set_theme(style="white")  # Reference has a plain white background
    plt.figure(figsize=(10, 6))

    # Matching the specific colors from your image
    # Blue: FFT, Orange: LoRA, Green: KD-LoRA
    palette = {
        'FFT': '#377eb8',
        'LoRA': '#ff7f00',
        'KD-LoRA': '#4daf4a'
    }
    target_order = ['FFT', 'LoRA', 'KD-LoRA']

    # 3. Plot Curves
    ax = sns.lineplot(
        data=history_df,
        x='Steps',
        y='Metric',
        hue='Method',
        palette=palette,
        hue_order=target_order,
        linewidth=4  # Thicker lines like the image
    )

    # 4. Add Baseline Horizontal Lines (Dashed)
    # We grab the final 'value' for each method in this specific slice
    for method in target_order:
        method_final = df[df['Method'] == method]
        if not method_final.empty:
            final_val = method_final.iloc[0]['value']
            plt.axhline(y=final_val, color=palette[method],
                        linestyle='--', linewidth=1.5, alpha=0.8)

    # 5. Professional Formatting
    task_name = TASK_NAME_MAP.get(TASK, TASK).upper()
    # Handle family name (e.g., "BERT/DistilBERT" from your image title)
    family_display = FAMILY_NAME_MAP.get(MODEL_FAMILY, MODEL_FAMILY)

    plt.title(f'Fine-Tuning {family_display} on {task_name}', fontsize=24, pad=20)
    plt.xlabel('Steps', fontsize=20)

    # Map metric key to readable label
    metric_label = METRIC_NAME_MAP.get(df.iloc[0]['metric'], 'Score')
    if df.iloc[0]['metric'] == 'eval_matthews_correlation':
        metric_label = "Matthews correlation"

    plt.ylabel(metric_label, fontsize=20)

    # Grid and Ticks
    plt.grid(True, axis='y', linestyle='-', alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Legend - Large and at bottom right as requested
    plt.legend(prop={'size': 20}, loc='lower right', frameon=True)

    plt.tight_layout()

    # 6. Save
    output_dir = Path('./converge')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f'{MODEL_FAMILY}-{TASK}-{METRIC}-{PEFT}-seed{SEED}.png', dpi=300)
    plt.close()  # Close to free up memory during batch processing


# --- Execution ---
# Ensure 'Method' is assigned before grouping
df['Method'] = df.apply(get_method_name, axis=1)

# Group by the experimental setup and generate plots
for (task, family, seed, metric, peft), sub_df in df.groupby(['task', 'family', 'seed', 'metric', 'peft']):
    print(f"Generating plot for {family} | {task} | Seed: {seed}")
    print(sub_df)
    plot_converge_function(sub_df, task, family, seed, metric, peft)
