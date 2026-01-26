import json
import os
import sys
import glob
from pathlib import Path

results_root = "results"
# mapping from task to primary metric key
task_metric = {
    "cola": "eval_matthews_correlation",
    "mnli": "matched_accuracy",
    "mrpc": "eval_accuracy",  # could also use eval_f1
    "qnli": "eval_accuracy",
    "qqp": "eval_accuracy",   # could use eval_f1
    "rte": "eval_accuracy",
    "sst2": "eval_accuracy",
    "stsb": "eval_pearson",
    "wnli": "eval_accuracy",
}

def extract_metrics(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    # parse path components
    parts = Path(filepath).parts
    # variant is the directory under results (fft, lora, kd-lora)
    variant = parts[parts.index("results") + 1]
    # task directory name like task_cola_bert_42
    task_dir = parts[parts.index("results") + 2]
    # extract task, model, seed
    _, task, model, seed = task_dir.split('_')
    # peft directory name like peft_lora_16_0.05_8
    peft_dir = parts[-2]
    peft = peft_dir.split('_')[1]  # e.g., lora, mrlora
    # metric selection
    metric_key = task_metric.get(task)
    if metric_key is None:
        # fallback: find any key starting with eval_ (excluding eval_loss, eval_runtime, etc.)
        for key in data.keys():
            if key.startswith("eval_") and key not in ("eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"):
                metric_key = key
                break
        if metric_key is None:
            # try matched_accuracy or mismatched_accuracy
            if "matched_accuracy" in data:
                metric_key = "matched_accuracy"
            elif "mismatched_accuracy" in data:
                metric_key = "mismatched_accuracy"
            else:
                metric_key = ""
    metric_value = data.get(metric_key, "")
    # trainable params count
    trainable_params = data.get("train", {}).get("trainable_params_count", "")
    train_time = data.get("train", {}).get("train_time", "")
    # args
    args = data.get("args", {})
    rank = args.get("rank", "")
    lora_alpha = args.get("lora_alpha", "")
    lora_dropout = args.get("lora_dropout", "")
    lora_ranks = args.get("lora_ranks", "")
    peft_type = args.get("peft", "")
    type_num = args.get("type", "")
    model_family = args.get("model_family", "")
    
    return {
        "variant": variant,
        "task": task,
        "model": model,
        "seed": seed,
        "peft": peft,
        "peft_type": peft_type,
        "type": type_num,
        "metric_key": metric_key,
        "metric_value": metric_value,
        "trainable_params_count": trainable_params,
        "train_time": train_time,
        "rank": rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_ranks": lora_ranks,
        "model_family": model_family,
        "file": filepath,
    }

def main():
    # find all metrics.json files for BERT model
    pattern = os.path.join(results_root, "*", "task_*_bert_*", "base_*", "peft_*", "metrics.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} files")
    # CSV header
    header = ["variant", "task", "seed", "peft", "metric_key", "metric_value", "trainable_params_count", "train_time", "rank", "lora_alpha", "lora_dropout", "lora_ranks", "type", "model_family"]
    print(",".join(header))
    for f in sorted(files):
        try:
            row = extract_metrics(f)
            # filter for BERT model only (should already be filtered by glob)
            if row["model"] != "bert":
                continue
            # output row
            values = [str(row.get(col, "")) for col in header]
            print(",".join(values))
        except Exception as e:
            sys.stderr.write(f"Error processing {f}: {e}\n")
            continue

if __name__ == "__main__":
    main()
