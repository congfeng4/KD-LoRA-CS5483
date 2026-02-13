import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

TASK_METRIC = {
    "cola": ["eval_matthews_correlation"],
    "mnli": ["eval_accuracy"],
    "mrpc": ["eval_accuracy", "eval_f1"],
    "qnli": ["eval_accuracy"],
    "qqp": ["eval_accuracy", "eval_f1"],
    "rte": ["eval_accuracy"],
    "sst2": ["eval_accuracy"],
    "stsb": ["eval_pearson", "eval_spearman"],
    "wnli": ["eval_accuracy"],
}

def plot_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    log_history = data.get("log_history", [])
    args = data.get("args", {})
    task = args.get('task')
    # Extract data for plotting
    train_steps, train_loss = [], []
    eval_steps, eval_loss, eval_matthews = [], [], defaultdict(list)

    for entry in log_history:
        step = entry.get("step")
        if "loss" in entry:  # Training log
            train_steps.append(step)
            train_loss.append(entry["loss"])
        elif "eval_loss" in entry:  # Evaluation log
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])
            for metric in TASK_METRIC[task]:
                eval_matthews[metric].append(entry.get(metric))

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Losses
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    lns1 = ax1.plot(train_steps, train_loss, color='tab:blue', label='Train Loss', marker='.')
    lns2 = ax1.plot(eval_steps, eval_loss, color='tab:orange', label='Eval Loss', marker='x')
    
    # Plot Evaluation Metric (Secondary Y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Metric Value')
    linestyle = ['--', ':']
    lns4 = []
    for i, (name, values) in enumerate(eval_matthews.items()):
        lns3 = ax2.plot(eval_steps, values, color='tab:green', label=name, linestyle=linestyle[i])
        lns4.extend(lns3)

    # Combine legends
    lns = lns1 + lns2 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')

    # Formatting title with metadata
    title = (f"Task: {args.get('task')} | Variant: {data.get('variant')} | "
             f"PEFT: {args.get('peft')} | Seed: {args.get('seed')}")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Save and close
    save_path = json_path.parent / "log_history.jpg"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {save_path}")

# Execute rglob search
if __name__ == "__main__":
    # Change "." to your results root directory if needed
    for metrics_file in Path("./ablation3").rglob("metrics.json"):
        try:
            plot_metrics(metrics_file)
        except Exception as e:
            print(f"Failed to process {metrics_file}: {e}")