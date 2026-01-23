import os
from datasets import load_dataset

# 1. 如果你在国内，建议开启镜像加速，确保下载稳定
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. GLUE 榜单包含的所有子任务名
# 注意：mnli 包含 matched 和 mismatched 两个验证集
glue_tasks = [
    "cola", "sst2", "mrpc", "qqp", "stsb",
    "mnli", "qnli", "rte", "wnli", "ax"
]


def download_all_glue():
    print(f"开始下载 GLUE 数据集，共 {len(glue_tasks)} 个任务...\n")

    for task in glue_tasks:
        print(f"--- 正在下载任务: {task.upper()} ---")
        try:
            # load_dataset 会自动处理下载、解压和缓存
            # 默认路径：~/.cache/huggingface/datasets
            dataset = load_dataset("glue", task, cache_dir='./dataset')
            print(f"成功加载 {task}，训练集样本数: {len(dataset['train']) if 'train' in dataset else 'N/A'}")
        except Exception as e:
            print(f"下载 {task} 时出错: {e}")
        print("-" * 30)

    print("\n所有任务下载尝试已完成！")


if __name__ == "__main__":
    download_all_glue()

# (py312) fengcong@MacBook-Air-68 KD-LoRA % python download_glue.py
# 开始下载 GLUE 数据集，共 10 个任务...

# --- 正在下载任务: COLA ---
# 成功加载 cola，训练集样本数: 8551
# ------------------------------
# --- 正在下载任务: SST2 ---
# 成功加载 sst2，训练集样本数: 67349
# ------------------------------
# --- 正在下载任务: MRPC ---
# 成功加载 mrpc，训练集样本数: 3668
# ------------------------------
# --- 正在下载任务: QQP ---
# 成功加载 qqp，训练集样本数: 363846
# ------------------------------
# --- 正在下载任务: STSB ---
# 成功加载 stsb，训练集样本数: 5749
# ------------------------------
# --- 正在下载任务: MNLI ---
# 成功加载 mnli，训练集样本数: 392702
# ------------------------------
# --- 正在下载任务: QNLI ---
# 成功加载 qnli，训练集样本数: 104743
# ------------------------------
# --- 正在下载任务: RTE ---
# 成功加载 rte，训练集样本数: 2490
# ------------------------------
# --- 正在下载任务: WNLI ---
# 成功加载 wnli，训练集样本数: 635
# ------------------------------
# --- 正在下载任务: AX ---
# 成功加载 ax，训练集样本数: N/A
# ------------------------------

# 所有任务下载尝试已完成！
