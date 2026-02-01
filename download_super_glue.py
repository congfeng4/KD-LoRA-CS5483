import os
from datasets import load_dataset

# 1. 镜像加速（国内环境建议开启）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. SuperGLUE 子任务列表
super_glue_tasks = [
    "boolq", "cb", "copa", "multirc", "record",
    "rte", "wic", "wsc", "axb", "axg"
]


def download_and_save_super_glue(base_path='./download'):
    # 创建基础保存目录
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    print(f"开始下载并保存 SuperGLUE 到本地目录: {base_path}\n")

    for task in super_glue_tasks:
        print(f"--- 正在处理任务: {task.upper()} ---")
        save_dir = os.path.join(base_path, task)

        # 如果目录已存在且不为空，可以选择跳过
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
            print(f"任务 {task} 已存在于本地，跳过下载。")
            continue

        try:
            # 加载数据集
            dataset = load_dataset("super_glue", task)

            # 3. 核心步骤：保存到磁盘
            # 这会将所有划分 (train, validation, test) 一起保存
            dataset.save_to_disk(save_dir)

            # 打印统计信息
            splits = ", ".join([f"{k}({len(v)})" for k, v in dataset.items()])
            print(f"成功保存 {task} 到 {save_dir}")
            print(f"数据详情: {splits}")

        except Exception as e:
            print(f"处理 {task} 时出错: {e}")
        print("-" * 50)

    print("\n所有 SuperGLUE 任务已成功持久化到本地磁盘！")


if __name__ == "__main__":
    download_and_save_super_glue()
