from datasets import load_dataset

glue_tasks = [
    "cola", "sst2", "mrpc", "qqp", "stsb",
    "mnli", "qnli", "rte", "wnli", "ax"
]

for task in glue_tasks:
    dataset = load_dataset('glue', task, cache_dir='./dataset')
    print(task, len(dataset))
    dataset.save_to_disk(f'./glue-dataset/{task}')
