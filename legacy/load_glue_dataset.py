from datasets import load_from_disk

glue_tasks = [
    "cola", "sst2", "mrpc", "qqp", "stsb",
    "mnli", "qnli", "rte", "wnli", "ax"
]

for task in glue_tasks:
    dataset = load_from_disk('./glue-dataset/{}'.format(task))
    print(task, dataset)
