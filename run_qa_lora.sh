conda activate lora

accelerate launch --main_process_port 2605 src/BERT_QA.py -t 2 --from_disk 1