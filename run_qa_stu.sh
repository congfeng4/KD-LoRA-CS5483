conda activate lora

accelerate launch --main_process_port 2602 src/BERT_QA.py -t 1 --from_disk 1