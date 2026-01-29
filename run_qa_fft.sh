conda activate lora

accelerate launch --main_process_port 2600 src/BERT_QA.py -t 0 --from_disk 1