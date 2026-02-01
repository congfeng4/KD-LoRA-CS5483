conda activate lora

accelerate launch --main_process_port 2502 src/BERT_Distill_LoRA.py -t 1
