conda activate lora

accelerate launch --main_process_port 29501 src/BERT_Distill_LoRA.py -t 1 --from_disk 1
