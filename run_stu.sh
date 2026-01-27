conda activate lora

accelerate launch --main_process_port 0 src/BERT_Distill_LoRA.py -t 1 --from_disk 1
