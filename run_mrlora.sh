conda activate lora

accelerate launch --main_process_port 2505 src/BERT_MrLoRA.py --ours
