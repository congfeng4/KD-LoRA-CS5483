conda activate lora

accelerate launch --main_process_port 2506 src/BERT_MrLoRA.py --ours
