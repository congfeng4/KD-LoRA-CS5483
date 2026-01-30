conda activate lora
export WANDB_DISABLED=true
accelerate launch --main_process_port 2608 src/BERT_QA.py -t 3 --from_disk 1