conda activate lora
accelerate launch --main_process_port 2505 src/run_adalora.py --from_disk 1 "$@"