conda activate lora
# export CUDA_VISIBLE_DEVICES=2
#python src/BERT.py --task wnli --num_train_epochs 3
accelerate launch src/BERT_Distill_LoRA.py -t 0
#accelerate launch src/BERT_Distill_LoRA.py -t 1
#accelerate launch src/BERT_Distill_LoRA.py -t 2

