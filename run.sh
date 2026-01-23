conda activate lora
export CUDA_VISIBLE_DEVICES=2
#python src/BERT.py --task wnli --num_train_epochs 3
python src/BERT_Distill_LoRA.py
