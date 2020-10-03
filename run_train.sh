#!/bin/bash

epochs=3
lr=5e-3
seed=42
train_batch=200
eval_batch=64
gpu_id=0
checkpoint=300

CUDA_VISIBLE_DEVICES=$gpu_id python3 src/main.py \
	--model bert --epochs $epochs \
	--lr $lr --seed $seed \
	--train_batch_size $train_batch \
	--eval_batch_size $eval_batch \
	--checkpoint $checkpoint \
	--save_dir "./save/second" \
	--data_dir "data/week3_data/"

