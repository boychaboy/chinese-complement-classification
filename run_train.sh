#!/bin/bash

if [[ $# -eq 2 ]]
then
	model_name=$1
	gpu_id=$2
else 
	echo "Run format > ./run_inference.sh {model_dir} {gpu_id}"
	exit 1
fi

epochs=5
lr=5e-5
seed=42
train_batch=200
eval_batch=400
checkpoint=100

CUDA_VISIBLE_DEVICES=$gpu_id python3 src/main.py \
	--model bert --epochs $epochs \
	--lr $lr --seed $seed \
	--train_batch_size $train_batch \
	--eval_batch_size $eval_batch \
	--checkpoint $checkpoint \
	--save_dir "outputs/$model_name/$model_name.tar" \
	--model_name_or_path hfl/chinese-bert-wwm-ext \
	--train_data "data/train.json" \
	--val_data "data/val.json"

