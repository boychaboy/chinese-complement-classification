#!/bin/bash

if [[ $# -eq 2 ]]
then
	model_name=$1
	gpu_id=$2
else 
	echo "Run format > ./run_inference.sh {model_dir} {gpu_id}"
	exit 1
fi

if [[ ! -d models/"$model_name" ]]
then
	mkdir models/"$model_name"
fi

epochs=5
lr=1e-4
seed=42
train_batch=256
eval_batch=512
checkpoint=100

CUDA_VISIBLE_DEVICES=$gpu_id python3 src/main.py \
	--model bert --epochs $epochs \
	--lr $lr --seed $seed \
	--train_batch_size $train_batch \
	--eval_batch_size $eval_batch \
	--checkpoint $checkpoint \
	--save_dir "models/$model_name/$model_name.tar" \
	--model_name_or_path hfl/chinese-bert-wwm \
	--one_sent \
	--train_data "data/train.json" \
	--val_data "data/val.json"

