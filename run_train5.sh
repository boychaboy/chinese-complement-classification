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

epochs=10
lr=5e-5
seed=42
train_batch=128

CUDA_VISIBLE_DEVICES=$gpu_id python3 src/main.py \
	--model bert --epochs $epochs \
	--lr $lr --seed $seed \
	--train_batch_size $train_batch \
	--save_dir "models/$model_name/$model_name.tar" \
	--model_name_or_path hfl/chinese-macbert-large \
	--train_data "data/train5.json" \
	--val_data "data/val5.json"
