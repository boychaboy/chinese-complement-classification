#!/bin/bash

if [[ $# -eq 2 ]]
then
	model_name=$1
	gpu_id=$2
else 
	echo "Run format > ./run_inference.sh {model_dir} {gpu_id}"
	exit 1
fi

CUDA_VISIBLE_DEVICES=$gpu_id python3 src/inference5.py \
	--test_path "data/new_test5.json" \
	--model_path "models/$model_name/$model_name.tar" \
	--model_type "hfl/chinese-macbert-large" \
	--result_path "models/$model_name/test_result.txt" \
	--wrong_path "models/$model_name/wrong_sent.txt" \
	--one_sent

