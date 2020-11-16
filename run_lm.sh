TRAIN_FILE=$1
OUTPUT_DIR=$2
GPU_ID=$3

CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_language_modeling.py \
	--output_dir=$OUTPUT_DIR \
	--model_type=bert \
	--model_name_or_path=hfl/chinese-macbert-base \
	--do_train \
	--line_by_line \
	--train_data_file=$TRAIN_FILE \
	--mlm
