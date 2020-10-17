TRAIN_FILE=$1
OUTPUT_DIR=$2

CUDA_VISIBLE_DEVICES=0 python src/run_language_modeling.py \
	--output_dir=$OUTPUT_DIR \
	--model_type=bert \
	--model_name_or_path=bert-base-chinese \
	--do_train \
	--line_by_line \
	--train_data_file=$TRAIN_FILE \
	--mlm
