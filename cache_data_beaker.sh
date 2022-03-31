#!/usr/bin/env bash

set -xe

TMSPAN=$1
SKIP_TRAIN=$2

if [ ${SKIP_TRAIN} = true ];then
    SKIP_TRAIN_STR="--skip_train"
else
    SKIP_TRAIN_STR=""
fi

if [ ${TMSPAN} = tag_mspan ];then
  echo "Use tag_mspan model..."
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${OUT_DIR} --model_path ${MODEL_DIR} --tag_mspan ${SKIP_TRAIN_STR}
else
  echo "Use mspan model..."
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${OUT_DIR} --model_path ${MODEL_DIR} ${SKIP_TRAIN_STR}
fi
