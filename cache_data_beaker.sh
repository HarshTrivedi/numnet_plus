#!/usr/bin/env bash

set -xe

TMSPAN=$1

if [ ${TMSPAN} = tag_mspan ];then
  echo "Use tag_mspan model..."
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${OUT_DIR} --model_path ${MODEL_DIR} --tag_mspan
else
  echo "Use mspan model..."
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${OUT_DIR} --model_path ${MODEL_DIR}
fi
