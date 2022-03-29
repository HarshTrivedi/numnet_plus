#!/usr/bin/env bash

set -xe

BATCH=16

if test -f "${CKPT_DIR}/is_tag_mspan.txt"; then
  echo "Use tag_mspan model..."
  MODEL_CONFIG="--gcn_steps 3 --use_gcn --tag_mspan"
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${DATA_DIR} --model_path ${MODEL_DIR} --tag_mspan
  CACHE_DIR=${DATA_DIR}
else
  echo "Use mspan model..."
  MODEL_CONFIG="--gcn_steps 3 --use_gcn"
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${DATA_DIR} --model_path ${MODEL_DIR}
  CACHE_DIR=${DATA_DIR}
fi

DATA_CONFIG="--data_dir ${CACHE_DIR} --save_dir ${OUT_DIR}"
BERT_CONFIG="--roberta_model ${MODEL_DIR}"

echo "Starting evaluation..."
TEST_CONFIG="--eval_batch_size ${BATCH} --pre_path ${CKPT_DIR}/checkpoint_best.pt --data_mode dev --dump_path ${OUT_DIR}/dev.json \
             --inf_path ${DATA_DIR}/drop_dataset_dev.json"

CODE_DIR=.

python ${CODE_DIR}/roberta_predict.py \
    ${DATA_CONFIG} \
    ${TEST_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}

python ${CODE_DIR}/drop_eval.py \
    --gold_path ${DATA_DIR}/drop_dataset_dev.json \
    --prediction_path ${OUT_DIR}/dev.json --output_path ${OUT_DIR}/metrics.json
