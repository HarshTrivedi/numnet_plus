#!/usr/bin/env bash

set -xe

SEED=$1
EPOCHS=$2
LR=$3
BLR=$4
WD=$5
BWD=$6
BATCH=$7
GRAD=$8
TMSPAN=$9

CODE_DIR=.

if [ ${TMSPAN} = tag_mspan ];then
  echo "Use tag_mspan model..."
  CACHED_TRAIN=${DATA_DIR}/tmspan_cached_roberta_train.pkl
  CACHED_DEV=${DATA_DIR}/tmspan_cached_roberta_dev.pkl
  MODEL_CONFIG="--gcn_steps 3 --use_gcn --tag_mspan"
  if [ \( ! -e "${CACHED_TRAIN}" \)  -o \( ! -e "${CACHED_DEV}" \) ]; then
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${OUT_DIR} --model_path ${MODEL_DIR} --tag_mspan
  CACHE_DIR=${OUT_DIR}
  else
  CACHE_DIR=${DATA_DIR}
  fi
else
  echo "Use mspan model..."
  CACHED_TRAIN=${DATA_DIR}/cached_roberta_train.pkl
  CACHED_DEV=${DATA_DIR}/cached_roberta_dev.pkl
  MODEL_CONFIG="--gcn_steps 3 --use_gcn"
  if [ \( ! -e "${CACHED_TRAIN}" \)  -o \( ! -e "${CACHED_DEV}" \) ]; then
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${OUT_DIR} --model_path ${MODEL_DIR}
  CACHE_DIR=${OUT_DIR}
  else
  CACHE_DIR=${DATA_DIR}
  fi
fi


SAVE_DIR=${OUT_DIR}/numnet_plus_${SEED}_LR_${LR}_BLR_${BLR}_WD_${WD}_BWD_${BWD}${TMSPAN}
DATA_CONFIG="--data_dir ${CACHE_DIR} --save_dir ${SAVE_DIR}"
TRAIN_CONFIG="--batch_size ${BATCH} --eval_batch_size ${BATCH} --max_epoch ${EPOCHS} --warmup 0.06 --optimizer adam \
              --learning_rate ${LR} --weight_decay ${WD} --seed ${SEED} --gradient_accumulation_steps ${GRAD} \
              --bert_learning_rate ${BLR} --bert_weight_decay ${BWD} --log_per_updates 100 --eps 1e-6"
BERT_CONFIG="--roberta_model ${MODEL_DIR}"


echo "Start training..."
python ${CODE_DIR}/roberta_gcn_cli.py \
    ${DATA_CONFIG} \
    ${TRAIN_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}

echo "Starting evaluation..."
TEST_CONFIG="--eval_batch_size ${BATCH} --pre_path ${SAVE_DIR}/checkpoint_best.pt --data_mode dev --dump_path ${SAVE_DIR}/dev.json \
             --inf_path ${DATA_DIR}/drop_dataset_dev.json"

python ${CODE_DIR}/roberta_predict.py \
    ${DATA_CONFIG} \
    ${TEST_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}

python ${CODE_DIR}/drop_eval.py \
    --gold_path ${DATA_DIR}/drop_dataset_dev.json \
    --prediction_path ${SAVE_DIR}/dev.json --output_path ${OUT_DIR}/metrics.json
