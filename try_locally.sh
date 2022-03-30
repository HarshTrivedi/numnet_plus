#!/usr/bin/env bash

python prepare_roberta_data.py --input_path drop_dataset_fixture --output_dir serialized_drop_dataset_fixture --model_path roberta.large --tag_mspan

python roberta_gcn_cli.py \
    --gcn_steps 3 --use_gcn \
    --data_dir serialized_drop_dataset_fixture --save_dir serialization_dir \
    --batch_size 16 --eval_batch_size 16 --max_epoch 5 --num_instances_per_epoch 10 \
    --warmup 0.06 --optimizer adam  --learning_rate 5e-4 --weight_decay 5e-5 --seed 100 \
    --gradient_accumulation_steps 8 --bert_learning_rate 1.5e-5 --bert_weight_decay  0.01 \
    --log_per_updates 100 --eps 1e-6 \
    --roberta_model roberta.large --tag_mspan
