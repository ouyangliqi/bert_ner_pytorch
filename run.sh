#!/bin/bash
MODEL_PATH_A=checkpoints/pretrain-model.torch
MODEL_PATH_B=checkpoints/finetune-model.torch
DATASET=datasets
#CUDA_VISIBLE_DEVICES=2 python bin/main.py \
#    --model crf \
#    --batch_size 32 \
#    --learning_rate 2e-5 \
#    --do_train \
#    --do_test \
#    --trainset  $DATASET/train_data_raw.json \
#    --evalset $DATASET/test_data_raw.json \
#    --testset $DATASET/test_data_raw.json \
#    --bert_load_mode from_pretrained \
#    --model_save_path $MODEL_PATH_A
#
#CUDA_VISIBLE_DEVICES=2 python bin/main.py \
#    --model crf\
#    --batch_size 32 \
#    --learning_rate 2e-5 \
#    --do_train \
#    --do_test \
#    --trainset  $DATASET/appen/train_data_raw.json \
#    --evalset $DATASET/appen/val_data_raw.json \
#    --testset $DATASET/appen/test_data_raw.json \
#    --bert_load_mode bert_only \
#    --model_load_path $MODEL_PATH_A \
#    --model_save_path $MODEL_PATH_B

# baseline
CUDA_VISIBLE_DEVICES=2 python bin/main.py \
    --model crf \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --do_train \
    --do_test \
    --do_outdict \
    --trainset  $DATASET/appen_not_filter/train_data_raw.json \
    --evalset $DATASET/appen_not_filter/val_data_raw.json \
    --testset $DATASET/appen_not_filter/test_data_raw.json \
    --bert_load_mode from_pretrained \
    --model_save_path checkpoints/baseline.torch