#!/bin/bash
#MODEL_PATH_A=checkpoints/noisy_data_2_softmax_seed1/pretrain-model.torch
#MODEL_PATH_B=checkpoints/noisy_data_2_softmax_seed1/finetune-model.torch
DATASET=datasets
#CUDA_VISIBLE_DEVICES=2 python bin/main.py \
#    --model crf \
#    --batch_size 32 \
#    --learning_rate 2e-5 \
#    --epochs 5 \
#    --do_train \
#    --do_test \
#    --trainset  $DATASET/noisy_data_2/train_data_raw.json \
#    --evalset $DATASET/noisy_data_2/test_data_raw.json \
#    --testset $DATASET/noisy_data_2/test_data_raw.json \
#    --bert_load_mode from_pretrained \
#    --model_save_path $MODEL_PATH_A
#
#CUDA_VISIBLE_DEVICES=2 python bin/main.py \
#    --model softmax \
#    --batch_size 32 \
#    --learning_rate 2e-5 \
#    --epochs 5 \
#    --do_train \
#    --do_test \
#    --do_outdict \
#    --trainset  $DATASET/appen_not_filter/train_data_raw.json \
#    --evalset $DATASET/appen_not_filter/val_data_raw.json \
#    --testset $DATASET/appen_not_filter/test_data_raw.json \
#    --bert_load_mode bert_only \
#    --model_load_path $MODEL_PATH_A \
#    --model_save_path $MODEL_PATH_B

# baseline
#CUDA_VISIBLE_DEVICES=2 python bin/main.py \
#    --model crf \
#    --batch_size 32 \
#    --learning_rate 2e-5 \
#    --epochs 5 \
#    --do_train \
#    --do_test \
#    --do_outdict \
#    --trainset  $DATASET/appen_not_filter_equal_sampling/train_data_raw.json \
#    --evalset $DATASET/appen_not_filter_equal_sampling/val_data_raw.json \
#    --testset $DATASET/appen_not_filter_equal_sampling/test_data_raw.json \
#    --bert_load_mode from_pretrained \
#    --model_save_path checkpoints/baseline_appen_not_filter_equal_sampling_seed1.torch

# experiment
CUDA_VISIBLE_DEVICES=2 python bin/main.py \
    --model crf \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --epochs 5 \
    --do_train \
    --do_test \
    --trainset  $DATASET/merge_clean_raw_sent/train_data_raw.json \
    --evalset $DATASET/appen_not_filter/test_data_raw.json \
    --testset $DATASET/appen_not_filter/test_data_raw.json \
    --bert_load_mode from_pretrained \
    --model_save_path checkpoints/merge_clean_seed12345.torch
