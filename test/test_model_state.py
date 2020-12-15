# -*- coding: utf-8 -*-
# @Time    : 12/14/2020 10:03 AM
# @Author  : Chloe Ouyang
# @FileName: test_model_state.py
import torch
from models import BERTSoftmax, BERTCRF
from transformers import BertConfig


def test_model_state():
    pretrain_model_path = "checkpoints/pretrain-model.torch"
    model_name = "bert-base-chinese"

    config = BertConfig.from_pretrained(model_name)
    pretrain_model_state = torch.load(pretrain_model_path)
    finetune_model_state = BERTCRF.from_pretrained(model_name, config=config, num_labels=33).state_dict()
    assert pretrain_model_state.keys() == finetune_model_state.keys()
    for key in pretrain_model_state.keys():
        if key.startswith("bert."):
            assert pretrain_model_state[key] == finetune_model_state[key]
        else:
            assert pretrain_model_state[key] != finetune_model_state[key]