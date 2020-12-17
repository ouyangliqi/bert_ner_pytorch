# -*- coding: utf-8 -*-
# @Time    : 12/14/2020 10:03 AM
# @Author  : Chloe Ouyang
# @FileName: test_model_state.py
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch
from models import BERTSoftmax, BERTCRF
from transformers import BertConfig
from model_util import from_state_dict


def test_model_state():
    pretrain_model_path = "checkpoints/noisy_data_1/pretrain-model.torch"
    model_name = "bert-base-chinese"

    config = BertConfig.from_pretrained(model_name)
    pretrain_model_state = torch.load(pretrain_model_path)
    finetune_model = BERTCRF.from_pretrained(model_name, config=config, num_labels=33).to("cuda")
    finetune_model = from_state_dict(finetune_model, pretrain_model_state)

    finetune_model_state = finetune_model.state_dict()

    assert pretrain_model_state.keys() == finetune_model_state.keys()

    for key in pretrain_model_state.keys():
        if key.startswith("bert."):
            assert torch.equal(pretrain_model_state[key], finetune_model_state[key])
        else:
            assert not torch.equal(pretrain_model_state[key], finetune_model_state[key])


if __name__ == '__main__':
    test_model_state()
