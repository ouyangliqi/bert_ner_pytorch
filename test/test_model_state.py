# -*- coding: utf-8 -*-
# @Time    : 12/14/2020 10:03 AM
# @Author  : Chloe Ouyang
# @FileName: test_model_state.py
import torch


def test_model_state(params, model):
    pretrain_model_state = torch.load(params["model_load_path"])
    assert pretrain_model_state.keys() == model.state_dict().keys()
    for key in pretrain_model_state.keys():
        if key.startswith("bert."):
            assert pretrain_model_state[key] == model.state_dict[key]
        else:
            assert pretrain_model_state[key] != model.state_dict[key]