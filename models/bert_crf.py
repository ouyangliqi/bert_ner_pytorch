# -*- coding: utf-8 -*-
# @Time    : 11/16/2020 5:52 PM
# @Author  : Chloe Ouyang
# @FileName: bert_crf.py

from transformers import BertPreTrainedModel, BertModel
from models.layers.crf_layer import *
import torch.nn as nn


class BERTCRFClass(BertPreTrainedModel):
    """继承父类PreTrainedModel"""
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf = LinearChainCRF(in_dim=config.hidden_size, num_tags=num_labels)

        self.init_weights()
        self.crf.init_weight()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]  # (b, MAX_LEN, 768)

        sequence_output = self.dropout(sequence_output)

        outputs = (sequence_output,)

        if labels is not None:
            loss = self.crf.nll_loss(sequence_output.float(), labels, length_index=attention_mask)
            outputs = (loss,) + outputs
        return outputs
