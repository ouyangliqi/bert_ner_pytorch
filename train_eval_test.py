# -*- coding: utf-8 -*-
# @Time    : 10/21/2020 5:50 PM
# @Author  : Chloe Ouyang
# @FileName: train_eval_test.py
from train_helper import train_model
from dataloder import NerProcessor, NERDataSet
import torch
from torch.utils import data
from transformers import BertTokenizer
from model_util import evaluate
import numpy as np


def train(params, model, ner_processor):
    tokenizer = BertTokenizer.from_pretrained(params["bert_model"])

    train_examples = ner_processor.get_train_examples(params['trainset'])

    train_dataset = NERDataSet(data_list=train_examples, tokenizer=tokenizer, label_map=params["label_map"],
                               max_len=256)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=params['batch_size'],
                                 shuffle=True,
                                 num_workers=0)

    params["num_train_examples"] = len(train_examples)

    eval_examples = ner_processor.get_test_examples(params['evalset'])

    eval_dataset = NERDataSet(data_list=eval_examples, tokenizer=tokenizer, label_map=params["label_map"],
                              max_len=256)

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=params['batch_size'],
                                shuffle=False,
                                num_workers=0)

    train_model(params, model, train_iter, eval_iter)


def test(params, model, ner_processor):
    tokenizer = BertTokenizer.from_pretrained(params["bert_model"])

    test_examples = ner_processor.get_test_examples(params['testset'])

    test_dataset = NERDataSet(data_list=test_examples, tokenizer=tokenizer, label_map=params["label_map"],
                              max_len=256)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=params['batch_size'],
                                shuffle=False,
                                num_workers=0)

    run_eval(params, model, test_iter)
    if params["bert_load_mode"] == "bert_only":
        out_dict_test_examples = ner_processor.get_outdic_test_examples(params['testset'])

        out_dict_test_dataset = NERDataSet(data_list=out_dict_test_examples, tokenizer=tokenizer,
                                           label_map=params["label_map"],
                                           max_len=256)

        out_dict_test_iter = data.DataLoader(dataset=out_dict_test_dataset,
                                             batch_size=params['batch_size'],
                                             shuffle=False,
                                             num_workers=0)
        print("out dict performance")
        run_eval(params, model, out_dict_test_iter)


def run_eval(params, model, iter_data):
    device = params["device"]

    model = model.eval()
    predictions, true_labels = [], []
    for step, batch in enumerate(iter_data):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_labels, b_input_mask, b_token_type_ids = batch

        with torch.no_grad():
            output = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels)

        loss, logits = output[:2]
        label_ids = b_labels.to('cpu').numpy()

        if params["model"] == "softmax":
            logits = logits.detach().cpu().numpy()

            prediction = [list(p) for p in np.argmax(logits, axis=2)]
            assert len(prediction) == len(label_ids)

            for pred, lab in zip(prediction, label_ids):
                preds = []
                labels = []
                for p, l in zip(pred, lab):
                    if l != 0:
                        preds.append(p)
                        labels.append(l)
                assert len(preds) == len(labels)
                predictions.append(preds)
                true_labels.append(labels)
        else:
            paths, scores = model.crf.viterbi_decode(logits, length_index=b_input_mask)
            assert len(paths) == len(label_ids)

            for pred, lab in zip(paths, label_ids):
                preds = []
                labels = []
                for p, l in zip(pred[0], lab):
                    preds.append(p)
                    labels.append(l)
                assert len(preds) == len(labels)

                predictions.append(preds)
                true_labels.append(labels)

    evaluate(params, predictions, true_labels)
