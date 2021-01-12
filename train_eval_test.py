# -*- coding: utf-8 -*-
# @Time    : 10/21/2020 5:50 PM
# @Author  : Chloe Ouyang
# @FileName: train_eval_test.py
from train_helper import train_model
from dataloder import NerProcessor, NERDataSet
from torch.utils import data
from transformers import BertTokenizer
from model_util import run_eval_classification_report, run_eval_classification_report_dict, evaluate, ner_index, get_word_index_with_tokenizer
from utils import read_json


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


def test(params, model, ner_processor, output_dict=False):
    tokenizer = BertTokenizer.from_pretrained(params["bert_model"])

    test_examples = ner_processor.get_test_examples(params['testset'])

    test_dataset = NERDataSet(data_list=test_examples, tokenizer=tokenizer, label_map=params["label_map"],
                              max_len=256)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=params['batch_size'],
                                shuffle=False,
                                num_workers=0)
    if output_dict:
        overall = run_eval_classification_report_dict(params, model, test_iter)
    else:
        run_eval_classification_report(params, model, test_iter)

    if params["do_outdict"]:
        out_dict_test_examples = ner_processor.get_outdic_test_examples(params['testset'])

        out_dict_test_dataset = NERDataSet(data_list=out_dict_test_examples, tokenizer=tokenizer,
                                           label_map=params["label_map"],
                                           max_len=256)

        out_dict_test_iter = data.DataLoader(dataset=out_dict_test_dataset,
                                             batch_size=params['batch_size'],
                                             shuffle=False,
                                             num_workers=0)
        if output_dict:
            out_dict = run_eval_classification_report_dict(params, model, out_dict_test_iter)
        else:
            print("out dict performance")
            run_eval_classification_report(params, model, out_dict_test_iter)

    if output_dict:
        return overall, out_dict


def test_and_save(params, model, ner_processor):
    tokenizer = BertTokenizer.from_pretrained(params["bert_model"])

    test_examples = ner_processor.get_test_examples(params['testset'])

    test_dataset = NERDataSet(data_list=test_examples, tokenizer=tokenizer, label_map=params["label_map"],
                              max_len=256)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=params['batch_size'],
                                shuffle=False,
                                num_workers=0)

    pred_tags, valid_tags = evaluate(params, model, test_iter)

    tobepredict = read_json(params['testset'])

    # mapping
    mapping = {}
    cindex = 0
    ctext = ''
    index_list = []
    for index, example in enumerate(test_examples):
        ctext += example.text
        index_list.append(index)

        tobepredict_text = ''.join(
            [i.lower() if len(i.encode('utf-8')) != 4 else ' ' for i in tobepredict[cindex]['content']])
        if ctext == tobepredict_text:
            mapping[cindex] = index_list
            cindex += 1
            ctext = ''
            index_list = []

    result = []
    for key, value in mapping.items():
        pred = [p_i for p in value for p_i in pred_tags[p]]
        text = tobepredict[key]['content']
        p_text = ''.join([s.lower() if len(s.encode('utf-8')) != 4 else ' ' for s in text])
        p_text = p_text.replace('\u200b', '').replace('\u200c', '').replace('\u200e', '')

        word_index = get_word_index_with_tokenizer(p_text, tokenizer)

        ners = []
        for r in ner_index(pred):
            tmp = word_index[r[0] - 1:r[1] - 1]
            if tmp:
                start_ind = tmp[0][0]
                end_ind = tmp[-1][1]
                ners.append({
                    'text_segment': text[start_ind:end_ind],
                    'label_name': r[-1],
                    'start_ind': start_ind,
                    'end_ind': end_ind
                })
        result.append({"content": tobepredict[key]['content'], "ner": ners, "isbert": True})
    return result

