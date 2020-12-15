# -*- coding: utf-8 -*-
# @Time    : 12/10/2020 6:15 PM
# @Author  : Chloe Ouyang
# @FileName: main.py
import argparse
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from train_eval_test import train, test
import torch
from models import BERTSoftmax, BERTCRF
from dataloder import NerProcessor
from transformers import BertConfig, BertTokenizer
from model_util import from_state_dict


def main():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument("--batch_size", default=64, help="batch size", type=int)
    parser.add_argument("--vocab_size", default=30000, help="Vocabulary size", type=int)
    parser.add_argument("--embed_size", default=256, help="Words embeddings dimension", type=int)
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer "
                             "API documentation on tensorflow site for more details.", type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped",
                        type=float)
    parser.add_argument('--cov_loss_wt', default=0.5, help='Weight of coverage loss (lambda in the paper).'
                                                           ' If zero, then no incentive to minimize coverage loss.',
                        type=float)

    # path
    # /ckpt/checkpoint/checkpoint
    parser.add_argument("--model_path", help="Path to a specific model", default="", type=str)
    parser.add_argument("--trainset", default='{}/datasets/train_data_raw.json'.format(BASE_DIR),
                        help="train set")
    parser.add_argument("--evalset", default='{}/datasets/val_data_raw.json'.format(BASE_DIR),
                        help="train_seg_x_dir")
    parser.add_argument("--testset", default='{}/datasets'.format(BASE_DIR),
                        help="train_seg_x_dir")
    parser.add_argument("--vocab_path", default='{}/datasets/vocab.txt'.format(BASE_DIR), help="Vocab path")
    parser.add_argument("--log_file", help="File in which to redirect console outputs", default="", type=str)
    parser.add_argument("--test_save_dir", default='{}/datasets/'.format(BASE_DIR), help="test_save_dir")

    # others
    parser.add_argument("--steps_per_epoch", default=8087, help="max_train_steps", type=int)
    parser.add_argument("--checkpoints_save_steps", default=10, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--epochs", default=3, help="train epochs", type=int)

    # mode
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run testing.")
    parser.add_argument("--do_outdict", action='store_true', help="Whether to run testing result of out dictionary.")
    parser.add_argument("--bert_load_mode", default='test', help="training, eval or test options")
    parser.add_argument("--bert_model", default='bert-base-chinese', help="which base model to be selected")
    parser.add_argument("--model", default='crf', help="which model to be selected")
    parser.add_argument("--model_save_path", default='{}/checkpoints/model.torch'.format(BASE_DIR), help="which model to be selected")

    args = parser.parse_args()
    params = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params["device"] = device

    bert_model = params["bert_model"]
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    ner_processor = NerProcessor(tokenizer)
    tags_vals = ner_processor.get_labels()

    label_map = {}

    for (i, label) in enumerate(tags_vals):
        label_map[label] = i
    id2tag = {lid: label for label, lid in label_map.items()}
    params["label_map"] = label_map
    params["id2tag"] = id2tag

    # load model
    config = BertConfig.from_pretrained(bert_model)

    if params["bert_load_mode"] == "from_pretrained":
        if params["model"] == "softmax":
            model = BERTSoftmax.from_pretrained(bert_model, num_labels=len(label_map))
        elif params["model"] == "crf":
            model = BERTCRF.from_pretrained(bert_model, config=config, num_labels=len(label_map))

    elif params["bert_load_mode"] == "bert_only":
        all_state = torch.load(params["model_save_path"])

        if params["model"] == "softmax":
            model = BERTSoftmax.from_pretrained(bert_model, num_labels=len(label_map))

        elif params["model"] == "crf":
            model = BERTCRF.from_pretrained(bert_model, config=config, num_labels=len(label_map))
        # print(all_state.keys())
        model = from_state_dict(model, all_state)

    model.to(params["device"])
    print("-------------------model loaded------------------")

    if params["do_train"]:
        train(params, model, ner_processor)
        print("---------------train completed--------------")

    if params["do_test"]:
        model.load_state_dict(torch.load(params["model_save_path"]))
        test(params, model, ner_processor)


if __name__ == '__main__':
    main()
