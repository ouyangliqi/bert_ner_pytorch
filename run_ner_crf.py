# -*- coding: utf-8 -*-
# @Time    : 10/21/2020 5:52 PM
# @Author  : Chloe Ouyang
# @FileName: run_ner_crf.py
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--trainset", type=str, default="conll2003/train.txt")
    parser.add_argument("--validset", type=str, default="conll2003/valid.txt")