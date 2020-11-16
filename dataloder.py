# -*- coding: utf-8 -*-
# @Time    : 10/7/2020 2:40 PM
# @Author  : Chloe Ouyang
# @FileName: dataloder.py
import torch
from torch.utils import data
from util import *
MAX_LEN = 256 - 2


def readfile(filename):
    """ read file"""
    def normalize(entry, sents, labels, data):
        sentence = [i.lower() if len(i.encode('utf-8')) != 4 else ' ' for i in entry['content']]
        tags = get_sentence_token_label(''.join(sentence), entry['ner'])
        sents.append(sentence)
        labels.append(tags)
        data.append((sentence, tags))

    data = []
    entries = read_json(filename)
    sents, labels = [], []  # list of lists

    for entry in entries:
        sentence = [i.lower() if len(i.encode('utf-8')) != 4 else ' ' for i in entry['content']]
        if len(sentence) > MAX_LEN:
            entry_list = []
            # TODO: need to split sentences and ners
            # for i in range(int(len(sentence) / MAX_LEN) + 1):
            #     entry_list.append(sentence[i * MAX_LEN:(i + 1) * MAX_LEN])

            for subentry in entry_list:
                normalize(subentry, sents, labels, data)
        else:
            normalize(entry, sents, labels, data)

    assert len(sents) == len(labels)
    return data


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, segment_ids=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.segment_ids = segment_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(data_dir), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(data_dir), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(data_dir), "test")

    def get_labels(self):
        return ["O", "B-Activity", "B-Brand", "B-Channel", "B-Location", "B-Money", "B-Person", "B-Product", "B-Time",
                "E-Activity", "E-Brand", "E-Channel", "E-Location", "E-Money", "E-Person", "E-Product", "E-Time",
                "I-Activity", "I-Brand", "I-Channel", "I-Location", "I-Money", "I-Person", "I-Product", "I-Time",
                "S-Activity", "S-Brand", "S-Channel", "S-Location", "S-Money", "S-Person", "S-Product", "S-Time",
                "[CLS]", "[SEP]"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ''.join(sentence)
            label = label
            examples.append(InputExample(guid=guid, text=text_a, label=label))
        return examples


class NERDataSet(data.Dataset):
    """ DataSet loader for crf"""
    def __init__(self, data_list, tokenizer, label_map, max_len):
        self.max_len = max_len
        self.label_map = label_map
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_example = self.data_list[idx]
        text = input_example.text
        labels = input_example.label
        word_tokens = ['[CLS]']
        label_list = ['[CLS]']
        label_mask = [0]  # value in (0, 1) - 0 signifies invalid token

        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        label_ids = [self.label_map['[CLS]']]

        # iterate over individual tokens and their labels
        tokens = self.tokenizer.tokenize(text)
        assert len(tokens) == len(labels)

        for word, label in zip(tokens, labels):
            word_tokens.append(word)
            input_ids.append(self.tokenizer.convert_tokens_to_ids(word))

            label_list.append(label)
            label_ids.append(self.label_map[label])
            label_mask.append(1)

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask)

        if len(word_tokens) >= self.max_len:
            word_tokens = word_tokens[:(self.max_len - 1)]
            label_list = label_list[:(self.max_len - 1)]
            input_ids = input_ids[:(self.max_len - 1)]
            label_ids = label_ids[:(self.max_len - 1)]
            label_mask = label_mask[:(self.max_len - 1)]

        assert len(word_tokens) < self.max_len, len(word_tokens)

        word_tokens.append('[SEP]')
        label_list.append('[SEP]')
        input_ids.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        label_ids.append(self.label_map['[SEP]'])
        label_mask.append(0)

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask)

        sentence_id = [0 for _ in input_ids]
        attention_mask = [1 for _ in input_ids]

        while len(input_ids) < self.max_len:
            input_ids.append(0)
            label_ids.append(0)
            attention_mask.append(0)
            sentence_id.append(0)
            label_mask.append(0)

        assert len(word_tokens) == len(label_list)
        assert len(input_ids) == len(label_ids) == len(attention_mask) == len(sentence_id) == len(
            label_mask) == self.max_len, len(input_ids)
        # return word_tokens, label_list,
        return torch.LongTensor(input_ids), torch.LongTensor(label_ids), torch.LongTensor(
            attention_mask), torch.LongTensor(sentence_id), torch.BoolTensor(label_mask)
