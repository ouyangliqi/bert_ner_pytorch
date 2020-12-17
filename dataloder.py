# -*- coding: utf-8 -*-
# @Time    : 10/7/2020 2:40 PM
# @Author  : Chloe Ouyang
# @FileName: dataloder.py
import torch
from torch.utils import data
from util import is_overlapping, read_json

MAX_LEN = 256 - 2


def convert_label_bieos(label_token_index_list):
    index_label_dict = {}
    for label_token_index in label_token_index_list:
        label = label_token_index[0]
        label_token_indexs = label_token_index[1]
        length = len(label_token_indexs)
        if length == 1:
            index_label_dict[tuple(label_token_indexs[0])] = 'S-' + label
        elif length > 1:
            index_label_dict[tuple(label_token_indexs[0])] = 'B-' + label
            for ind in range(1, length - 1):
                index_label_dict[tuple(label_token_indexs[ind])] = 'I-' + label
            index_label_dict[tuple(label_token_indexs[-1])] = 'E-' + label
    return index_label_dict


def get_label_token_index(word_indexs, ner_label_index):
    label_token_index_list = []
    for label_index in ner_label_index:
        label_token_index = []
        span_index = [label_index['start_ind'], label_index['end_ind']]
        span_label = label_index['label_name']
        flat_word_indexs = [item for item in word_indexs]
        for word_index in flat_word_indexs:
            if is_overlapping(span_index, word_index):
                label_token_index.append(word_index)
        label_token_index_list.append([span_label, label_token_index])
    return label_token_index_list


def get_word_index_with_tokenizer(text, tokenizer):
    originial_index = 0
    word_index = []

    for token in tokenizer.tokenize(text):
        if token == '[UNK]':
            word_index.append([originial_index, originial_index + 1])
            originial_index += 1
            continue

        if token.startswith("##"):
            token = token[2:]

        tmp = []
        start_ind = text.find(token, originial_index)
        originial_index = start_ind
        for index, char in enumerate(token):
            if char == text[originial_index]:
                tmp.append(char)
                originial_index += 1
        #         if ''.join(tmp) != token:
        #             print([text])
        #         assert ''.join(tmp)==token
        end_ind = originial_index

        word_index.append([start_ind, end_ind])
    return word_index


def get_sentence_token_label(sentence, ner_list, tokenizer):
    word_indexs = get_word_index_with_tokenizer(sentence, tokenizer)
    label_token_index_list = get_label_token_index(word_indexs, ner_list)
    index_label_dict = convert_label_bieos(label_token_index_list)

    token_labels = [index_label_dict.get(
        tuple(item), 'O') for item in word_indexs]
    return token_labels


def split_max_len(entry):
    break_point, ner_break_point = 0, 10000

    # 按照长度切分 但是不能切到实体
    # 切了一次之后长度还是超过max_len
    for index in range(len(entry['ner'])):
        ner = entry['ner'][index]
        if ner['start_ind'] < MAX_LEN and ner['end_ind'] > MAX_LEN:
            break_point = ner['start_ind']
            ner_break_point = index
        elif ner['start_ind'] >= MAX_LEN and ner_break_point > index:
            ner_break_point = index

    if break_point == 0:
        break_point = MAX_LEN

    entry_f = {
        "content": entry['content'][:break_point],
        "ner": entry['ner'][:ner_break_point]
    }

    entry_b = {
        "content": entry['content'][break_point:],
        "ner": []
    }
    # reindex
    for i in entry['ner'][ner_break_point:]:
        entry_b['ner'].append({
            "text_segment": i["text_segment"],
            "label_name": i["label_name"],
            "start_ind": i["start_ind"] - len(entry_f['content']),
            "end_ind": i["end_ind"] - len(entry_f['content'])
        })

    return entry_f, entry_b


def readfile(filename, tokenizer, outdict=False):
    """
    read file
    """

    def normalize(entry, sents, labels, data, tokenizer):
        sentence = [i.lower() if len(i.encode('utf-8')) != 4 else ' ' for i in entry['content']]
        tags = get_sentence_token_label(''.join(sentence), entry['ner'], tokenizer)
        sents.append(sentence)
        labels.append(tags)
        data.append((sentence, tags))

    data = []
    entries = read_json(filename)
    sents, labels = [], []  # list of lists

    for entry in entries:
        if outdict:
            if not entry['outdict']:
                continue
        sentence = [i.lower() if len(i.encode('utf-8')) != 4 else ' ' for i in entry['content']]
        if len(sentence) > MAX_LEN:
            # TODO: need to split sentences and ners
            if len(split_max_len(entry)[1]['content']) > MAX_LEN:
                entry_list = []
                entry_list.append(split_max_len(entry)[0])
                entry_list.extend(split_max_len(split_max_len(entry)[1]))
            else:
                entry_list = split_max_len(entry)

            for subentry in entry_list:
                normalize(subentry, sents, labels, data, tokenizer)
        else:
            normalize(entry, sents, labels, data, tokenizer)

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

    @classmethod
    def __init__(cls, tokenizer):
        cls.tokenizer = tokenizer

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
    def _read_file(cls, input_file, outdict=False, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file, cls.tokenizer, outdict=outdict)


class NerProcessor(DataProcessor):
    """Processor for data set."""

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

    def get_outdic_test_examples(self, data_dir):
        return self._create_examples(
            self._read_file(data_dir, outdict=True), "test")

    def get_labels(self):
        return ["X", "O", "B-Activity", "B-Brand", "B-Channel", "B-Location", "B-Money", "B-Person", "B-Product",
                "B-Time",
                "E-Activity", "E-Brand", "E-Channel", "E-Location", "E-Money", "E-Person", "E-Product", "E-Time",
                "I-Activity", "I-Brand", "I-Channel", "I-Location", "I-Money", "I-Person", "I-Product", "I-Time",
                "S-Activity", "S-Brand", "S-Channel", "S-Location", "S-Money", "S-Person", "S-Product", "S-Time"]

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
    """ DataSet loader"""

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

        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        label_ids = [self.label_map['O']]

        # iterate over individual tokens and their labels
        tokens = self.tokenizer.tokenize(text)
        assert len(tokens) == len(labels)

        for word, label in zip(tokens, labels):
            word_tokens.append(word)
            input_ids.append(self.tokenizer.convert_tokens_to_ids(word))

            label_list.append(label)
            label_ids.append(self.label_map[label])

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids)

        if len(word_tokens) >= self.max_len:
            word_tokens = word_tokens[:(self.max_len - 1)]
            label_list = label_list[:(self.max_len - 1)]
            input_ids = input_ids[:(self.max_len - 1)]
            label_ids = label_ids[:(self.max_len - 1)]

        assert len(word_tokens) < self.max_len, len(word_tokens)

        word_tokens.append('[SEP]')
        label_list.append('[SEP]')
        input_ids.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        label_ids.append(self.label_map['O'])

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids)

        sentence_id = [0 for _ in input_ids]
        attention_mask = [1 for _ in input_ids]

        while len(input_ids) < self.max_len:
            input_ids.append(0)
            label_ids.append(self.label_map['X'])
            attention_mask.append(0)
            sentence_id.append(0)

        assert len(word_tokens) == len(label_list)
        assert len(input_ids) == len(label_ids) == len(attention_mask) == len(sentence_id) == self.max_len, len(
            input_ids)
        return torch.LongTensor(input_ids), torch.LongTensor(label_ids), torch.LongTensor(
            attention_mask), torch.LongTensor(sentence_id)
