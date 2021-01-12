# -*- coding: utf-8 -*-
# @Time    : 7/8/2020 5:37 PM
# @Author  : Chloe Ouyang
# @FileName: utils.py

import collections
import glob
import re
import unicodedata
import sentencepiece as spm
import pandas as pd
import json


def read_json(file_name, encoding='utf-8'):
    json_data = [json.loads(d) for d in open(file_name, encoding=encoding)]
    if len(json_data) == 1:
        return json_data[0]
    else:
        return json_data


def save_json(json_data, file_name, sort_keys=None, ensure_ascii=False, encoding='utf-8', indent=None):
    if sort_keys:
        from collections import OrderedDict

        def get_order_dict(json_input, sort_keys_list):
            tuple_list = []
            for k in sort_keys_list:
                tuple_list.append((k, json_input[k]))
            return OrderedDict(tuple_list)

    if isinstance(json_data, list):
        with open(file_name, 'w', encoding=encoding) as f:
            for data in json_data:
                if sort_keys:
                    data = get_order_dict(data, sort_keys)
                f.write(json.dumps(data, ensure_ascii=ensure_ascii, indent=indent) + '\n')
    else:
        with open(file_name, 'w', encoding=encoding) as f:
            if sort_keys:
                json_data = get_order_dict(json_data, sort_keys)
            json.dump(json_data, f, ensure_ascii=ensure_ascii, indent=indent)


def save_text(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        if isinstance(data, list):
            for d in data:
                f.write(d + '\n')
        else:
            f.write(data + '\n')


def get_raw_data(path):
    files = glob.glob(path)
    raw_data = []
    for file in files:
        data = read_json(file)['result']
        for d in data:
            if d['ner']:
                raw_data.append({"docid": d['docid'], "content": d['content'], "ner": d['ner']})
    return raw_data


def index_conversion(text):
    original_index = range(len(text) + 1)
    new_index = []
    incident = 0
    for i in range(len(text)):
        new_index.append(i + incident)
        if len(text[i].encode('utf-8')) == 4:
            incident += 1
    new_index.append(len(text) + incident)
    return dict(zip(new_index, original_index))


def align_ner_result_index(result):
    content = result['content']
    ner = result['ner']
    docid = result['docid']
    new_ner = []
    index_dict = index_conversion(content)
    for record in ner:
        new_ner.append({'text_segment': record['text_segment'],
                        'label_name': record['label_name'],
                        'start_ind': index_dict[record['start_ind']],
                        'end_ind': index_dict[record['end_ind']]})
    return {"content": content, "ner": new_ner, "docid": str(docid)}


def get_emoji_index_align(raw_data):
    return [align_ner_result_index(d) for d in raw_data]


class BasicTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, do_lower_case=True, normalize=True):
        token_specification = [
            ('URL', r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
                    r'|[a-zA-Z]+(?:[a-zA-Z]|[0-9]|[.]|[!*\(\),])+\.[a-zA-Z]{2,}(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),'
                    r']|(?:%[0-9a-fA-F]'
                    r'[0-9a-fA-F]))*'),
            ('EMAIL', r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+[a-zA-Z0-9]+"),  #
            # ('ACCOUNT', r"@[^“”？@,?!！？。、，\n\xa0\t\r:\#\" ]+"),
            ('ACCOUNT', r"@[0-9a-zA-Z\u4E00-\u9FA5\-\_]+"),
            ("DIGIT", r"\d"),
            # ("DIGIT", r"(\d+[,.]?)+\d*"),
            ("LETTERS", r"Dr\."
                        '|'
                        "dr\."
                        "|"
                        "mrs?\."
                        "|"
                        "Mrs?\."
                        "|"
                        "[a-zA-Zàâçéèêëîïôûùüÿñæœ\.\'-\?]*[a-zA-Zàâçéèêëîïôûùüÿñæœ]"),
            # ("EMOTION", r"(\[\S+?\])|(\(emoji\))"),
            # ("QUESTION", r"([？?][”\"]?[\s]*)+"),
            ("PUNCTUATION", r"(([!\n\r。！？?][”\"]?[\s]*)+)|(\.[^\d\u4E00-\u9FA5])"),
            ('SPACE', r'[ ]+'),
            ('CHAR', '[\u4E00-\u9FA5]'),  # Chinese character
            # ('COMMA', r'[,；;，]+'),  # Any other character
            ('SPECIAL', r'.'),  # Any other character
        ]

        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
        self.WORD_RE = re.compile(tok_regex, re.VERBOSE)

        self.Token = collections.namedtuple('Token', ['type', 'value', 'start', 'end'])
        self.do_lower_case = do_lower_case
        self.normalize = normalize

    def tokenize(self, text):
        origin_text = str(text)
        if self.normalize:
            text = unicodedata.normalize('NFKC', text)
        if self.do_lower_case:
            text = text.lower()
        word_token_data = self.get_word_tokenize(text)
        result = self.sent_tokenize_rm_eng_space(word_token_data)
        toks = []
        sen_indexs = []
        word_indexs = []
        types = []
        sents = []
        for sen in result:
            values = [tok.value for tok in sen]
            ty = [tok.type for tok in sen]
            ind = [sen[0].start, sen[-1].end]
            word_ind = [[tok.start, tok.end] for tok in sen]
            sents.append(origin_text[sen[0].start: sen[-1].end])
            sen_indexs.append(ind)
            word_indexs.append(word_ind)
            types.append(ty)
            toks.append(values)
        token_infos = {"token": toks, "type": types, "sent_indexs": sen_indexs, "word_indexs": word_indexs,
                       "sents": sents}
        return token_infos

    def yield_word_tokenize(self, code):
        for mo in self.WORD_RE.finditer(code):
            kind = mo.lastgroup
            value = mo.group()
            start = mo.start()
            end = mo.end()
            yield self.Token(kind, value, start, end)

    def get_word_tokenize(self, text):
        return [token for token in self.yield_word_tokenize(text)]

    def sent_tokenize_rm_eng_space(self, word_token_data):
        sentences = []
        tmp = []
        for token in word_token_data:

            if tmp:
                if tmp[-1].type == 'LETTERS' and token.type == 'SPACE':
                    pass
                else:
                    tmp.append(token)
            else:
                tmp.append(token)
            if token.type == 'PUNCTUATION' or token.type == 'QUESTION':
                sentences.append(tmp)
                tmp = []
        if tmp:
            sentences.append(tmp)
        return sentences


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, bpe_model_file, do_lower_case=True, normalize=True):
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, normalize=normalize)
        self.bpe_processor = spm.SentencePieceProcessor()
        self.bpe_processor.load(bpe_model_file)

    def tokenize(self, text):
        output_token = []
        output_type = []
        output_word_indexes = []
        output_token_ids = []
        basic_token_infos = self.basic_tokenizer.tokenize(text)
        # print (basic_token_infos)
        for sent_tok, sent_type, sent_word in zip(basic_token_infos['token'], basic_token_infos['type'],
                                                  basic_token_infos['word_indexs']):
            sen_token_list = []
            sen_token_id_list = []
            sen_token_type_list = []
            sen_token_index_list = []
            for w_tok, w_type, w_word in zip(sent_tok, sent_type, sent_word):
                if w_type in ['LETTERS', 'CHAR']:
                    bpe_toks = self.get_bpe_tokens(w_tok)
                    for bpe_tok in bpe_toks:
                        sen_token_list.append(bpe_tok[0])
                        sen_token_id_list.append(self.get_bpe_ids(bpe_tok[0]))
                        sen_token_type_list.append(w_type)
                        sen_token_index_list.append([w_word[0] + bpe_tok[1], w_word[0] + bpe_tok[2]])
                elif w_type == 'DIGIT':
                    w_tok = '<digit>'
                    sen_token_list.append(w_tok)
                    sen_token_type_list.append(w_type)
                    sen_token_index_list.append(w_word)
                    sen_token_id_list.append(self.get_bpe_ids(w_tok))
                elif w_type == 'SPACE':
                    w_tok = '<space>'
                    sen_token_list.append(w_tok)
                    sen_token_type_list.append(w_type)
                    sen_token_index_list.append(w_word)
                    sen_token_id_list.append(self.get_bpe_ids(w_tok))

                elif w_type in ['SPECIAL']:
                    sen_token_list.append(w_tok)
                    sen_token_type_list.append(w_type)
                    sen_token_index_list.append(w_word)
                    sen_token_id_list.append(self.get_bpe_ids(w_tok))

            output_token.append(sen_token_list)
            output_type.append(sen_token_type_list)
            output_word_indexes.append(sen_token_index_list)
            output_token_ids.append(sen_token_id_list)

        basic_token_infos['token'] = output_token
        basic_token_infos['type'] = output_type
        basic_token_infos['word_indexs'] = output_word_indexes
        basic_token_infos['token_ids'] = output_token_ids

        # print(basic_token_infos)
        return basic_token_infos

    # def get_bpe_ids(self, text):
    #     pieces = self.bpe_processor.encode_as_pieces(text)
    #     ids = self.bpe_processor.encode_as_ids(text)
    #     output = []
    #     for p, id in zip(pieces, ids):
    #         if p !='▁':
    #             output.append(id)
    #     return output[0]

    def get_bpe_ids(self, bpe_tok):
        return self.bpe_processor.piece_to_id(bpe_tok)

    def get_bpe_tokens(self, text):
        pieces = self.bpe_processor.encode_as_pieces(text)
        # ids = self.bpe_processor.encode_as_ids(text)
        output = []
        offset = 0
        for p in pieces:
            if p != '▁':
                token_length = len(p.replace('▁', ''))
                output.append((p, offset, offset + token_length))
                offset = offset + token_length
        return output


def is_overlapping(*parameter):
    if len(parameter) == 2:
        span_ind1, span_ind2 = parameter
        x1, x2 = span_ind1
        y1, y2 = span_ind2
    elif len(parameter) == 4:
        x1, x2, y1, y2 = parameter
    return max(x1, y1) < min(x2, y2)


#  获得ner的label及segment下标
def get_label_token_index(word_indexs, ner_label_index):
    label_token_index_list = []
    for label_index in ner_label_index:
        label_token_index = []
        span_index = [label_index['start_ind'], label_index['end_ind']]
        span_label = label_index['label_name']
        flat_word_indexs = [item for sublist in word_indexs for item in sublist]
        for word_index in flat_word_indexs:
            if is_overlapping(span_index, word_index):
                label_token_index.append(word_index)
        label_token_index_list.append([span_label, label_token_index])
    return label_token_index_list


# ner长度为1，则为S-label，否则变成B-Label, I-label, E-label
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


def get_sentence_token_label(label_data, tokenizer):
    label_index = label_data['ner']
    isbert = label_data['isbert']
    token_infos = tokenizer.tokenize(label_data['content'])
    word_indexs = token_infos['word_indexs']

    label_token_index_list = get_label_token_index(word_indexs, label_index)
    index_label_dict = convert_label_bieos(label_token_index_list)

    token_labels = [[index_label_dict.get(tuple(item), 'O') for item in sent_index] for sent_index in word_indexs]

    try:
        output = [{"token_labels": x, "token": y, "type": z, 'sents': s, "token_ids": t} for x, y, z, s, t in
                  zip(token_labels, token_infos['token'], token_infos['type'], token_infos['sents'],
                      token_infos['token_ids'])]
    except KeyError:
        output = [{"token_labels": x, "token": y, "type": z, 'sents': s} for x, y, z, s in
                  zip(token_labels, token_infos['token'], token_infos['type'], token_infos['sents'])]
    for d in output:
        d["isbert"] = isbert
    return output


def preprocess(raw_data, tokenizer, filter_others=True):
    token_labels = []
    for record in raw_data:
        sentence_token_labels = get_sentence_token_label(record, tokenizer)
        token_labels = token_labels + sentence_token_labels
    if filter_others:
        token_labels = [item for item in token_labels if set(item['token_labels']) != {'O'}]
    return token_labels


def item_list_overlap(item, list_item):
    flag = 0
    for i in list_item:
        if is_overlapping(i[0], i[1], item[0], item[1]):
            flag = 1
    return flag


def get_negative_sents(token_info, ner_x, ner_y):
    output = []
    sent_indexs = token_info['sent_indexs']
    sents = token_info['sents']
    label_indexs = [[item['start_ind'], item['end_ind']] for item in ner_x] + [[item['start_ind'], item['end_ind']]
                                                                               for
                                                                               item in ner_y]
    for sent, sent_ind in zip(sents, sent_indexs):
        if sent.rstrip():
            if not item_list_overlap(sent_ind, label_indexs):
                output.append(sent)
    return output


def get_negative_samples(group_1_file, group_2_file, tokenizer):
    raw_data_1 = get_raw_data(group_1_file)
    raw_data_2 = get_raw_data(group_2_file)
    raw_data_1 = get_emoji_index_align(raw_data_1)
    raw_data_2 = get_emoji_index_align(raw_data_2)

    df1 = pd.DataFrame(raw_data_1)
    df2 = pd.DataFrame(raw_data_2)

    df_merge = df1.merge(df2, on='docid')

    df_merge['tokens'] = df_merge['content_x'].apply(lambda x: tokenizer.tokenize(x))

    negative_sents = []
    for i in range(len(df_merge)):
        negative_sents = negative_sents + get_negative_sents(df_merge['tokens'][i], df_merge['ner_x'][i],
                                                             df_merge['ner_y'][i])
    return negative_sents


def get_negative_token_label(sents, tokenizer):
    output = []
    token_infos_list = [tokenizer.tokenize(sent) for sent in sents]
    for token_infos in token_infos_list:
        try:
            for x, y, z, t in zip(token_infos['token'], token_infos['type'], token_infos['sents'],
                                  token_infos['token_ids']):
                label = ['O' for i in range(len(x))]
                output.append({"token_labels": label, "token": x, "type": y, 'sents': z, "token_ids": t})
        except KeyError:
            for x, y, z in zip(token_infos['token'], token_infos['type'], token_infos['sents']):
                label = ['O' for i in range(len(x))]
                output.append({"token_labels": label, "token": x, "type": y, 'sents': z})
    return output


def get_class_labels(token_data):
    labels = []
    for t in token_data:
        labels = labels + list(set(t['token_labels']))
    return sorted(list(set(labels)))


def article_to_sentence_label(label_data, tokenizer):
    output = []
    label_index = label_data['ner']
    token_infos = tokenizer.tokenize(label_data['content'])
    # print (token_infos)
    for sent, sent_ind in zip(token_infos['sents'], token_infos['sent_indexs']):
        # print (sent)
        # print (sent_ind)
        tmp_sent_index = []
        for item in label_index:
            if is_overlapping([item['start_ind'], item['end_ind']], sent_ind):
                start_ind = item['start_ind'] - sent_ind[0]
                end_ind = item['end_ind'] - sent_ind[0]
                tmp_sent_index.append({'text_segment': item['text_segment'],
                                       'label_name': item['label_name'],
                                       'start_ind': start_ind,
                                       'end_ind': end_ind})
        if tmp_sent_index:
            output.append({"content": sent, "ner": tmp_sent_index})
    return output


def article_to_sentence_label_list(label_data_list, tokenizer):
    output = []
    for label_data in label_data_list:
        output = output + article_to_sentence_label(label_data, tokenizer)
    return output


def train_bpe(file, token_ids=['char', '5000', '8000']):
    train_data = read_json(file)
    save_text([item['content'].rstrip().lower() for item in train_data], '../vocab/train_data_bpe_corpus.txt')

    spm.SentencePieceTrainer.train(
        '--input=../vocab/train_data_bpe_corpus.txt --normalization_rule_name=identity --model_type=char '
        '--model_prefix=../vocab/ner_data_sents-{} --pad_id=0 --unk_id=3  '
        '--character_coverage=0.9995 --user_defined_symbols=<digit>,<space>'.format(token_ids[0]))
    spm.SentencePieceTrainer.train(
        '--input=../vocab/train_data_bpe_corpus.txt --normalization_rule_name=identity --model_type=bpe '
        '--model_prefix=../vocab/ner_data_sents-{} --pad_id=0 --unk_id=3  '
        '--character_coverage=0.9995 --user_defined_symbols=<digit>,<space> --vocab_size={}'.format(token_ids[1],
                                                                                                    token_ids[1]))
    spm.SentencePieceTrainer.train(
        '--input=../vocab/train_data_bpe_corpus.txt --normalization_rule_name=identity --model_type=bpe '
        '--model_prefix=../vocab/ner_data_sents-{} --pad_id=0 --unk_id=3  '
        '--character_coverage=0.9995 --user_defined_symbols=<digit>,<space> --vocab_size={}'.format(token_ids[2],
                                                                                                    token_ids[2]))


def generate_final_data(input_file, output_file, tokenizer_model, filter_others):
    tokenizer = FullTokenizer(bpe_model_file=tokenizer_model, do_lower_case=True,
                              normalize=False)
    train_data_raw = read_json(input_file)
    train_token_labels = preprocess(train_data_raw, tokenizer, filter_others=filter_others)
    train_token_labels = [item for item in train_token_labels if item['token']]

    save_json(train_token_labels, output_file)


