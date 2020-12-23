# -*- coding: utf-8 -*-
# @Time    : 12/11/2020 2:56 PM
# @Author  : Chloe Ouyang
# @FileName: model_util.py
import numpy as np
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOBES
import torch


def from_state_dict(model, pretrain_state_dict):
    state_dict = model.state_dict()
    new_pretrain_state_dict = dict()
    for key in list(pretrain_state_dict):
        if key.startswith("bert."):
            # logger.info(f"from_state_dict: Dropping {key}")
            new_pretrain_state_dict[key] = pretrain_state_dict[key]
    state_dict.update(new_pretrain_state_dict)
    model.load_state_dict(state_dict)
    return model


def get_evaluation_result(prediction_chunk, label_chunk):
    final_evaluate = dict()
    correct_preds = dict()
    total_preds = dict()
    total_correct = dict()
    all_keys = []
    for lab_pred_chunks, lab_chunks in zip(prediction_chunk, label_chunk):
        keys = list(set([item[-1] for item in lab_pred_chunks] + [item[-1] for item in lab_chunks]))
        all_keys = all_keys + keys
        for key in keys:
            key_lab_pred_chunks = set([item for item in lab_pred_chunks if item[-1] == key])
            key_lab_chunks = set([item for item in lab_chunks if item[-1] == key])

            correct_preds[key] = correct_preds.get(key, 0) + len(key_lab_chunks & key_lab_pred_chunks)
            total_preds[key] = total_preds.get(key, 0) + len(key_lab_pred_chunks)
            total_correct[key] = total_correct.get(key, 0) + len(key_lab_chunks)

    all_keys = list(set(all_keys))
    for key in all_keys:
        p = correct_preds.get(key, 0) / total_preds.get(key, 0) if total_preds.get(key, 0) > 0 else 0
        r = correct_preds.get(key, 0) / total_correct.get(key, 0) if total_correct.get(key, 0) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        final_evaluate[key] = ("%.4f" % f1, "%.4f" % p, "%.4f" % r, total_correct.get(key, 0))

    return final_evaluate


def ner_index(output):
    from seqeval.scheme import IOBES
    from seqeval.scheme import Entities

    def get_entity_seqeval_strcit_iobes(tag):
        """Gets entities from sequence.
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            seq = ['B-PER', 'E-PER', 'O', 'S-LOC']
            get_entity_seqeval_strcit_iobes(seq)
            #output
            [(0, 2, 'PER'), (3, 4, 'LOC')]
        """
        chunks = Entities([tag], IOBES).entities
        return [(c.start, c.end, c.tag) for c in chunks[0]]

    return get_entity_seqeval_strcit_iobes([i for i in output if i != 'X'])


def evaulate(params, model, iter_data):
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
                    if l != 0:
                        preds.append(p)
                        labels.append(l)
                assert len(preds) == len(labels)
                predictions.append(preds)
                true_labels.append(labels)

    assert len(predictions) == len(true_labels)
    id2tag = params["id2tag"]
    pred_tags = [[id2tag[p_i] for p_i in p] for p in predictions]
    valid_tags = [[id2tag[l_i] for l_i in l] for l in true_labels]
    return pred_tags, valid_tags


def run_eval_classification_report(params, model, iter_data):
    pred_tags, valid_tags = evaulate(params, model, iter_data)
    print(classification_report(valid_tags, pred_tags, mode='strict', scheme=IOBES, digits=4))


def run_eval_classification_report_dict(params, model, iter_data):
    pred_tags, valid_tags = evaulate(params, model, iter_data)
    return classification_report(valid_tags, pred_tags, mode='strict', scheme=IOBES, digits=4, output_dict=True)


def run_eval_f1_score(params, model, iter_data):
    pred_tags, valid_tags = evaulate(params, model, iter_data)
    return f1_score(valid_tags, pred_tags, mode='strict', scheme=IOBES)
