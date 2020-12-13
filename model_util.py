# -*- coding: utf-8 -*-
# @Time    : 12/11/2020 2:56 PM
# @Author  : Chloe Ouyang
# @FileName: model_util.py
import numpy as np
import torch


def from_state_dict(model, pretrain_state_dict):
    state_dict = model.state_dict()
    for key in list(pretrain_state_dict):
        if not key.startswith("bert."):
            # logger.info(f"from_state_dict: Dropping {key}")
            del pretrain_state_dict[key]
    state_dict.update(pretrain_state_dict)
    model.load_state_dict(state_dict)
    return model


def run_quick_evaluate(params, model, iter_data):
    device = params["device"]
    id2tag = params["id2tag"]

    model = model.eval()
    predictions, true_labels = [], []
    ps, rs, f1s, s = [], [], [], []
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

                pred_tags = [id2tag[p] for p in preds]
                valid_tags = [id2tag[l] for l in labels]

                lab_pred_chunks = set(ner_index(pred_tags))
                lab_chunks = set(ner_index(valid_tags))

                predictions.append(lab_pred_chunks)
                true_labels.append(lab_chunks)
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

                valid_preds = [id2tag[p] for p in preds]
                valid_tags = [id2tag[l] for l in labels]

                lab_pred_chunks = set(ner_index(valid_preds))
                lab_chunks = set(ner_index(valid_tags))

                predictions.append(lab_pred_chunks)
                true_labels.append(lab_chunks)

    res = get_evaluation_result(predictions, true_labels)
    weights_map = {
        'Activity': 0.1,
        'Brand': 0.1,
        'Channel': 0.1,
        'Location': 0.1,
        'Money': 0.1,
        'Person': 0.1,
        'Product': 0.3,
        'Time': 0.1
    }
    weights = []
    for type_name, values in res.items():
        f1 = float(values[0])
        p = float(values[1])
        r = float(values[2])

        ps.append(p)
        rs.append(r)
        f1s.append(f1)

        weights.append(weights_map[type_name])
    return np.average(ps, weights=weights), np.average(rs, weights=weights), np.average(f1s, weights=weights)


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
    # dict_name = {v: k for k, v in name_dict.items()}
    result = []
    single_result = []
    for i in range(len(output)):
        if output[i] == 'O':
            if len(single_result) != 0:
                result.append((single_result[0][0],
                               single_result[-1][0] + 1,
                               single_result[0][1].split('-')[1]))
                single_result = []
        elif ('B-' in output[i]) or ('S-' in output[i]):
            if len(single_result) != 0:
                result.append((single_result[0][0],
                               single_result[-1][0] + 1,
                               single_result[0][1].split('-')[1]))
            single_result = []
            single_result.append([i, output[i]])
        elif ('I-' in output[i]) or ('E-' in output[i]):
            if len(single_result) != 0:
                single_result.append([i, output[i]])
    if len(single_result) != 0:
        result.append((single_result[0][0],
                       single_result[-1][0] + 1,
                       single_result[0][1].split('-')[1]))
    return result


def evaluate(params, predictions, true_labels):
    id2tag = params["id2tag"]

    assert len(predictions) == len(true_labels)
    chunk_pred_labels = []
    chunk_true_labels = []
    correct_preds, total_correct, total_preds = 0., 0., 0.

    for pred, lab in zip(predictions, true_labels):
        #         sent = tokenizer.convert_ids_to_tokens(sentences[index], skip_special_tokens=False)
        #         sent = [token for token in sent if token not in ['[CLS]', '[SEP]', '[PAD]', '[MASK]']]
        #         raw_text = ''.join(sent)
        pred_tags = [id2tag[p] for p in pred]
        true_tags = [id2tag[l] for l in lab]

        pred_chunks = set(ner_index(pred_tags))
        true_chunks = set(ner_index(true_tags))

        chunk_pred_labels.append(pred_chunks)
        chunk_true_labels.append(true_chunks)

        correct_preds += len(true_chunks & pred_chunks)
        total_preds += len(pred_chunks)
        total_correct += len(true_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    res = get_evaluation_result(chunk_pred_labels, chunk_true_labels)
    table = my_classification_report(res, (p, r, f1))
    print(table)


def my_classification_report(res, micro, digits=2):
    name_width = 0
    for e in res.keys():
        name_width = max(name_width, len(e))

    last_line_heading = 'macro avg'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[0])}
    for type_name, values in res.items():
        f1 = float(values[0])
        p = float(values[1])
        r = float(values[2])
        nb_true = int(values[3])
        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    #     compute averages
    #     report += row_fmt.format('micro avg',
    #                              micro[0],
    #                              micro[1],
    #                              micro[2],
    #                              np.sum(s),
    #                              width=width, digits=digits)
    report += row_fmt.format(last_line_heading,
                             np.average(ps),
                             np.average(rs),
                             np.average(f1s),
                             np.sum(s),
                             width=width, digits=digits)
    return report
