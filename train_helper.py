# -*- coding: utf-8 -*-
# @Time    : 10/7/2020 3:08 PM
# @Author  : Chloe Ouyang
# @FileName: train_helper.py
from tqdm import tqdm
import torch


def train_model(model, train_iter, eval_iter, params):
    num_train_optimization_steps = int(len(train_examples) / params['batch_size']) * params['n_epochs']

    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())

    optimizer_grouped_parameters = [{"params": [p for n, p in bert_param_optimizer], "lr": 2e-5},
                                    {"params": [p for n, p in crf_param_optimizer], "lr": 28 * (2e-5)}]

    warmup_steps = int(0.1 * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=params['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    def train_step(model, optimizer, scheduler):
        model.train()
        max_grad_norm = 1.0
        device = params['device']
        best_acc = 0

        for _ in range(params['n_epochs']):
            tr_loss = 0
            nb_tr_steps = 0
            for step, batch in enumerate(tqdm(train_iter)):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch

                if step == 0 and _ == 0:
                    print("=====sanity check======")
                    print("x:", b_input_ids)
                    print("mask:", b_input_mask)
                    print("tags:", b_labels)
                    print("=======================")

                output = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels,
                               label_masks=b_label_masks)

                loss = output[0]
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                # update parameters
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        # eval
        test_f1 = run_quick_evaluate(eval_iter, model, device)
        if test_f1 > best_acc:
            torch.save(model.state_dict(), params['model_save'])
        torch.save(model.state_dict(), params['model_save'])
        print(f'Epoch: {_}, Loss:  {loss.item()}')

