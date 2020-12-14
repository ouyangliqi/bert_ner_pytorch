# bert_ner_pytorch

This project is for bert ner model. Still working on.

dataset 

```json
{
"content":".....",
"ner":[{
	"text_segment":"",
    "label_name":"",
    "start_ind":"",
    "end_ind": ""}, 
]
}
```

model

| Model          | params                         | macro avg precision | macro avg recall | macro avg f1 score |
| -------------- | ------------------------------ | ------------------- | ---------------- | ------------------ |
| BERT + Softmax | {<br />"num_epochs": 5,<br />} |                     |                  |                    |
| BERT + CRF     | {}                             |                     |                  |                    |

