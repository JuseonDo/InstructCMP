# InstructCMP: Length Control in Sentence Compression through Instruction-based Large Language Models

* Authors: Juseon-Do, Jingun Kwon, Hidetaka Kamigaito, Manabu Okumura
* Paper Link: [InstructCMP](https://arxiv.org/abs/2406.11097)


## InstructCMP Dataset
Dataset folder has the following structure:
```
InstructCMP
├── dataset folder
│   ├── Google
│   │   ├──google_test.jsonl
│   │   ├──google_valid.jsonl
│   │   └──google_train.jsonl
|   |
│   ├── Broadcast
│   │   └──broadcast_test.jsonl
|   |
│   ├── BNC
│   │   └──bnc_test.jsonl
|   |
│   └── DUC2004
│       └──duc2004_test.jsonl
|
├── src
│   ├── evaluate_utils
│   │   evaluate_functions.py
|   |
│   ├── inference_utils
│   │   └──functions.py
|   |
│   └── utils
|      └──templates.py
|
└── run.py
```

## Models



# Evaluation
The metrics used in this work are listed in [evaluation_metrics](https://github.com/JuseonDo/InstructCMP/evaluation). For each metric, we have steps.txt which presents the steps to setup and run the metric.
# Contact
If you have any questions about this work, please contact **Juseon-Do** using the following email addresses: **dojuseon@gmail.com** or **doju00@naver.com**. 

