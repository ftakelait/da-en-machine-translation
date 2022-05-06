# Danish-English Neural Machine Translation

## Introduction

In this work, we apply two transfoermer models to the EuroParl Danish-English dataset for a translation task. This repository contains code for the two models: RoBerta-base-model and transformer-base-model.

## Installation

You can install the package via

```bash
pip install git+https://github.com/kev-zhao/life-after-bert
```

Or **(recommended)** you can download the source code and install the package in editable mode for each model directory:

```bash
git clone https://github.com/kev-zhao/life-after-bert
cd life-after-bert
pip install -e .
```

## Transformer Base Model:

1. First, you need to create tokenizers for both models, run the following code for **transformer-base-model** directories:

```bash
python cli/create_tokenizer.py --vocab_size 32000  --save_dir da_en_output_dir --source_lang da --target_lang en`
```

2. Once the tokenizers are created run the following code to train your models: 
```bash
python cli/train.py --dataset_name stas/wmt14-en-de-pre-processed --dataset_config ende --source_lang en --target_lang de --output_dir en_de_output_dir --batch_size 32 --num_warmup_steps 5000 --learning_rate 3e-4 --num_train_epochs 1 --eval_every 5000
```

## RoBerta-base-model:

1. To use the pre-trained RoBERTa model, first create the target tokenizer by running the following code for **roberta-base-danish** directory: 
```bash
python cli/create_target_tokenizer.py  --vocab_size 32000  --save_dir en_output_dir --target_lang en
```

2. Once the target language's tokenizer is created run the following code to train your model: 
```bash
python cli/train_final.py --source_lang da --target_lang en --output_dir en_output_dir --batch_size 32 --num_warmup_steps 5000 --learning_rate 3e-4 --num_train_epochs 1 --eval_every 5000
```

