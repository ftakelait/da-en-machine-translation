This repository contains code for the two models: RoBerta-base--model and transformer-base-model.

Transformer Base Model:
1. To train and create tokenizers for the model from scratch run the following code as as example in the "transformer-base-model" directory:
_python cli/create_tokenizer.py --dataset_name stas/wmt14-en-de-pre-processed --dataset_config ende --vocab_size 32_000  --save_dir en_de_output_dir --source_lang da --target_lang en _

2. Once the tokenizers are created run the following code as an example: _python cli/train.py --dataset_name stas/wmt14-en-de-pre-processed --dataset_config ende --source_lang en --target_lang de --output_dir en_de_output_dir --batch_size 32 --num_warmup_steps 5000 --learning_rate 3e-4 --num_train_epochs 1 --eval_every 5000.
