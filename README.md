# Danish-English Neural Machine Translation
This repository contains code for the two models: RoBerta-base--model and transformer-base-model.

Transformer Base Model:
1. To train and create tokenizers for the model from scratch run the following code as as example in the "transformer-base-model" directory:
 `python cli/create_tokenizer.py --vocab_size 32000  --save_dir da_en_output_dir --source_lang da --target_lang en`.
2. Once the tokenizers are created run the following code as an example: `python cli/train.py --dataset_name stas/wmt14-en-de-pre-processed --dataset_config ende --source_lang en --target_lang de --output_dir en_de_output_dir --batch_size 32 --num_warmup_steps 5000 --learning_rate 3e-4 --num_train_epochs 1 --eval_every 5000`.


RoBerta-base--model:
1. To use the pre-trained RoBerta model, first create the target tokenizer by running the following code as an example: `python cli/create_target_tokenizer.py  --vocab_size 32000  --save_dir en_output_dir --target_lang en` 
2. Once the target language's tokenizer is created run the following code as an example: `python cli/train_final.py --source_lang da --target_lang en --output_dir en_output_dir --batch_size 32 --num_warmup_steps 5000 --learning_rate 3e-4 --num_train_epochs 1 --eval_every 5000`.

