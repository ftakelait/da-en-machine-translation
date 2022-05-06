This repository contains code for the two models: Roberta-model and scratch-model.

1. To run train and create a tokenizers for the model from scratch run the following code in the "transformer-base-model" directory:
python cli/create_tokenizer.py --dataset_name stas/wmt14-en-de-pre-processed --dataset_config ende --vocab_size 32_000  --save_dir en_de_output_dir --source_lang da --target_lang en 
