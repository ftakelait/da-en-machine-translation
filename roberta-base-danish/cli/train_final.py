

# Python standard library imports
import argparse
import logging
import math
import os
import random
from functools import partial
from packaging import version

# Import from third party libraries
import datasets
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
import transformers
from datasets import load_from_disk

# Imports from our module
from transformer_mt.modeling_transformer_final import TransfomerEncoderDecoderModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

from transformer_mt import utils


# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_warning()


# To compute BLEU we will use Huggingface Datasets implementation of it
# Sacrebleu is a flavor of BLEU that standardizes some of the BLEU parameters.
bleu = datasets.load_metric("sacrebleu")


def parse_args():

    parser = argparse.ArgumentParser(description="Train machine translation transformer model")

    # Required arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=("Where to store the final model. "
              "Should contain the source and target tokenizers in the following format: "
              r"output_dir/{source_lang}_tokenizer and output_dir/{target_lang}_tokenizer. "
              "Both of these should be directories containing tokenizer.json files."
        ),
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        required=True,
        help="Source language id for translation.",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="Target language id for translation.",
    )
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="code_x_glue_tt_text_to_text",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="da_en",
        help=("Many datasets in Huggingface Dataset repository have multiple versions or configs. "
              "For the case of machine translation these usually indicate the language pair like "
              "en-es or zh-fr or similar. To look up possible configs of a dataset, "
              "find it on huggingface.co/datasets."),
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Whether to use a small subset of the dataset for debugging.",
    )
    # Model arguments
    parser.add_argument(
        "--num_layers",
        default=6,
        type=int,
        help="Number of hidden layers in the Transformer encoder",
    )
    parser.add_argument(
        "--hidden_size",
        default=768,
        type=int,
        help="Hidden size of the Transformer encoder",
    )
    parser.add_argument(
        "--num_heads",
        default=8,
        type=int,
        help="Number of attention heads in the Transformer encoder",
    )
    parser.add_argument(
        "--fcn_hidden",
        default=2048,
        type=int,
        help="Hidden size of the FCN",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum total sequence length for source and target texts after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=None,
        help="Overwrite the cached training and evaluation sets",
    )

    # Training arguments
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu) on which the code should run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout rate of the Transformer encoder",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=5000,
        help="Perform evaluation every n network updates.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Compute and log training batch metrics every n steps.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=transformers.SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--generation_type",
        choices=["greedy", "beam_search"],
        default="beam_search",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help=("Beam size for beam search generation. "
              "Decreasing this parameter will make evaluation much faster, "
              "increasing this (until a certain value) would likely improve your results."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--wandb_project", 
        default="transformer_mt",
        help="wandb project name to log metrics to"
    )

    args = parser.parse_args()

#     if f"{args.source_lang}_tokenizer" not in os.listdir(args.output_dir):
#         raise ValueError(f"The source tokenizer is not found in {args.output_dir}")
    
#     if f"{args.target_lang}_tokenizer" not in os.listdir(args.output_dir):
#         raise ValueError(f"The target tokenizer is not found in {args.output_dir}")

    return args


def preprocess_function(
    examples,
    source_lang,
    target_lang,
    max_seq_length,
    source_tokenizer,
    target_tokenizer,
):
#     inputs = examples['source']
#     targets = examples['target']
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]

    model_inputs = source_tokenizer(inputs, max_length=max_seq_length, truncation=True)

    targets = target_tokenizer(targets, max_length=max_seq_length - 1, truncation=True)
    target_ids = targets["input_ids"]


    decoder_input_ids = []
    labels = []
    for target in target_ids:
        decoder_input_ids.append([target_tokenizer.bos_token_id] + target)
        labels.append(target + [target_tokenizer.eos_token_id])
        
    for target in target_ids:
        decoder_input_ids.append( target)
        labels.append(target)


    model_inputs["decoder_input_ids"] = decoder_input_ids
    model_inputs["labels"] = labels

    return model_inputs


def collation_function_for_seq2seq(batch, source_pad_token_id, target_pad_token_id):

    input_ids_list = [ex["input_ids"] for ex in batch]
    decoder_input_ids_list = [ex["decoder_input_ids"] for ex in batch]
    labels_list = [ex["labels"] for ex in batch]

    collated_batch = {
        "input_ids": utils.pad(input_ids_list, source_pad_token_id),
        "decoder_input_ids": utils.pad(decoder_input_ids_list, target_pad_token_id),
        "labels": utils.pad(labels_list, target_pad_token_id),
    }

    collated_batch["encoder_padding_mask"] = collated_batch["input_ids"] == source_pad_token_id
    return collated_batch


def evaluate_model(
    model,
    dataloader,
    *,
    target_tokenizer,
    device,
    max_seq_length,
    generation_type,
    beam_size,
):
    n_generated_tokens = 0
    model.eval()
    for batch in tqdm(dataloader, desc="Evaluation"):
        with torch.inference_mode():
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            key_padding_mask = batch["encoder_padding_mask"].to(device)


            generated_tokens = model.generate(
                input_ids,
                bos_token_id=target_tokenizer.bos_token_id,
                eos_token_id=target_tokenizer.eos_token_id,
                pad_token_id=target_tokenizer.pad_token_id,
                key_padding_mask=key_padding_mask,
                max_length=max_seq_length,
                kind=generation_type,
                beam_size=beam_size,
            )
            decoded_preds = target_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = target_tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred in decoded_preds:
                n_generated_tokens += len(target_tokenizer(pred)["input_ids"])

            decoded_preds, decoded_labels = utils.postprocess_text(decoded_preds, decoded_labels)

            bleu.add_batch(predictions=decoded_preds, references=decoded_labels)

    model.train()
    eval_metric = bleu.compute()
    evaluation_results = {
        "bleu": eval_metric["score"],
        "generation_length": n_generated_tokens / len(dataloader.dataset),
    }
    return evaluation_results, input_ids, decoded_preds, decoded_labels


def main():
    # Parse the arguments
    args = parse_args()
    logger.info(f"Starting script with arguments: {args}")

    wandb.init(project=args.wandb_project, config=args)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the datasets
    raw_datasets = load_from_disk("danish_english_dataset_updated")
#     raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    if "validation" not in raw_datasets:

        raw_datasets = raw_datasets["train"].train_test_split(test_size=2000, seed=42)

    if args.debug:
        raw_datasets = utils.sample_small_debug_dataset(raw_datasets)

    ###############################################################################
    # Part 2: Create the model and load the tokenizers
    ###############################################################################

    tgt_tokenizer_path = os.path.join(args.output_dir, f"{args.target_lang}_tokenizer")

    source_tokenizer = AutoTokenizer.from_pretrained("flax-community/roberta-base-danish")

    target_tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tgt_tokenizer_path)
  
    encoder = AutoModelForMaskedLM.from_pretrained("flax-community/roberta-base-danish")



    model = TransfomerEncoderDecoderModel(num_layers = args.num_layers, encoder_model = encoder,
        num_heads = args.num_heads,
        hidden =  args.hidden_size,
        fcn_hidden = args.fcn_hidden,                                  
        max_seq_len = args.max_seq_length,
        src_vocab_size = source_tokenizer.vocab_size,
        tgt_vocab_size = target_tokenizer.vocab_size,                                                                   
        dropout = args.dropout_rate).to(args.device)

    column_names = raw_datasets["train"].column_names


    preprocess_function_wrapped = partial(
        preprocess_function,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_seq_length=args.max_seq_length,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )

    processed_datasets = raw_datasets.map(
        preprocess_function_wrapped,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"] if "validaion" in processed_datasets else processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 2):
       
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        logger.info(f"Decoded input_ids: {source_tokenizer.decode(train_dataset[index]['input_ids'])}")
        logger.info(f"Decoded labels: {target_tokenizer.decode(train_dataset[index]['labels'])}")
        logger.info("\n")

    ###############################################################################
    # Part 4: Create PyTorch dataloaders that handle data shuffling and batching
    ###############################################################################

    collation_function_for_seq2seq_wrapped = partial(
        collation_function_for_seq2seq,
        source_pad_token_id=source_tokenizer.pad_token_id,
        target_pad_token_id=target_tokenizer.pad_token_id,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collation_function_for_seq2seq_wrapped, batch_size=args.batch_size, num_workers = args.preprocessing_num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=collation_function_for_seq2seq_wrapped, batch_size=args.batch_size, num_workers = args.preprocessing_num_workers
    )
    # YOUR CODE ENDS HERE

    ###############################################################################
    # Part 5: Create optimizer and scheduler
    ###############################################################################

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps))

    # Log a pre-processed training example to make sure the pre-processing does not have bugs in it
    # and we do not input garbage to our model.
    batch = next(iter(train_dataloader))
    logger.info("Look at the data that we input into the model, check that it looks like what we expect.")
    for index in random.sample(range(len(batch)), 2):
        logger.info(f"Decoded input_ids: {source_tokenizer.decode(batch['input_ids'][index])}")
        logger.info(f"Decoded labels: {target_tokenizer.decode(batch['labels'][index])}")
        logger.info("\n")

    ###############################################################################
    # Part 6: Training loop
    ###############################################################################
    global_step = 0

    # iterate over epochs
    for epoch in range(args.num_train_epochs):
        model.train()  # make sure that model is in training mode, e.g. dropout is enabled

        # iterate over batches
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(args.device)
            decoder_input_ids = batch["decoder_input_ids"].to(args.device)
            key_padding_mask = batch["encoder_padding_mask"].to(args.device)
            
            
            labels = batch["labels"].to(args.device)

            logits = model(
                input_ids,
                decoder_input_ids=decoder_input_ids,
                key_padding_mask=key_padding_mask,
            )
#             print("Input" , input_ids.shape)
#             print("after train",  logits.view(-1, logits.shape[-1]).shape,  labels.view(-1).shape  )
            

            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=target_tokenizer.pad_token_id,
            )

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            wandb.log(
                {
                    "train_loss": loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            if global_step % args.logging_steps == 0:
           
                predictions = logits.argmax(-1)
                label_nonpad_mask = labels != target_tokenizer.pad_token_id
                num_words_in_batch = label_nonpad_mask.sum().item()

                accuracy = (predictions == labels).masked_select(label_nonpad_mask).sum().item() / num_words_in_batch

                wandb.log(
                    {"train_batch_word_accuracy": accuracy},
                    step=global_step,
                )

            if global_step % args.eval_every_steps == 0 or global_step == args.max_train_steps:
                eval_results, last_input_ids, last_decoded_preds, last_decoded_labels = evaluate_model(
                    model=model,
                    dataloader=eval_dataloader,
                    target_tokenizer=target_tokenizer,
                    device=args.device,
                    max_seq_length=args.max_seq_length,
                    generation_type=args.generation_type,
                    beam_size=args.beam_size,
                )
                # YOUR CODE ENDS HERE
                wandb.log(
                    {
                        "eval/bleu": eval_results["bleu"],
                        "eval/generation_length": eval_results["generation_length"],
                    },
                    step=global_step,
                )
                logger.info("Generation example:")
                random_index = random.randint(0, len(last_input_ids) - 1)
                logger.info(f"Input sentence: {source_tokenizer.decode(last_input_ids[random_index], skip_special_tokens=True)}")
                logger.info(f"Generated sentence: {last_decoded_preds[random_index]}")
                logger.info(f"Reference sentence: {last_decoded_labels[random_index][0]}")

                logger.info("Saving model checkpoint to %s", args.output_dir)
                model.save_pretrained(args.output_dir)

            if global_step >= args.max_train_steps:
                break

    ###############################################################################
    # Part 8: Save the model
    ###############################################################################

    logger.info("Saving final model checkpoint to %s", args.output_dir)
    model.save_pretrained(args.output_dir)

    logger.info("Uploading tokenizer, model and config to wandb")
    wandb.save(os.path.join(args.output_dir, "*"))

    logger.info(f"Script finished succesfully, model saved in {args.output_dir}")


if __name__ == "__main__":
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")

    main()
