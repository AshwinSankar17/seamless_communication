# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import random
import numpy as np

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from seamless_communication.cli.m4t.finetune import dataloader, dist_utils, trainer
from seamless_communication.models.unity import (
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
)

logger = logging.getLogger("finetune")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example finetuning script for M4T models"
    )
    parser.add_argument(
        "--train_dataset",
        nargs='+',
        type=str,
        required=True,
        help="Path to manifest with train samples",
    )
    parser.add_argument(
        "--eval_dataset",
        nargs='+',
        type=str,
        required=True,
        help="Path to manifest with eval samples",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="seamlessM4T_medium",
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)",
    )
    parser.add_argument(
        "--is_v1",
        action="store_true",
        help="Use this flag if the seamless model is v1",
    )
    parser.add_argument(
        "--save_model_to",
        type=Path,
        required=True,
        help="Path to save best finetuned model",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1000,
        help=("Save model with frequency."),
    )
    parser.add_argument(
        "--freq_type",
        type=str,
        choices=['epoch', 'step'],
        default='step',
        help=("Save model every n steps."),
    )
    parser.add_argument(
        "--load_model_from",
        type=Path,
        required=False,
        help="Path of checkpoint to load the model from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Randomizer seed value",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "Set early termination after `patience` number of evaluations "
            "without eval loss improvements"
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help=("Max number of training epochs"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-7,
        help=("Finetuning learning rate"),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help=("Number of steps with linearly increasing learning rate"),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help=("Get eval loss after each `eval_steps` training steps "),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help=("Log inner loss after each `log_steps` training steps"),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help=("How many steps to accumulate gradients for before stepping optimizer"),
    )
    parser.add_argument(
        "--adam_8bit",
        action="store_true",
        help=("Use 8-bit Adam optimizer for training"),
    )
    parser.add_argument(
        "--max_src_tokens",
        type=int,
        default=7000,
        help=("Maximum number of src_tokens per batch, used to avoid GPU OOM and maximize the effective batch size"),
    )
    parser.add_argument(
        "--mode",
        type=trainer.FinetuneMode,
        choices=list(trainer.FinetuneMode),
        default=trainer.FinetuneMode.SPEECH_TO_TEXT,
        help=(
            "* `SPEECH_TO_SPEECH` -- finetune S2T and T2U parts of the model; "
            "* `TEXT_TO_SPEECH` -- finetune only T2U; "
            "* `SPEECH_TO_TEXT` -- finetune only S2T"
        ),
    )
    parser.add_argument(
        "--freeze_layers",
        nargs="*",
        required=False,
        default=None,
        # TODO: better description
        help=("A list of modules to freeze in the model. If empty, everything will be trained."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=("Device to fine-tune on. See `torch.device`."),
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="seamless_finetune",
        help=("Name of the Weights & Biases run."),
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help=("Weights & Biases entity name (team or user). If not specified, defaults to the logged-in user."),
    )
    return parser

def seed_everything(seed: int) -> None:
    """
    Seed all relevant random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use for all libraries.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch random module
    
    # If using CUDA, set deterministic flags for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    
    # Ensure deterministic behavior in cuDNN (may slightly reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seeding everything with seed: {seed}")


@record
def main() -> None:
    args = init_parser().parse_args()
    seed_everything(args.seed)
    
    dist_utils.init_distributed([logger, trainer.logger])
    # float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16
    float_dtype = torch.bfloat16
    if dist_utils.get_rank() == 0:
        text_tokenizer = load_unity_text_tokenizer(args.model_name)
        unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
        del text_tokenizer
        del unit_tokenizer
    dist.barrier()
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    
    finetune_params = trainer.FinetuneParams(
        model_name=args.model_name,
        finetune_mode=args.mode,
        save_model_path=args.save_model_to,
        device=torch.device(args.device),
        float_dtype=float_dtype,
        save_freq=args.save_freq,
        freq_type=args.freq_type,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        patience=args.patience,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adam_8bit=args.adam_8bit,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
    )
    
    logger.info(f"Finetune Params: {finetune_params}")
    
    if dist_utils.get_rank() == 0:
        model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
        del model
    dist.barrier()
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    if args.load_model_from:
        if os.path.exists(args.load_model_from):
            checkpoint = torch.load(args.load_model_from)
            model.load_state_dict(checkpoint['model'])
            
    assert model.target_vocab_info == text_tokenizer.vocab_info
    
    if (
        finetune_params.finetune_mode == trainer.FinetuneMode.SPEECH_TO_TEXT
        and model.t2u_model is not None
    ):
        model.t2u_model = None
    
    if model.text_encoder is not None:
        model.text_encoder = None
    
    # Put model on selected device
    model = model.to(finetune_params.device)

    # TODO: delete unused params to reduce GPU memory consumption
    train_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.train_batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            max_audio_length_sec=15.0,
            float_dtype=finetune_params.float_dtype,
        ),
        mode="train",
        dataset_manifest_path=args.train_dataset,
        max_src_tokens_per_batch=args.max_src_tokens,
        is_v1=args.is_v1)
    
    eval_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.eval_batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            max_audio_length_sec=15.0,
            float_dtype=finetune_params.float_dtype,
        ),
        mode="test",
        dataset_manifest_path=args.eval_dataset,
        is_v1=args.is_v1)
    
    finetune = trainer.UnitYFinetune(
        model=model,
        params=finetune_params,
        train_data_loader=train_dataloader,
        eval_data_loader=eval_dataloader,
        checkpoint=checkpoint if args.load_model_from else None,
        freeze_modules=args.freeze_layers)
    
    finetune.run()


if __name__ == "__main__":
    main()
