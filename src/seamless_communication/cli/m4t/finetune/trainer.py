# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


import logging
import time
import wandb
from contextlib import contextmanager
import dataclasses
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from fairseq2.data import VocabularyInfo
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn.padding import PaddingMask
from fairseq2.optim.lr_scheduler import MyleLR
from fairseq2.typing import Device
from torch.optim import AdamW

from seamless_communication.cli.m4t.finetune import dataloader, dist_utils
from seamless_communication.models.unity import (
    UnitYModel,
    UnitYT2UModel,
)

logger = logging.getLogger(__name__)


class FinetuneMode(Enum):
    SPEECH_TO_SPEECH = "SPEECH_TO_SPEECH"
    SPEECH_TO_TEXT = "SPEECH_TO_TEXT"
    TEXT_TO_SPEECH = "TEXT_TO_SPEECH"


@dataclass
class FinetuneParams:
    model_name: str
    """Model name of model being finetuned."""
    
    save_model_path: Path
    """Path were to save finetuned model."""
    
    finetune_mode: FinetuneMode = FinetuneMode.TEXT_TO_SPEECH
    """Allows to freeze S2T or T2U part of the model"""
    
    float_dtype: torch.dtype = torch.float16
    """Float Dtype"""

    save_freq: int = 1000
    """Save every n steps/epochs."""

    freq_type: str = 'step'
    """Save every step/epoch. save_freq gets updated if it is epoch."""

    max_epochs: int = 10
    """ Maximum number of trainign epochs"""

    label_smoothing: float = 0.2
    """ Label smoothing coefficient for nll_loss """

    warmup_steps: int = 100
    """ Number of steps with linearly increasing LR"""

    log_steps: int = 10
    """ Log inner loss after each `log_steps` training steps"""
    
    gradient_accumulation_steps: int = 8
    """ Log inner loss after each `log_steps` training steps"""

    eval_steps: int = 50
    """ Get eval loss after each `eval_steps` training steps """

    patience: int = 3
    """ Terminate if eval loss did not improve
    over the last `patience * eval_steps` training steps"""

    learning_rate: float = 1e-5
    """ Optimizer learining rate """

    train_batch_size: int = 5
    """The batch size during train steps"""

    eval_batch_size: int = 5
    """The batch size during evaluation."""

    device: Device = torch.device("cuda")
    """ Where to run computation"""
    
    entity: str = "indic-asr"
    """ Wandb project entity"""
    
    run_name: str = "seamlessm4t_v2_indic_stt"
    """ Wandb project entity"""


class UnitYFinetuneWrapper(nn.Module):
    """Convenience wrapper that does a forward pass
    and returns S2T and T2U logits"""

    def __init__(self, model: UnitYModel, mode: FinetuneMode, device: Device):
        super().__init__()
        self.model: UnitYModel = model
        self.freeze_s2t: bool = mode == FinetuneMode.TEXT_TO_SPEECH
        self.freeze_t2u: bool = mode == FinetuneMode.SPEECH_TO_TEXT
        logger.info(f"Freeze s2t: {self.freeze_s2t}, freeze t2u: {self.freeze_t2u}")
        self.device = device

    def forward(
        self, batch: dataloader.MultimodalSeqsBatch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dummy_context = contextmanager(lambda: iter([None]))()
        with torch.no_grad() if self.freeze_s2t else dummy_context:  # type:ignore
            assert batch.speech_to_text.src_tokens is not None
            seqs = batch.speech_to_text.src_tokens.to(self.device)
            assert batch.speech_to_text.src_lengths is not None
            seq_lens = batch.speech_to_text.src_lengths.to(self.device)
            speech_encoder_out, speech_encoder_padding_mask = self.model.encode_speech(
                seqs=seqs, padding_mask=PaddingMask(seq_lens, seqs.size(1))
            )
            assert batch.speech_to_text.prev_output_tokens is not None
            seqs = batch.speech_to_text.prev_output_tokens.to(self.device)
            assert batch.speech_to_text.target_lengths is not None
            seq_lens = batch.speech_to_text.target_lengths.to(self.device)
            text_decoder_out, text_decoder_padding_mask = self.model.decode(
                seqs=seqs,
                padding_mask=PaddingMask(seq_lens, seqs.size(1)),
                encoder_output=speech_encoder_out,
                encoder_padding_mask=speech_encoder_padding_mask,
            )
            assert self.model.final_proj is not None
            text_logits = self.model.final_proj(text_decoder_out)
        if self.freeze_t2u:
            return (text_logits, None)
        assert self.model.t2u_model is not None
        assert batch.text_to_units.prev_output_tokens is not None
        dummy_context = contextmanager(lambda: iter([None]))()
        with torch.no_grad() if self.freeze_t2u else dummy_context:  # type:ignore
            if not isinstance(self.model.t2u_model, UnitYT2UModel):
                raise NotImplementedError(
                    "T2U finetuning implemented only for UnitYT2UModel"
                )
            (
                unit_encoder_out,
                unit_encoder_padding_mask,
            ) = self.model.t2u_model.encode(
                seqs=text_decoder_out,
                padding_mask=text_decoder_padding_mask,
            )
            seqs = batch.text_to_units.prev_output_tokens.to(self.device)
            assert batch.text_to_units.target_lengths is not None
            seq_lens = batch.text_to_units.target_lengths.to(self.device)
            unit_decoder_out, _ = self.model.t2u_model.decode(
                seqs=seqs,
                padding_mask=PaddingMask(seq_lens, seqs.size(1)),
                encoder_output=unit_encoder_out,
                encoder_padding_mask=unit_encoder_padding_mask,
            )
            unit_logits = self.model.t2u_model.final_proj(unit_decoder_out)

        return (text_logits, unit_logits)


class CalcLoss:
    """Calculates negative log likelihood loss for S2T and T2U"""

    def __init__(
        self,
        label_smoothing: float,
        s2t_vocab_info: VocabularyInfo,
        t2u_vocab_info: Optional[VocabularyInfo],
    ):
        self.label_smoothing = label_smoothing
        self.s2t_vocab_info = s2t_vocab_info
        self.t2u_vocab_info = t2u_vocab_info

    def __call__(
        self,
        batch: dataloader.MultimodalSeqsBatch,
        text_logits: torch.Tensor,
        unit_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert batch.speech_to_text.target_lengths is not None
        prefix_skip_len = 1  # language tokens to skip
        s2t_numel = torch.sum(batch.speech_to_text.target_lengths - prefix_skip_len).to(
            text_logits.device
        )
        assert batch.speech_to_text.target_tokens is not None
        s2t_loss = SequenceModelOutput(
            logits=text_logits, vocab_info=self.s2t_vocab_info
        ).compute_loss(
            targets=batch.speech_to_text.target_tokens.to(text_logits.device),
            ignore_prefix_size=prefix_skip_len,
            label_smoothing=self.label_smoothing,
        )
        if unit_logits is None:
            return s2t_loss / s2t_numel
        assert batch.text_to_units.target_lengths is not None
        s2u_numel = torch.sum(batch.text_to_units.target_lengths - prefix_skip_len).to(
            unit_logits.device
        )
        assert batch.text_to_units.target_tokens is not None
        assert self.t2u_vocab_info is not None
        s2u_loss = SequenceModelOutput(
            logits=unit_logits, vocab_info=self.t2u_vocab_info
        ).compute_loss(
            targets=batch.text_to_units.target_tokens.to(unit_logits.device),
            ignore_prefix_size=prefix_skip_len,
            label_smoothing=self.label_smoothing,
        )
        return s2t_loss / s2t_numel + s2u_loss / s2u_numel


class LossCollector:
    """Aggregrates loss history across nodes"""

    def __init__(self, device: Optional[Device] = None, reduce_op: str = "avg"):
        self.n_samples: float = 0
        self.val_sum: float = 0.0
        self.reduce_op = reduce_op
        self.device = device
        self.is_distributed = dist_utils.is_dist_initialized()

    def reset(self) -> None:
        self.n_samples = 0
        self.val_sum = 0.0

    def update(self, n_samples: int, batch_loss: float) -> None:
        self.n_samples += n_samples
        self.val_sum += batch_loss

    def reduce(self) -> float:
        n_samples, val_sum = self._collect()
        if self.reduce_op == "avg":
            return val_sum / (n_samples + 1)
        if self.reduce_op == "sum":
            return val_sum
        raise ValueError()

    def _collect(self) -> Tuple[float, float]:
        if not self.is_distributed:
            return self.n_samples, self.val_sum
        local_val = torch.tensor([[self.n_samples, self.val_sum]], device=self.device)
        all_vals = [
            torch.zeros((1, 2), device=self.device)
            for _ in range(dist_utils.get_world_size())
        ]
        dist.all_gather(all_vals, local_val)
        losses = torch.concat(all_vals, dim=0)
        reduced = torch.sum(losses, dim=0).reshape(2).cpu()
        return reduced[0].item(), reduced[1].item()

class WandbLogger:
    def __init__(self, params, project_name: str = "STT_Translation"):
        print(params)
        self.is_main_process = dist_utils.is_main_process()
        if self.is_main_process:
            wandb.init(
                project=project_name,
                config=dataclasses.asdict(params),
                name=params.run_name,
                entity=params.entity,
            )

    def log(self, metrics: dict, step: int = None):
        if self.is_main_process:
            wandb.log(metrics, step=step)

    def finish(self):
        if self.is_main_process:
            wandb.finish()


class UnitYFinetune:
    def __init__(
        self,
        model: UnitYModel,
        params: FinetuneParams,
        train_data_loader: dataloader.UnitYDataLoader,
        eval_data_loader: Optional[dataloader.UnitYDataLoader] = None,
        checkpoint: Optional[Dict[str, torch.Tensor]] = None,
        freeze_modules: Optional[List[Union[str, torch.nn.Module]]] = None
    ):
        self.params = params
        self.calc_loss = CalcLoss(
            label_smoothing=self.params.label_smoothing,
            s2t_vocab_info=model.target_vocab_info,
            t2u_vocab_info=model.t2u_model.target_vocab_info
            if model.t2u_model is not None
            else None,
        )
        
        self.model = self._wrap_model_for_trainining(model=model)
        if freeze_modules:
            self._freeze_modules(freeze_modules)
        
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        
        self.grad_scaler = torch.cuda.amp.GradScaler()  # type: ignore
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.params.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-08,
            maximize=False,
            weight_decay=0.0,
            fused=(self.params.device.type == "cuda"),
        )
        self.lr_scheduler = MyleLR(
            optimizer=self.optimizer,
            num_warmup_steps=self.params.warmup_steps,
            start_lr=1e-9,
        )
        self.skip_batches = 0
        if checkpoint:
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'dataloader' in checkpoint:
                self.skip_batches = checkpoint['dataloader']['batch_idx']
                # self.train_dataloader.load_state_dict(checkpoint['dataloader']['train_data_loader'])
                # self.eval_dataloader.load_state_dict(checkpoint['dataloader']['eval_data_loader'])

        if self.params.freq_type == 'epoch':
            self.params.save_freq *= len(train_data_loader.data_loader)
        
        self.logger = WandbLogger(params=self.params, project_name="STT_Translation")
        self.train_loss_hist = LossCollector(device=params.device)
        self.epoch_idx: int = self.lr_scheduler.last_epoch
        self.update_idx: int = 0
        self.patience_left: int = self.params.patience
        self.best_eval_loss: Optional[float] = None
        self.is_best_state: bool = False
        torch.set_float32_matmul_precision("high")

    def _reset_stats(self) -> None:
        self.train_loss_hist.reset()
        # self.epoch_idx = 0
        self.update_idx = 0
        self.patience_left = self.params.patience
        self.best_eval_loss = None
        self.is_best_state = False

    def _wrap_model_for_trainining(self, model: UnitYModel) -> nn.Module:
        wrapped_model = UnitYFinetuneWrapper(
            model=model, mode=self.params.finetune_mode, device=self.params.device
        )
        if not dist_utils.is_dist_initialized():
            return wrapped_model
        find_unused = self.params.finetune_mode == FinetuneMode.TEXT_TO_SPEECH
        return nn.parallel.DistributedDataParallel(
            wrapped_model,
            device_ids=[dist_utils.get_local_rank()],
            find_unused_parameters=find_unused,
        )
        
    def _freeze_modules(self, frozen_modules: List[str] = []) -> None:
        for icecube in frozen_modules:
            for (name, module) in self.model.named_modules():
                if name.startswith(icecube):
                    logger.info(f"Freezing Module: {name}")
                    for param in module.parameters():
                        param.requires_grad = False

    def _update_eval_stats(self, eval_loss: float) -> None:
        self.is_best_state = (
            self.best_eval_loss is None or eval_loss < self.best_eval_loss
        )
        self.best_eval_loss = eval_loss if self.is_best_state else self.best_eval_loss
        self.patience_left = (
            self.params.patience if self.is_best_state else self.patience_left - 1
        )

        if self.logger.is_main_process:
            self.logger.log({"eval_loss": eval_loss, "best_eval_loss": self.best_eval_loss}, step=self.update_idx)

        logger.info(
            f"Eval after {self.update_idx} updates: "
            f"loss={eval_loss:.4f} "
            f"best_loss={self.best_eval_loss:.4f} "
            f"patience_steps_left={self.patience_left}"
        )

    @torch.no_grad()
    def _eval_model(self, n_batches: int) -> None:
        """Calc avg loss on eval dataset and update evaluation stats"""
        if self.eval_data_loader is None:
            return
        logger.info(f"Evaluation Step {self.update_idx // self.params.eval_steps}...")
        loss_hist = LossCollector(device=self.params.device)
        # loss_hist = []
        self.model.eval()
        for batch in self.eval_data_loader.data_loader:
            if n_batches == 0:
                break
            assert batch.speech_to_text.src_tokens is not None
            with torch.autocast(device_type=self.params.device.type, dtype=self.params.float_dtype):
                loss = self.calc_loss(batch, *self.model(batch))
            if loss.isnan():
                logger.warning("Eval batch loss value is NaN, skipping")
                continue
            del batch  # force memory release
            # loss_hist.append(loss.item())
            loss_hist.update(1, loss.item())
            n_batches -= 1
        # eval_loss = torch.tensor(loss_hist).to(loss).mean()
        # del loss_hist
        # dist.all_reduce(eval_loss, op=dist.ReduceOp.AVG)
        eval_loss = loss_hist.reduce()
        self._update_eval_stats(eval_loss)

    def _train_step_log(self) -> None:
        """Log train stats"""
        if (self.update_idx + 1) % self.params.log_steps == 0:
            avg_loss = self.train_loss_hist.reduce()
            self.train_loss_hist.reset()

            if self.logger.is_main_process:
                self.logger.log({"train_loss": avg_loss, "lr": self.lr_scheduler.get_last_lr()[0]}, step=self.update_idx)

            logger.info(
                f"Epoch {str(self.epoch_idx + 1).zfill(3)} / "
                f"update {str(self.update_idx + 1).zfill(5)}: "
                f"train loss={avg_loss:.4f} "
                f"last lr={self.lr_scheduler.get_last_lr()[0]:.2E}"
            )

    def _train_step(self, batch: List[dataloader.MultimodalSeqsBatch]) -> None:
        """Run one train step"""
        self.model.train()
        # self.optimizer.zero_grad()
        with torch.autocast(device_type=self.params.device.type, dtype=self.params.float_dtype):
            tokens, units = self.model(batch)
        
        loss = self.calc_loss(batch, tokens, units)
        if loss.isnan().any().item():
            logger.error(batch.speech_to_text)
            raise RuntimeError("Train loss is NaN! Something is wrong in the model!")
        
        # self.grad_scaler.scale(loss).backward()
        # self.grad_scaler.step(self.optimizer)
        # self.grad_scaler.update()
        # self.lr_scheduler.step()
        
        assert batch.speech_to_text.src_tokens is not None
        self.train_loss_hist.update(1, loss.item())
        # dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        self._train_step_log()
        self.update_idx += 1
        return loss

    def _save_model(self, batch_idx) -> None:
        logger.info("Saving model")
        if dist_utils.is_main_process():
            torch.save({
                "model_name": self.params.model_name,
                "model": {
                    key.replace("module.model.model.", ""): value
                    for key, value in self.model.state_dict().items()
                },
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "dataloader": {
                    "batch_idx": batch_idx
                }
                # "dataloader": {
                #     "train_data_loader": self.train_dataloader.state_dict() if self.train_data_loader else None,
                #     "eval_data_loader": self.eval_dataloader.state_dict() if self.eval_data_loader else None,
                # }
            }, self.params.save_model_path)
        if dist_utils.is_dist_initialized():
            dist.barrier()
    
    def _save_model_every_x_steps(self, step_idx, batch_idx) -> None:
        logger.info("Saving model")
        if not isinstance(self.params.save_model_path, Path):
            save_dir = Path(self.params.save_model_path).parent
        else:
            save_dir = self.params.save_model_path.parent
        
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        if dist_utils.is_main_process():
            torch.save({
                "model_name": self.params.model_name,
                "model": {
                    key.replace("module.model.model.", ""): value
                    for key, value in self.model.state_dict().items()
                },
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "dataloader": {
                    "batch_idx": batch_idx
                }
                # "dataloader": {
                #     "train_data_loader": self.train_dataloader.state_dict() if self.train_data_loader else None,
                #     "eval_data_loader": self.eval_dataloader.state_dict() if self.eval_data_loader else None,
                # }
            }, save_dir / f"checkpoint_step={step_idx}.pt")
        if dist_utils.is_dist_initialized():
            dist.barrier()

    def run(self) -> None:
        logger.info("Start Finetuning")
        self._reset_stats()
        self._eval_model(n_batches=100)
        train_dataloader = self.train_data_loader.data_loader
        try:
            step_idx = 0
            while self.epoch_idx < self.params.max_epochs and self.patience_left:
                batch_idx = 0
                for train_batch in tqdm(train_dataloader, desc="Training Steps"):
                    if batch_idx < self.skip_batches:
                        batch_idx += 1
                        continue
                    # Run batch through train step
                    loss = self._train_step(train_batch)
                    self.grad_scaler.scale(loss / self.params.gradient_accumulation_steps).backward()
                    
                    batch_idx += 1
                    step_idx += 1
                    if step_idx % self.params.gradient_accumulation_steps == 0:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                    if step_idx % self.params.save_freq == 0:
                        self._save_model_every_x_steps(step_idx, batch_idx)
                    # Perform eval if its time to eval
                    if not self.update_idx or self.update_idx % self.params.eval_steps != 0:
                        continue
                    
                    # Clear GPU memory for eval
                    torch.cuda.empty_cache()
                    self._eval_model(n_batches=100)
                        
                    # Save the current model if its the best we've ever had
                    if self.is_best_state:
                        self._save_model(batch_idx)
                    elif not self.patience_left:
                        no_improve_steps = self.params.eval_steps * self.params.patience
                        logger.info(
                            "Early termination, as eval loss did not improve "
                            f"over last {no_improve_steps} updates"
                        )
                        break
                    
                self.epoch_idx += 1
            torch.cuda.empty_cache()
            self._eval_model(n_batches=100)
            self._save_model_every_x_steps(step_idx, batch_idx)
        finally:
            self.logger.finish()