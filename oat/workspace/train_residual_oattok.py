"""
train_residual_oattok.py
========================
Minimal fork of train_oattok.py for training ResidualOATTok.

Changes vs train_oattok.py (all marked with # [CHANGED]):
  1. Import ResidualOATTok instead of OATTok
  2. Normalizer is fit on residuals (past_action[:,-1,:] subtracted from action)
     instead of raw actions via dataset.get_normalizer()
  3. Reconstruction eval: tokenizer.autoencode(samples, baseline) requires baseline
  4. Task config uses libero10_with_past (ZarrDatasetWithPastAction, past_n=7)

Everything else — training loop, EMA, LR scheduler, checkpointing,
accelerator setup — is identical to train_oattok.py.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pathlib
import copy
import tqdm
from accelerate import Accelerator
from accelerate.utils import (
    set_seed as accelerate_set_seed, DistributedDataParallelKwargs)

from oat.workspace.base_workspace import BaseWorkspace
from oat.dataset.base_dataset import BaseDataset
from oat.common.checkpoint_util import TopKCheckpointManager
from oat.common.json_logger import JsonLogger
from oat.common.pytorch_util import dict_apply
from oat.common.hydra_util import register_new_resolvers
from oat.model.common.lr_scheduler import get_scheduler
from oat.model.common.normalizer import LinearNormalizer
from oat.model.common.misc import get_generator, detect_bf16_support
from oat.tokenizer.oat.residual_tokenizer import ResidualOATTok   # [CHANGED]

register_new_resolvers()


def fit_residual_normalizer(dataloader, device) -> LinearNormalizer:           # [CHANGED]
    """
    Compute a LinearNormalizer fitted on ZeroOrder residuals
    (action - past_action[:, -1, :]) across the entire training split.

    Replaces the standard `dataset.get_normalizer()` call.
    The dataloader must yield batches with "action" and "past_action" keys,
    i.e. the dataset must be ZarrDatasetWithPastAction (past_n >= 1).
    """
    all_residuals = []

    for batch in tqdm.tqdm(dataloader, desc="Fitting residual normalizer", leave=False):
        action      = batch["action"].to(device)           # (B, T, d)
        past_action = batch["past_action"].to(device)      # (B, past_n, d)
        baseline    = past_action[:, -1, :]                # (B, d)  = a_t
        residual    = action - baseline.unsqueeze(1)       # (B, T, d)

        B, T, d = residual.shape
        all_residuals.append(residual.reshape(B * T, d).cpu())

    residuals = torch.cat(all_residuals, dim=0)            # (N*T, d)

    normalizer = LinearNormalizer()
    normalizer.fit({"action": residuals})

    mean = residuals.mean(0)
    std  = residuals.std(0)
    print(f"Residual normalizer fitted on {residuals.shape[0]} samples")
    print(f"  mean: {mean.tolist()}")
    print(f"  std : {std.tolist()}")
    return normalizer


class TrainResidualOATTokWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None, lazy_instantiation=True):
        super().__init__(cfg, output_dir=output_dir)
        if lazy_instantiation:
            self.model = None
            self.ema_model = None
            self.optimizer = None
        else:
            self.model = hydra.utils.instantiate(cfg.tokenizer)
            if cfg.training.use_ema:
                self.ema_model = copy.deepcopy(self.model)
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure accelerator
        accelerator = Accelerator(
            log_with="wandb",
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=False)
            ],
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
            mixed_precision="bf16" if cfg.training.allow_bf16 and detect_bf16_support() else "no",
        )
        device = accelerator.device

        # set seed
        seed = int(cfg.training.seed)
        accelerate_set_seed(seed, device_specific=True)

        # configure model, ema, optimizer
        self.model: ResidualOATTok = hydra.utils.instantiate(cfg.tokenizer)  # [CHANGED type hint]
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        torch_generator = get_generator(seed=seed, device=device)

        # configure dataset  (must be ZarrDatasetWithPastAction)
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.tokenizer.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # [CHANGED] fit normalizer on residuals, not raw actions
        normalizer = fit_residual_normalizer(train_dataloader, device=device)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure checkpoint
        if accelerator.is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"),
                **cfg.checkpoint.topk
            )

        # resume training
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)
                if self.epoch >= cfg.training.num_epochs:
                    accelerator.print(f"Already trained for {self.epoch} epochs. Exiting.")
                    return

        # prepare with accelerator
        (
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
        ) = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
        )
        if cfg.training.use_ema:
            self.ema_model = accelerator.prepare(self.ema_model)
            ema = hydra.utils.instantiate(cfg.ema, model=accelerator.unwrap_model(self.ema_model))

        # configure lr scheduler
        len_train_dataloader = len(train_dataloader)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len_train_dataloader * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )

        # configure logging
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop("project")
        wandb_cfg['dir'] = str(self.output_dir)
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )
        if accelerator.is_main_process:
            accelerator.get_tracker("wandb").run.config.update({
                "output_dir": str(self.output_dir)
            })

        # training loop
        with JsonLogger(os.path.join(self.output_dir, 'logs.json')) as json_logger:
            while self.epoch < cfg.training.num_epochs:

                if accelerator.is_main_process:
                    step_log = dict()

                # train
                self.model.train()
                if cfg.training.use_ema:
                    self.ema_model.train()

                loss_info = torch.zeros(2, device=device)
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                    mininterval=cfg.training.tqdm_interval_sec
                ) as tepoch:

                    for batch_idx, batch in enumerate(tepoch):
                        with accelerator.accumulate(self.model):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                            with accelerator.autocast():
                                # [UNCHANGED] ResidualOATTok.forward(batch) handles
                                # baseline extraction internally
                                loss = self.model(batch)

                            accelerator.backward(loss)

                            batch_size = batch['action'].shape[0]
                            loss_info[0] += loss.item() * batch_size
                            loss_info[1] += batch_size

                            if accelerator.sync_gradients:
                                if cfg.training.max_grad_norm is not None:
                                    accelerator.clip_grad_norm_(
                                        self.model.parameters(),
                                        cfg.training.max_grad_norm
                                    )

                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            lr_scheduler.step()

                            if cfg.training.use_ema:
                                ema.step(accelerator.unwrap_model(self.model))

                            is_last_batch = (batch_idx == (len_train_dataloader - 1))
                            if accelerator.is_main_process:
                                loss_cpu = loss.item()
                                tepoch.set_postfix(loss=loss_cpu, refresh=False)
                                step_log = {
                                    'train_loss': loss_cpu,
                                    'global_step': self.global_step,
                                    'epoch': self.epoch,
                                    'lr': lr_scheduler.get_last_lr()[0],
                                }
                                if not is_last_batch:
                                    accelerator.log(step_log, step=self.global_step)
                                    json_logger.log(step_log)

                            if not is_last_batch:
                                self.global_step += 1

                            if (cfg.training.max_train_steps is not None) \
                                    and batch_idx >= (cfg.training.max_train_steps - 1):
                                break

                # end-of-epoch
                accelerator.wait_for_everyone()
                loss_info = accelerator.reduce(loss_info, reduction='sum')
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    step_log['train_loss'] = (loss_info[0] / loss_info[1]).item()

                # ========= eval =========
                tokenizer = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    tokenizer = accelerator.unwrap_model(self.ema_model)
                tokenizer.eval()

                # validation loss (in normalized residual space)
                if (self.epoch % cfg.training.val_every) == 0:
                    loss_info = torch.zeros(2, device=device)
                    with torch.inference_mode():
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:

                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                # [UNCHANGED] ResidualOATTok.forward() handles baseline
                                loss = tokenizer(batch).item()

                                batch_size = batch['action'].shape[0]
                                loss_info[0] += loss * batch_size
                                loss_info[1] += batch_size

                                if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break

                    accelerator.wait_for_everyone()
                    loss_info = accelerator.reduce(loss_info, reduction='sum')
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        step_log['val_loss'] = (loss_info[0] / loss_info[1]).item()

                # reconstruction eval (MSE in raw action space — apples-to-apples with OATTok)
                if self.epoch % cfg.training.sample_every == 0:
                    loss_info = torch.zeros(2, device=device)
                    with torch.inference_mode():
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Reconstruction epoch {self.epoch}",
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:

                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                                samples  = batch['action']
                                # [CHANGED] pass baseline to autoencode
                                baseline = batch['past_action'][:, -1, :]
                                reconst_samples = tokenizer.autoencode(
                                    samples=samples,
                                    baseline=baseline,
                                )
                                # MSE in raw action space (same metric as OATTok baseline)
                                mse = F.mse_loss(reconst_samples, samples).item()

                                batch_size = samples.shape[0]
                                loss_info[0] += mse * batch_size
                                loss_info[1] += batch_size

                                if (cfg.training.max_reconst_steps is not None) \
                                        and batch_idx >= (cfg.training.max_reconst_steps - 1):
                                    break

                    accelerator.wait_for_everyone()
                    loss_info = accelerator.reduce(loss_info, reduction='sum')
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        step_log['test_reconst_mse'] = (loss_info[0] / loss_info[1]).item()

                # checkpoint
                if accelerator.is_main_process and (self.epoch % cfg.training.checkpoint_every) == 0:
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if cfg.training.use_ema:
                        ema_model_ddp = self.ema_model
                        self.ema_model = accelerator.unwrap_model(self.ema_model)

                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    self.model = model_ddp
                    if cfg.training.use_ema:
                        self.ema_model = ema_model_ddp

                # end of epoch
                if accelerator.is_main_process:
                    accelerator.log(step_log, step=self.global_step)
                    json_logger.log(step_log)

                self.epoch += 1
                self.global_step += 1

        accelerator.wait_for_everyone()
        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem
)
def main(cfg):
    workspace = TrainResidualOATTokWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()