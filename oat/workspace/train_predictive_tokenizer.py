"""Workspace for predictive-tokenizer (rate-distortion coupled) training.

Subclasses ``TrainOATJointWorkspace`` and overrides ``run()`` to add:

  1. Per-epoch schedule hooks (``set_beta``, ``set_tau``, ``set_hard_finetune``):
       * Stage 0 distortion warmup: ``beta = 0`` for ``tokenizer_warmup_epochs``.
       * ``beta`` linear ramp 0 -> target over ``beta_ramp_epochs``.
       * Temperature anneal ``tau_start`` -> ``tau_end`` over ``tau_anneal_epochs``.
       * Mandatory hard-token finetune for the last ``hard_finetune_epochs``.
  2. Rate-distortion logging (recon D, rate R, per-dim rate, usage, AR accuracy,
     hard-token CE parity, marginal perplexity, tau, beta).
  3. The §16.1 fix: per-validation-batch ``policy.reset()`` before
     ``predict_action`` in the reconstruction diagnostic, so ``_past_buffer``
     does not leak across batches.

The shared ``TrainOATJointWorkspace`` is left untouched.
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
from datetime import timedelta
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pathlib
import copy
import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import (
    set_seed as accelerate_set_seed, DistributedDataParallelKwargs)

from oat.workspace.train_oat_joint import TrainOATJointWorkspace
from oat.dataset.base_dataset import BaseDataset
from oat.env_runner.base_runner import BaseRunner
from oat.common.checkpoint_util import TopKCheckpointManager
from oat.common.json_logger import JsonLogger
from oat.common.hydra_util import register_new_resolvers
from oat.common.pytorch_util import dict_apply, maybe_to_device
from oat.model.common.lr_scheduler import get_scheduler
from oat.model.common.misc import detect_bf16_support
from oat.policy.base_policy import BasePolicy

register_new_resolvers()


class TrainPredictiveTokenizerWorkspace(TrainOATJointWorkspace):
    include_keys = ['global_step', 'epoch']

    # ── Schedule ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_schedule(epoch: int, cfg):
        """Return (beta, tau, hard_finetune) for the given epoch."""
        t = cfg.training
        warmup = int(t.get('tokenizer_warmup_epochs', 0))
        beta_target = float(t.get('beta', 0.0))
        beta_ramp = int(t.get('beta_ramp_epochs', 0))
        # During the hard-token finetune the distortion path is already hard
        # (z_st is straight-through hard-forward in every phase), so the ONLY
        # behavioural change of hard_finetune is switching teacher forcing
        # soft -> hard. That switch trains the AR (p) only through the rate term
        # beta*R, so beta_hard MUST stay > 0 for the AR to actually adapt to the
        # hard/argmax regime it faces at inference. Default to the ramped target.
        beta_hard = float(t.get('beta_hard', beta_target))
        tau_start = float(t.get('tau_start', 2.0))
        tau_end = float(t.get('tau_end', 0.25))
        tau_anneal = int(t.get('tau_anneal_epochs', 0))
        hard_ft = int(t.get('hard_finetune_epochs', 0))
        num_epochs = int(t.num_epochs)

        # last K epochs: hard-token finetune
        in_hard = hard_ft > 0 and epoch >= (num_epochs - hard_ft)

        # beta: 0 during warmup, then linear ramp to target, then constant
        if epoch < warmup:
            beta = 0.0
        elif beta_ramp > 0 and epoch < warmup + beta_ramp:
            beta = beta_target * (epoch - warmup) / float(beta_ramp)
        else:
            beta = beta_target
        if in_hard:
            beta = beta_hard

        # tau: linear anneal tau_start -> tau_end over tau_anneal epochs
        if tau_anneal <= 0:
            tau = tau_end
        else:
            frac = min(1.0, epoch / float(tau_anneal))
            tau = tau_start + (tau_end - tau_start) * frac
        if in_hard:
            tau = tau_end

        return beta, tau, in_hard

    # ── Training loop ────────────────────────────────────────────────────────

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(
            log_with="wandb",
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=False),
                InitProcessGroupKwargs(timeout=timedelta(hours=2)),
            ],
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
            mixed_precision="bf16" if cfg.training.allow_bf16 and detect_bf16_support() else "no",
        )
        device = accelerator.device

        seed = int(cfg.training.seed)
        accelerate_set_seed(seed, device_specific=True)

        self.model: BasePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.policy.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        if accelerator.is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"),
                **cfg.checkpoint.topk
            )

        lazy_eval = cfg.task.policy.lazy_eval
        if (not lazy_eval) and accelerator.is_main_process:
            env_runner: BaseRunner = hydra.utils.instantiate(
                cfg.task.policy.env_runner,
                output_dir=self.output_dir
            )

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)
                if self.epoch >= cfg.training.num_epochs:
                    accelerator.print(f"Already trained for {self.epoch} epochs. Exiting.")
                    return

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

        len_train_dataloader = len(train_dataloader)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len_train_dataloader * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

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

                self.model.train()
                if cfg.training.use_ema:
                    self.ema_model.train()

                # ── per-epoch schedule (beta / tau / hard-finetune) ─────────
                beta, tau, hard_ft = self._compute_schedule(self.epoch, cfg)
                core = accelerator.unwrap_model(self.model)
                core.set_beta(beta)
                core.set_tau(tau)
                core.set_hard_finetune(hard_ft)
                if cfg.training.use_ema:
                    ema_core = accelerator.unwrap_model(self.ema_model)
                    ema_core.set_beta(beta)
                    ema_core.set_tau(tau)
                    ema_core.set_hard_finetune(hard_ft)

                # [tot loss, tot recon(D), tot rate(R), tot bs]
                loss_info = torch.zeros(4, device=device)
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                    mininterval=cfg.training.tqdm_interval_sec
                ) as tepoch:

                    for batch_idx, batch in enumerate(tepoch):
                        with accelerator.accumulate(self.model):
                            batch = dict_apply(batch, lambda x: maybe_to_device(x, device))

                            with accelerator.autocast():
                                out = self.model(batch)
                            loss = out['loss']

                            accelerator.backward(loss)

                            batch_size = batch['action'].shape[0]
                            loss_info[0] += loss.detach() * batch_size
                            loss_info[1] += out['recon'].detach() * batch_size
                            loss_info[2] += out['rate'].detach() * batch_size
                            loss_info[3] += batch_size

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

                            is_last_batch = (batch_idx == (len_train_dataloader-1))
                            if accelerator.is_main_process:
                                step_log = {
                                    'train_loss': loss.item(),
                                    'recon': out['recon'].item(),
                                    'rate': out['rate'].item(),
                                    'usage': out['usage'].item(),
                                    'hard_ce': out['hard_ce'].item(),
                                    'ar_acc': out['ar_acc'].item(),
                                    'usage_ppl': out['usage_ppl'].item(),
                                    'beta': beta,
                                    'tau': tau,
                                    'hard_finetune': float(hard_ft),
                                    'global_step': self.global_step,
                                    'epoch': self.epoch,
                                    'lr': lr_scheduler.get_last_lr()[0],
                                }
                                # per-dim rate breakdown
                                rpd = out['rate_per_dim'].detach().cpu()
                                for di in range(rpd.numel()):
                                    step_log[f'rate_dim{di}'] = rpd[di].item()
                                tepoch.set_postfix(loss=step_log['train_loss'], refresh=False)
                                if not is_last_batch:
                                    accelerator.log(step_log, step=self.global_step)
                                    json_logger.log(step_log)

                            if not is_last_batch:
                                self.global_step += 1

                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break

                # epoch-average train losses
                accelerator.wait_for_everyone()
                loss_info = accelerator.reduce(loss_info, reduction='sum')
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    step_log['train_loss'] = (loss_info[0] / loss_info[3]).item()
                    step_log['recon'] = (loss_info[1] / loss_info[3]).item()
                    step_log['rate'] = (loss_info[2] / loss_info[3]).item()

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = accelerator.unwrap_model(self.ema_model)
                policy.eval()

                if not lazy_eval:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and (self.epoch % cfg.training.rollout_every) == 0:
                        runner_log = env_runner.run(policy)
                        step_log.update(runner_log)
                    accelerator.wait_for_everyone()

                # ── validation ──────────────────────────────────────────────
                if (self.epoch % cfg.training.val_every) == 0:
                    loss_info = torch.zeros(4, device=device)
                    with torch.inference_mode():
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:

                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: maybe_to_device(x, device, non_blocking=True))
                                out = policy(batch)

                                batch_size = batch['action'].shape[0]
                                loss_info[0] += out['loss'].item() * batch_size
                                loss_info[1] += out['recon'].item() * batch_size
                                loss_info[2] += out['rate'].item() * batch_size
                                loss_info[3] += batch_size

                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break

                    accelerator.wait_for_everyone()
                    loss_info = accelerator.reduce(loss_info, reduction='sum')
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        step_log['val_loss'] = (loss_info[0] / loss_info[3]).item()
                        step_log['val_recon'] = (loss_info[1] / loss_info[3]).item()
                        step_log['val_rate'] = (loss_info[2] / loss_info[3]).item()

                # ── reconstruction diagnostic (tokenizer-only + full pipeline) ─
                if self.epoch % cfg.training.sample_every == 0:
                    recon_info = torch.zeros(3, device=device)   # [tok_mse, test_mse, bs]
                    with torch.inference_mode():
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Reconstruction epoch {self.epoch}",
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:

                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: maybe_to_device(x, device, non_blocking=True))
                                gt_action = batch['action']

                                # (a) tokenizer-only reconstruction (centered-grid path)
                                tok_recon = policy.reconstruct_actions(gt_action)
                                tok_mse = F.mse_loss(tok_recon, gt_action).item()

                                # (b) full-pipeline reconstruction (AR generate -> decode)
                                # §16.1 fix: reset per batch so _past_buffer does
                                # not leak across consecutive validation batches.
                                policy.reset()
                                result = policy.predict_action(batch['obs'])
                                pred_action = result['action_pred']
                                test_mse = F.mse_loss(pred_action, gt_action).item()

                                batch_size = batch['action'].shape[0]
                                recon_info[0] += tok_mse * batch_size
                                recon_info[1] += test_mse * batch_size
                                recon_info[2] += batch_size

                                if (cfg.training.max_reconst_steps is not None) \
                                    and batch_idx >= (cfg.training.max_reconst_steps-1):
                                    break

                    accelerator.wait_for_everyone()
                    recon_info = accelerator.reduce(recon_info, reduction='sum')
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        step_log['tok_reconst_mse'] = (recon_info[0] / recon_info[2]).item()
                        step_log['test_reconst_mse'] = (recon_info[1] / recon_info[2]).item()

                # ── checkpoint ──────────────────────────────────────────────
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

                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    self.model = model_ddp
                    if cfg.training.use_ema:
                        self.ema_model = ema_model_ddp

                if accelerator.is_main_process:
                    accelerator.log(step_log, step=self.global_step)
                    json_logger.log(step_log)

                self.epoch += 1
                self.global_step += 1

        if not lazy_eval:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and (not lazy_eval):
                env_runner.close()
            accelerator.wait_for_everyone()
        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainPredictiveTokenizerWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
