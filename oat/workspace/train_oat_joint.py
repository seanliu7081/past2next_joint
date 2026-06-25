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
from typing import Union

from oat.workspace.base_workspace import BaseWorkspace
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


class TrainOATJointWorkspace(BaseWorkspace):
    """End-to-end joint (co-)training of the action tokenizer + AR policy backbone.

    Mirrors `TrainPolicyWorkspace` but: builds a trainable tokenizer inside the
    policy (`OATPolicyJoint`), uses a joint optimizer that includes `tokenizer_lr`,
    runs a tokenizer-only warmup phase (`training.tokenizer_warmup_epochs`), and logs
    the reconstruction / cross-entropy loss components separately.
    """
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None, lazy_instantiation=True):
        super().__init__(cfg, output_dir=output_dir)

        if lazy_instantiation:
            self.model = None
            self.ema_model = None
            self.optimizer = None
        else:
            self.model = hydra.utils.instantiate(cfg.policy)
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
                DistributedDataParallelKwargs(find_unused_parameters=False),
                InitProcessGroupKwargs(timeout=timedelta(hours=2)), # sim eval can take long time
            ],
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
            mixed_precision="bf16" if cfg.training.allow_bf16 and detect_bf16_support() else "no",
        )
        device = accelerator.device

        # set seed
        seed = int(cfg.training.seed)
        accelerate_set_seed(seed, device_specific=True)

        # configure model, ema, and optimizer after seeding
        self.model: BasePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure dataset
        dataset: BaseDataset = hydra.utils.instantiate(
            cfg.task.policy.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # configure normalizer (covers obs encoder AND trainable tokenizer)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure checkpoint
        if accelerator.is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"),
                **cfg.checkpoint.topk
            )

        # configure env
        lazy_eval = cfg.task.policy.lazy_eval  # don't eval during training
        if (not lazy_eval) and accelerator.is_main_process:
            env_runner: BaseRunner = hydra.utils.instantiate(
                cfg.task.policy.env_runner,
                output_dir=self.output_dir
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
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
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

        # tokenizer warmup configuration
        tokenizer_warmup_epochs = cfg.training.get('tokenizer_warmup_epochs', 0)

        # training loop
        with JsonLogger(os.path.join(self.output_dir, 'logs.json')) as json_logger:
            while self.epoch < cfg.training.num_epochs:

                if accelerator.is_main_process:
                    step_log = dict()

                # model to train mode
                self.model.train()
                if cfg.training.use_ema:
                    self.ema_model.train()

                # tokenizer-only warmup: zero the CE weight so the backbone/obs encoder
                # are not trained yet. We still COMPUTE ce_loss (multiplied by 0.0) so
                # those params stay in the autograd graph -> DDP find_unused_parameters
                # =False stays happy, and weight_decay=0.0 keeps AdamW from drifting them.
                is_warmup = self.epoch < tokenizer_warmup_epochs
                current_ce_weight = 0.0 if is_warmup else \
                    accelerator.unwrap_model(self.model).ce_loss_weight
                accelerator.unwrap_model(self.model).set_ce_weight(current_ce_weight)

                loss_info = torch.zeros(4, device=device)   # [tot loss, tot recon, tot ce, tot bs]
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                    mininterval=cfg.training.tqdm_interval_sec
                ) as tepoch:

                    for batch_idx, batch in enumerate(tepoch):
                        with accelerator.accumulate(self.model):
                            # device transfer
                            batch = dict_apply(batch, lambda x: maybe_to_device(x, device))

                            # forward pass
                            with accelerator.autocast():
                                out = self.model(batch)
                            loss = out['loss']

                            # backward pass
                            accelerator.backward(loss)

                            # log loss
                            batch_size = batch['action'].shape[0]
                            loss_info[0] += loss.detach() * batch_size
                            loss_info[1] += out['recon_loss'].detach() * batch_size
                            loss_info[2] += out['ce_loss'].detach() * batch_size
                            loss_info[3] += batch_size

                            # step optimizer
                            if accelerator.sync_gradients:
                                # clip grad norm
                                if cfg.training.max_grad_norm is not None:
                                    accelerator.clip_grad_norm_(
                                        self.model.parameters(),
                                        cfg.training.max_grad_norm
                                    )

                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)
                                lr_scheduler.step()

                                # update ema
                                if cfg.training.use_ema:
                                    ema.step(accelerator.unwrap_model(self.model))

                            # logging
                            is_last_batch = (batch_idx == (len_train_dataloader-1))
                            if accelerator.is_main_process:
                                step_log = {
                                    'train_loss': loss.item(),
                                    'recon_loss': out['recon_loss'].item(),
                                    'ce_loss': out['ce_loss'].item(),
                                    'ce_weight': current_ce_weight,
                                    'global_step': self.global_step,
                                    'epoch': self.epoch,
                                    'lr': lr_scheduler.get_last_lr()[0],
                                }
                                tepoch.set_postfix(loss=step_log['train_loss'], refresh=False)
                                if not is_last_batch:
                                    accelerator.log(step_log, step=self.global_step)
                                    json_logger.log(step_log)

                            # increment global step
                            if not is_last_batch:
                                self.global_step += 1

                            # break if reach max training steps
                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break

                # at the end of each epoch
                # replace train_loss with epoch average
                accelerator.wait_for_everyone()
                loss_info = accelerator.reduce(loss_info, reduction='sum')
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    step_log['train_loss'] = (loss_info[0] / loss_info[3]).item()
                    step_log['recon_loss'] = (loss_info[1] / loss_info[3]).item()
                    step_log['ce_loss'] = (loss_info[2] / loss_info[3]).item()

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = accelerator.unwrap_model(self.ema_model)
                policy.eval()

                # run policy rollout
                if not lazy_eval:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and (self.epoch % cfg.training.rollout_every) == 0:
                        runner_log = env_runner.run(policy)
                        step_log.update(runner_log)
                    accelerator.wait_for_everyone()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    loss_info = torch.zeros(4, device=device)   # [tot loss, tot recon, tot ce, tot bs]
                    with torch.inference_mode():
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:

                            for batch_idx, batch in enumerate(tepoch):
                                # device transfer
                                batch = dict_apply(batch, lambda x: maybe_to_device(x, device, non_blocking=True))

                                # forward pass
                                out = policy(batch)

                                # log loss
                                batch_size = batch['action'].shape[0]
                                loss_info[0] += out['loss'].item() * batch_size
                                loss_info[1] += out['recon_loss'].item() * batch_size
                                loss_info[2] += out['ce_loss'].item() * batch_size
                                loss_info[3] += batch_size

                                # break if reach max val steps
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break

                    # logging
                    accelerator.wait_for_everyone()
                    loss_info = accelerator.reduce(loss_info, reduction='sum')
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        step_log['val_loss'] = (loss_info[0] / loss_info[3]).item()
                        step_log['val_recon_loss'] = (loss_info[1] / loss_info[3]).item()
                        step_log['val_ce_loss'] = (loss_info[2] / loss_info[3]).item()

                # action prediction eval (tokenizer-only recon + full pipeline recon)
                if self.epoch % cfg.training.sample_every == 0:
                    recon_info = torch.zeros(3, device=device)   # [tot tok_mse, tot test_mse, tot bs]
                    with torch.inference_mode():
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Reconstruction epoch {self.epoch}",
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:

                            for batch_idx, batch in enumerate(tepoch):
                                # device transfer
                                batch = dict_apply(batch, lambda x: maybe_to_device(x, device, non_blocking=True))

                                gt_action = batch['action']     # [B, Ta, Da]

                                # (a) tokenizer-only reconstruction (raw action space)
                                tok_recon = policy.action_tokenizer.autoencode(samples=gt_action)
                                tok_mse = F.mse_loss(tok_recon, gt_action).item()

                                # (b) full-pipeline reconstruction (AR generate -> detokenize)
                                result = policy.predict_action(batch['obs'])
                                pred_action = result['action_pred']  # [B, Ta, Da]
                                test_mse = F.mse_loss(pred_action, gt_action).item()

                                # log loss
                                batch_size = batch['action'].shape[0]
                                recon_info[0] += tok_mse * batch_size
                                recon_info[1] += test_mse * batch_size
                                recon_info[2] += batch_size

                                # early stop if reach max samples
                                if (cfg.training.max_reconst_steps is not None) \
                                    and batch_idx >= (cfg.training.max_reconst_steps-1):
                                    break

                    # logging
                    accelerator.wait_for_everyone()
                    recon_info = accelerator.reduce(recon_info, reduction='sum')
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        step_log['tok_reconst_mse'] = (recon_info[0] / recon_info[2]).item()
                        step_log['test_reconst_mse'] = (recon_info[1] / recon_info[2]).item()

                # checkpoint
                if accelerator.is_main_process and (self.epoch % cfg.training.checkpoint_every) == 0:
                    # unwrap
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if cfg.training.use_ema:
                        ema_model_ddp = self.ema_model
                        self.ema_model = accelerator.unwrap_model(self.ema_model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # restore
                    self.model = model_ddp
                    if cfg.training.use_ema:
                        self.ema_model = ema_model_ddp

                # end of epoch
                # log of last step is combined with validation and rollout
                if accelerator.is_main_process:
                    accelerator.log(step_log, step=self.global_step)
                    json_logger.log(step_log)

                # increment epoch and global step
                self.epoch += 1
                self.global_step += 1

        # clean up
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
    workspace = TrainOATJointWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
