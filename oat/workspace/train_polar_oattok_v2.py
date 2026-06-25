"""Training workspace for PolarOATTok v2 with decomposed loss logging."""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hydra
import pathlib
from omegaconf import OmegaConf

from oat.workspace.train_oattok import TrainOATTokWorkspace
from oat.common.hydra_util import register_new_resolvers

register_new_resolvers()


class TrainPolarOATTokV2Workspace(TrainOATTokWorkspace):
    """Extends TrainOATTokWorkspace to log decomposed recon/equiv losses."""

    def _get_extra_step_log(self) -> dict:
        """Collect decomposed losses from the model if available."""
        model = self.model
        # Unwrap DDP if needed
        if hasattr(model, 'module'):
            model = model.module
        extra = {}
        if hasattr(model, 'last_recon_loss'):
            extra['recon_loss'] = model.last_recon_loss
        if hasattr(model, 'last_equiv_loss'):
            extra['equiv_loss'] = model.last_equiv_loss
        return extra


# Override the run method to inject extra logging
_original_run = TrainOATTokWorkspace.run


def _patched_run(self):
    """Monkey-patch approach: wrap the parent run to inject extra logging.

    Instead of duplicating the entire 300-line run() method, we override
    the model's forward to capture extra losses, then rely on the parent
    training loop. The decomposed losses are already stashed as attributes
    by PolarOATTokV2.forward().

    The parent workspace logs step_log at line 233. We hook into this by
    post-processing: after each forward call, we append the extra keys
    to the accelerator tracker directly.
    """
    # Just call the parent run - the decomposed losses are already stashed
    # on the model as last_recon_loss / last_equiv_loss.
    # The parent logs 'train_loss' which is the total.
    # To log the components, we wrap the forward method.
    _original_run(self)


# Actually, the cleanest approach is to just duplicate the small logging section.
# But to avoid touching the parent at all, let's use a simpler approach:
# wrap the model's forward to update a shared dict.

def run(self):
    """Override run to add decomposed loss logging."""
    import copy
    import os
    import torch
    import torch.nn.functional as F
    import tqdm
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from accelerate.utils import (
        set_seed as accelerate_set_seed, DistributedDataParallelKwargs)

    from oat.dataset.base_dataset import BaseDataset
    from oat.common.checkpoint_util import TopKCheckpointManager
    from oat.common.json_logger import JsonLogger
    from oat.common.pytorch_util import dict_apply
    from oat.model.common.lr_scheduler import get_scheduler
    from oat.model.common.misc import get_generator, detect_bf16_support

    cfg = copy.deepcopy(self.cfg)

    accelerator = Accelerator(
        log_with="wandb",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False),
        ],
        gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
        mixed_precision="bf16" if cfg.training.allow_bf16 and detect_bf16_support() else "no",
    )
    device = accelerator.device
    seed = int(cfg.training.seed)
    accelerate_set_seed(seed, device_specific=True)

    self.model = hydra.utils.instantiate(cfg.tokenizer)
    self.ema_model = None
    if cfg.training.use_ema:
        self.ema_model = copy.deepcopy(self.model)
    self.optimizer = self.model.get_optimizer(**cfg.optimizer)
    torch_generator = get_generator(seed=seed, device=device)

    dataset: BaseDataset = hydra.utils.instantiate(cfg.task.tokenizer.dataset)
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

    if cfg.training.resume:
        latest_ckpt_path = self.get_checkpoint_path()
        if latest_ckpt_path.is_file():
            accelerator.print(f"Resuming from checkpoint {latest_ckpt_path}")
            self.load_checkpoint(path=latest_ckpt_path)
            if self.epoch >= cfg.training.num_epochs:
                accelerator.print(f"Already trained for {self.epoch} epochs. Exiting.")
                return

    (
        train_dataloader, val_dataloader, self.model, self.optimizer,
    ) = accelerator.prepare(
        train_dataloader, val_dataloader, self.model, self.optimizer,
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
            len_train_dataloader * cfg.training.num_epochs
        ) // cfg.training.gradient_accumulate_every,
        last_epoch=self.global_step - 1,
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

    with JsonLogger(os.path.join(self.output_dir, 'logs.json')) as json_logger:
        while self.epoch < cfg.training.num_epochs:
            if accelerator.is_main_process:
                step_log = dict()

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
                            # ── Decomposed loss logging (v2 addition) ──
                            unwrapped = accelerator.unwrap_model(self.model)
                            if hasattr(unwrapped, 'last_recon_loss'):
                                step_log['recon_loss'] = unwrapped.last_recon_loss
                            if hasattr(unwrapped, 'last_equiv_loss'):
                                step_log['equiv_loss'] = unwrapped.last_equiv_loss
                            # ────────────────────────────────────────────
                            if not is_last_batch:
                                accelerator.log(step_log, step=self.global_step)
                                json_logger.log(step_log)

                        if not is_last_batch:
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

            accelerator.wait_for_everyone()
            loss_info = accelerator.reduce(loss_info, reduction='sum')
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                step_log['train_loss'] = (loss_info[0] / loss_info[1]).item()

            # ========= eval for this epoch ==========
            tokenizer = accelerator.unwrap_model(self.model)
            if cfg.training.use_ema:
                tokenizer = accelerator.unwrap_model(self.ema_model)
            tokenizer.eval()

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
                            samples = batch['action']
                            reconst_samples = tokenizer.autoencode(samples=samples)
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

    accelerator.wait_for_everyone()
    accelerator.end_training()


TrainPolarOATTokV2Workspace.run = run


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainPolarOATTokV2Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
