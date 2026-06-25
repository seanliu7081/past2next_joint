import copy
import torch
import numpy as np
from typing import Dict, List, Optional

from oat.common.seq_sampler import SequenceSampler
from oat.dataset.zarr_dataset import ZarrDataset, is_numeric_dtype


class ZarrDatasetWithPastAction(ZarrDataset):
    """
    Extends ZarrDataset to also return `past_action` — the past_n action
    steps immediately preceding the current action chunk.

    Time alignment (To=2, Ta=32, past_n=7):
        obs:         [t,   t+1]                       (2 frames)
        past_action: [t-6, t-5, ..., t]               (7 frames)
        action:      [t+1, t+2, ..., t+32]            (32 frames)

    At episode boundaries the SequenceSampler zero-pads, so past_action
    is naturally all-zeros at the start of an episode.
    """

    def __init__(
        self,
        past_n: int = 7,
        # all ZarrDataset args
        zarr_path: str = "",
        obs_keys: List[str] = [],
        action_key: str = "action",
        n_obs_steps: int = 2,
        n_action_steps: int = 16,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
    ):
        # Initialise parent — this creates the seq_sampler with the
        # original padding.  We immediately override it below.
        super().__init__(
            zarr_path=zarr_path,
            obs_keys=obs_keys,
            action_key=action_key,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
        )

        self.past_n = past_n

        # ── Extend the sampling window to include past actions ────────────
        self.pad_before = max(n_obs_steps - 1, 0) + past_n
        self.pad_after = max(n_action_steps - 1, 0)
        self.seq_len = self.pad_before + 1 + self.pad_after

        # Re-create the sequence sampler with wider window
        self.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.seq_len,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.train_mask,
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.seq_len,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def _sample_to_data(self, sample):
        To = self.n_obs_steps
        Ta = self.n_action_steps
        past_n = self.past_n

        # ── obs (shifted by past_n relative to parent) ────────────────────
        obs = {}
        for k in self.numeric_obs_keys:
            raw = sample[k][past_n: past_n + To]
            if raw.dtype.kind == "f":
                obs[k] = raw.astype(np.float32)
            else:
                obs[k] = raw
        for k in self.text_obs_keys:
            obs[k] = sample[k][past_n]

        # ── action chunk (same real-world frames as parent) ───────────────
        action_start = past_n + max(To - 1, 0)
        action = sample[self.action_key][action_start: action_start + Ta].astype(np.float32)

        # ── past action (past_n steps immediately before the chunk) ───────
        past_action = sample[self.action_key][action_start - past_n: action_start].astype(np.float32)

        return {"obs": obs, "action": action, "past_action": past_action}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.seq_sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_obs = {}
        for k, v in data["obs"].items():
            if isinstance(v, np.ndarray) and is_numeric_dtype(v):
                torch_obs[k] = torch.from_numpy(v)
            elif isinstance(v, bytes):
                torch_obs[k] = v.decode("utf-8")
            else:
                torch_obs[k] = v

        torch_act = torch.from_numpy(data["action"])
        torch_past_act = torch.from_numpy(data["past_action"])

        return {
            "obs": torch_obs,
            "action": torch_act,
            "past_action": torch_past_act,
        }
