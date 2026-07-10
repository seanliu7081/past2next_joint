"""Shared pytest setup for the oat test suite.

The ``oat`` package is a PEP-420 namespace package that is not (correctly)
editable-installed in the env, so we bootstrap ``sys.path`` with the repo root
whenever ``oat.policy`` cannot already be resolved.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _oat_importable() -> bool:
    try:
        return importlib.util.find_spec("oat.policy.flow_policy") is not None
    except ModuleNotFoundError:
        return False


if not _oat_importable():
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from oat.model.common.normalizer import LinearNormalizer


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "sim: needs mujoco/libero sim + local data"
    )


# Known asymmetric raw ranges (min/max are pinned exactly in the sampled data
# below, so fitted stats equal these vectors bit-for-bit in float32).
ACTION_MIN = np.array([-0.6, -0.15, -0.4, -0.3, -0.9, -0.5, -1.0], dtype=np.float32)
ACTION_MAX = np.array([0.9, 0.45, 0.2, 0.7, 0.3, 0.5, 1.0], dtype=np.float32)
STATE_MIN = np.array([-2.0, -0.5, 0.1, -1.5, -0.25], dtype=np.float32)
STATE_MAX = np.array([1.0, 3.5, 0.9, 0.5, 0.75], dtype=np.float32)


def _uniform(rng, lo, hi, n):
    x = rng.uniform(lo, hi, size=(n, lo.shape[0])).astype(np.float32)
    x[0] = lo  # pin the extremes so min/max stats are exactly known
    x[1] = hi
    return x


@pytest.fixture()
def action_data():
    """Random (N, 7) raw actions with known asymmetric per-dim ranges."""
    rng = np.random.default_rng(0)
    return _uniform(rng, ACTION_MIN, ACTION_MAX, 4096)


@pytest.fixture()
def state_data():
    """Random (N, 5) low-dim obs with known asymmetric per-dim ranges."""
    rng = np.random.default_rng(1)
    return _uniform(rng, STATE_MIN, STATE_MAX, 4096)


@pytest.fixture()
def fitted_normalizer(action_data, state_data):
    """Baseline per-dim min-max LinearNormalizer fitted over the random data
    (mode='limits', the exact call ZarrDataset.get_normalizer makes)."""
    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": torch.from_numpy(action_data),
            "agent_state": torch.from_numpy(state_data),
        },
        last_n_dims=1,
        mode="limits",
    )
    return normalizer


@pytest.fixture()
def synthetic_stats():
    """Per-key raw stats dicts (the ``group_compatible_scale_offset`` input)."""

    def stats_of(lo, hi):
        return {
            "min": lo.astype(np.float64),
            "max": hi.astype(np.float64),
            "mean": ((lo + hi) / 2.0).astype(np.float64),
            "std": ((hi - lo) / np.sqrt(12.0)).astype(np.float64),
        }

    return {
        "action": stats_of(ACTION_MIN, ACTION_MAX),
        "agent_state": stats_of(STATE_MIN, STATE_MAX),
    }
