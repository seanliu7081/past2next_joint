"""Tests for oat.equi.sources: PROBE 4 (source invariance under rho_A(theta)),
physical_so2 equivalence with the diffusion-side EquiNoise, degrade-to-identity,
and the exact-randn contracts of GaussianSource / the Level-1 stub."""

import numpy as np
import pytest
import torch
from scipy import stats as sps

from oat.equi.blocks import (
    FREE_ISO,
    RHO1,
    action_rho_matrix,
    libero_action_blocks,
    to_noise_blocks,
)
from oat.equi.sources import (
    BlockIsotropicSource,
    GaussianSource,
    Level1ScaleHeadSource,
    build_source,
)
from oat.model.diffusion.equi_noise import EquiNoise

UNEQUAL_SCALES = {"xy": 0.7, "z": 1.3, "rot": 0.5, "grip": 2.0}
N_PROBE = 200_000
N_KS = 20_000


@pytest.fixture(scope="module")
def probe_samples():
    torch.manual_seed(0)
    source = BlockIsotropicSource(7, libero_action_blocks(), scales=UNEQUAL_SCALES)
    z = source.sample((N_PROBE, 1, 7)).reshape(N_PROBE, 7).numpy()
    torch.manual_seed(1)
    fresh = source.sample((N_PROBE, 1, 7)).reshape(N_PROBE, 7).numpy()
    return z, fresh


# ── PROBE 4: source invariance ───────────────────────────────────────────────

def test_probe4_within_block_isotropy(probe_samples):
    """Per rotating (rho1 / free_iso) block: dims uncorrelated and equal-var."""
    z, _ = probe_samples
    for b in libero_action_blocks():
        if b.rep not in (RHO1, FREE_ISO):
            continue
        idx = list(b.idx)
        block = z[:, idx]
        var = block.var(axis=0)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                corr = np.corrcoef(block[:, i], block[:, j])[0, 1]
                assert abs(corr) < 0.01, f"block '{b.name}' corr({i},{j})={corr}"
                ratio = var[i] / var[j]
                assert 0.98 < ratio < 1.02, f"block '{b.name}' var ratio {ratio}"


@pytest.mark.parametrize("rotate_rotation", [False, True])
@pytest.mark.parametrize("theta_deg", [10.0, 20.0, 30.0])
def test_probe4_ks_rotated_vs_fresh(probe_samples, theta_deg, rotate_rotation):
    """Per-dim KS test: rho_A(theta) z must be distributed like a fresh draw
    (deterministic under the module-scoped seeds, so not flaky)."""
    z, fresh = probe_samples
    rho_T = action_rho_matrix(np.deg2rad(theta_deg), rotate_rotation).T
    z_rot = z.astype(np.float64) @ rho_T
    for d in range(7):
        _, p = sps.ks_2samp(z_rot[:N_KS, d], fresh[:N_KS, d])
        assert p > 1e-3, f"dim {d}: KS p={p} at theta={theta_deg} rot={rotate_rotation}"


def test_block_scales_are_applied():
    torch.manual_seed(2)
    source = BlockIsotropicSource(7, libero_action_blocks(), scales=UNEQUAL_SCALES)
    expected = torch.tensor([0.7, 0.7, 1.3, 0.5, 0.5, 0.5, 2.0])
    assert torch.equal(source.std, expected)
    z = source.sample((50_000, 7))
    np.testing.assert_allclose(
        z.std(dim=0).numpy(), expected.numpy(), rtol=0.02
    )


# ── physical_so2 warp ────────────────────────────────────────────────────────

def test_physical_so2_matches_diffusion_equi_noise():
    """Under per-dim min-max scales (scale = 2/R) the flow-side source must
    reproduce the diffusion-side EquiNoise(mode='physical_so2') stds."""
    R = np.array([0.06, 0.02, 0.04, 0.5, 0.44, 0.52, 2.0])  # asymmetric xy
    blocks = libero_action_blocks()
    source = BlockIsotropicSource(
        7,
        blocks,
        warp_correction="physical_so2",
        normalizer_scale=torch.from_numpy(2.0 / R).float(),
    )
    noise = EquiNoise(7, to_noise_blocks(blocks, ranges=R), mode="physical_so2")
    torch.testing.assert_close(source.std, noise.std, rtol=1e-5, atol=1e-6)
    # the correction really is anisotropic on the xy pair
    assert not torch.allclose(source.std_correction, torch.ones(7))


def test_physical_so2_degrades_to_identity_under_tied_scales():
    """Tied (group-compatible) scales within every rho1 block => the warp
    correction is identity (up to float32 exp/log round-off <= 1 ulp)."""
    scale = torch.tensor([0.03737, 0.03737, 3.3, 1.2, 1.2, 1.2, 1.0])
    source = BlockIsotropicSource(
        7, libero_action_blocks(), warp_correction="physical_so2",
        normalizer_scale=scale,
    )
    assert torch.allclose(source.std_correction, torch.ones(7), atol=1e-6)
    assert torch.allclose(source.std, torch.ones(7), atol=1e-6)


def test_physical_so2_requires_normalizer_scale():
    with pytest.raises(AssertionError):
        BlockIsotropicSource(7, libero_action_blocks(), warp_correction="physical_so2")


# ── exact-randn contracts ────────────────────────────────────────────────────

def test_gaussian_source_bitwise_equals_randn():
    torch.manual_seed(3)
    a = GaussianSource().sample((8, 16, 7), dtype=torch.float32)
    torch.manual_seed(3)
    b = torch.randn((8, 16, 7), dtype=torch.float32)
    assert torch.equal(a, b)


def test_block_isotropic_unit_scales_bitwise_equals_randn():
    source = BlockIsotropicSource(7, libero_action_blocks())
    assert torch.equal(source.std, torch.ones(7))
    torch.manual_seed(4)
    a = source.sample((8, 16, 7))
    torch.manual_seed(4)
    b = torch.randn((8, 16, 7))
    assert torch.equal(a, b)


def test_level1_stub_zero_init_equals_randn():
    torch.manual_seed(5)
    source = Level1ScaleHeadSource(7, libero_action_blocks(), cond_dim=32)
    cond = torch.randn(8, 2, 32)

    torch.manual_seed(6)
    a = source.sample((8, 16, 7), dtype=torch.float32, cond_feat=cond)
    torch.manual_seed(6)
    b = torch.randn((8, 16, 7), dtype=torch.float32)
    assert torch.equal(a, b)
    assert a.shape == (8, 16, 7) and a.dtype == torch.float32

    # no cond_feat -> plain randn path, same contract
    torch.manual_seed(7)
    c = source.sample((8, 16, 7), dtype=torch.float32)
    torch.manual_seed(7)
    d = torch.randn((8, 16, 7), dtype=torch.float32)
    assert torch.equal(c, d)


def test_source_shape_and_dtype_contract():
    source = BlockIsotropicSource(7, libero_action_blocks(), scales=UNEQUAL_SCALES)
    z = source.sample((3, 5, 7), dtype=torch.float64)
    assert z.shape == (3, 5, 7) and z.dtype == torch.float64
    with pytest.raises(AssertionError):
        source.sample((3, 5, 6))


# ── build_source factory ─────────────────────────────────────────────────────

def test_build_source_disabled_returns_none():
    assert build_source(None, 7) is None
    assert build_source({}, 7) is None
    assert build_source({"enable": False, "kind": "block_isotropic"}, 7) is None


def test_build_source_kinds():
    assert isinstance(build_source({"enable": True, "kind": "gaussian"}, 7), GaussianSource)
    src = build_source(
        {"enable": True, "kind": "block_isotropic", "scales": UNEQUAL_SCALES}, 7
    )
    assert isinstance(src, BlockIsotropicSource)
    assert torch.equal(src.std, torch.tensor([0.7, 0.7, 1.3, 0.5, 0.5, 0.5, 2.0]))
    with pytest.raises(ValueError):
        build_source({"enable": True, "kind": "nope"}, 7)


def test_source_buffers_are_non_persistent():
    source = BlockIsotropicSource(7, libero_action_blocks(), scales=UNEQUAL_SCALES)
    assert source.state_dict() == {}
