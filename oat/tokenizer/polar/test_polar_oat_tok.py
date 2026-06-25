"""Unit tests for PolarOATTok tokenizer."""

import torch
import torch.nn.functional as F
import math
import pytest

from oat.tokenizer.polar.polar_decompose import PolarDecompose
from oat.tokenizer.polar.cyclic_vq import CyclicVQ
from oat.tokenizer.polar.tokenizer import PolarOATTok


# ── Test 1: PolarDecompose round-trip ────────────────────────────────────────

def test_polar_roundtrip():
    """PolarDecompose.forward() then .inverse() should recover original action."""
    polar = PolarDecompose()
    actions = torch.randn(100, 7)
    inv, eq, mask = polar(actions)
    reconstructed = polar.inverse(inv, eq)

    # For non-null entries (r > 0), should be exact up to float precision
    non_null = ~mask.any(dim=-1)
    if non_null.any():
        assert torch.allclose(
            actions[non_null], reconstructed[non_null], atol=1e-5
        ), "Polar round-trip failed for non-null entries"

    # For null entries, dx=dy=0 should still hold (r=0 => dx=dy=0 regardless of theta)
    null_trans = mask[:, 0]
    if null_trans.any():
        assert torch.allclose(
            reconstructed[null_trans, 0],
            torch.zeros_like(reconstructed[null_trans, 0]), atol=1e-6
        ), "Null trans: dx should be 0"
        assert torch.allclose(
            reconstructed[null_trans, 1],
            torch.zeros_like(reconstructed[null_trans, 1]), atol=1e-6
        ), "Null trans: dy should be 0"

    print("PASS: test_polar_roundtrip")


def test_polar_roundtrip_batched():
    """Round-trip works with (B, T, 7) shape."""
    polar = PolarDecompose()
    actions = torch.randn(8, 32, 7)
    inv, eq, mask = polar(actions)
    reconstructed = polar.inverse(inv, eq)

    non_null = ~mask.any(dim=-1)
    assert torch.allclose(
        actions[non_null], reconstructed[non_null], atol=1e-5
    ), "Batched polar round-trip failed"
    print("PASS: test_polar_roundtrip_batched")


# ── Test 2: CyclicVQ C_N equivariance ───────────────────────────────────────

def test_cyclic_vq_equivariance():
    """Rotating by exactly k bins should shift indices by k."""
    n_bins = 24
    cvq = CyclicVQ(n_bins=[n_bins, 12, 16])

    theta = torch.linspace(-math.pi + 0.01, math.pi - 0.01, 1000)
    angles = torch.zeros(1000, 3)
    angles[:, 0] = theta

    _, idx_original = cvq(angles)
    idx_orig_0 = idx_original[:, 0]

    # Rotate by exactly 1 bin = 2*pi/24
    delta = 2 * math.pi / n_bins
    angles_rotated = angles.clone()
    angles_rotated[:, 0] = theta + delta
    # Wrap to [-pi, pi]
    angles_rotated[:, 0] = torch.atan2(
        torch.sin(angles_rotated[:, 0]),
        torch.cos(angles_rotated[:, 0])
    )

    _, idx_rotated = cvq(angles_rotated)
    idx_rot_0 = idx_rotated[:, 0]

    expected = (idx_orig_0 + 1) % n_bins
    assert torch.all(idx_rot_0 == expected), \
        f"C_N equivariance broken! Mismatches: {(idx_rot_0 != expected).sum().item()}"
    print("PASS: test_cyclic_vq_equivariance")


def test_cyclic_vq_geodesic():
    """Verify geodesic distance is correct."""
    d = CyclicVQ.geodesic_distance(
        torch.tensor([0.0, math.pi - 0.1]),
        torch.tensor([0.1, -math.pi + 0.2])
    )
    assert torch.allclose(d[0], torch.tensor(0.1), atol=1e-5)
    # Wrapping: distance between pi-0.1 and -pi+0.2 = 0.3 (going through -pi/pi boundary)
    assert torch.allclose(d[1], torch.tensor(0.3), atol=1e-5)
    print("PASS: test_cyclic_vq_geodesic")


# ── Test 3: Invariance of invariant path ─────────────────────────────────────

def test_invariant_path_rotation_invariance():
    """Rotating action by any angle should not change invariant tokens."""
    tok = PolarOATTok(
        fsq_levels=[8, 10, 4, 3],
        n_bins_trans=24, n_bins_rot=12, n_bins_yaw=16,
    )
    tok.eval()

    actions = torch.randn(100, 7) * 0.3

    with torch.no_grad():
        _, tokens_original = tok._encode_step(actions)

    # Apply random SO(2) rotation to (dx, dy) and (droll, dpitch)
    phi = torch.rand(1) * 2 * math.pi
    cos_p, sin_p = torch.cos(phi), torch.sin(phi)
    rotated = actions.clone()
    rotated[:, 0] = cos_p * actions[:, 0] - sin_p * actions[:, 1]
    rotated[:, 1] = sin_p * actions[:, 0] + cos_p * actions[:, 1]
    rotated[:, 3] = cos_p * actions[:, 3] - sin_p * actions[:, 4]
    rotated[:, 4] = sin_p * actions[:, 3] + cos_p * actions[:, 4]

    with torch.no_grad():
        _, tokens_rotated = tok._encode_step(rotated)

    assert torch.all(tokens_original['inv'] == tokens_rotated['inv']), \
        "Invariant tokens changed under SO(2) rotation!"
    print("PASS: test_invariant_path_rotation_invariance")


# ── Test 4: Full encode-decode reconstruction quality ────────────────────────

def test_reconstruction_quality():
    """Reconstruction MSE should be finite and computable (untrained)."""
    tok = PolarOATTok(
        fsq_levels=[8, 10, 4, 3],
        n_bins_trans=24, n_bins_rot=12, n_bins_yaw=16,
    )
    tok.eval()

    actions = torch.randn(100, 7) * 0.3
    with torch.no_grad():
        latents, tokens = tok._encode_step(actions)
        recon = tok._decode_tokens(tokens, inv_quant=latents['inv'])

    mse = F.mse_loss(recon, actions)
    assert torch.isfinite(mse), "Reconstruction MSE is not finite"
    print(f"PASS: test_reconstruction_quality (untrained MSE={mse.item():.6f})")


# ── Test 5: Null token handling ──────────────────────────────────────────────

def test_null_tokens():
    """Zero-motion actions should produce NULL tokens for angles."""
    tok = PolarOATTok(
        fsq_levels=[8, 10, 4, 3],
        n_bins_trans=24, n_bins_rot=12, n_bins_yaw=16,
    )
    tok.eval()

    # Action with zero translation AND zero rotation
    actions = torch.zeros(10, 7)
    actions[:, 2] = 0.5   # nonzero dz
    actions[:, 5] = 0.1   # nonzero dyaw
    actions[:, 6] = 1.0   # grip

    with torch.no_grad():
        _, tokens = tok._encode_step(actions)

    # theta_trans should be NULL (index = n_bins_trans = 24)
    assert torch.all(tokens['theta_trans'] == 24), \
        f"theta_trans should be NULL (24), got {tokens['theta_trans']}"
    # theta_rot should be NULL (index = n_bins_rot = 12)
    assert torch.all(tokens['theta_rot'] == 12), \
        f"theta_rot should be NULL (12), got {tokens['theta_rot']}"
    # yaw should NOT be null
    assert torch.all(tokens['yaw'] < 16), \
        "yaw should never be NULL"

    print("PASS: test_null_tokens")


# ── Test 6: Vocab sizes consistency ──────────────────────────────────────────

def test_vocab_sizes():
    """Verify vocab sizes match expected values."""
    tok = PolarOATTok(
        fsq_levels=[8, 10, 4, 3],
        n_bins_trans=24, n_bins_rot=12, n_bins_yaw=16,
    )

    vs = tok.vocab_sizes
    assert vs['inv'] == 8 * 10 * 4 * 3, f"inv vocab wrong: {vs['inv']}"  # = 960
    assert vs['theta_trans'] == 24 + 1, f"theta_trans vocab wrong: {vs['theta_trans']}"  # = 25
    assert vs['theta_rot'] == 12 + 1, f"theta_rot vocab wrong: {vs['theta_rot']}"  # = 13
    assert vs['yaw'] == 16, f"yaw vocab wrong: {vs['yaw']}"

    effective = tok.effective_vocab_size
    expected = 960 * 25 * 13 * 16
    assert effective == expected, f"Effective vocab wrong: {effective} != {expected}"
    print(f"PASS: test_vocab_sizes (effective={effective:,})")


# ── Test 7: Forward pass produces valid loss ─────────────────────────────────

def test_forward_loss():
    """forward() should produce a finite, differentiable scalar loss."""
    tok = PolarOATTok(
        fsq_levels=[8, 10, 4, 3],
        n_bins_trans=24, n_bins_rot=12, n_bins_yaw=16,
    )

    # Simulate a batch (without normalizer, use identity normalization)
    from oat.model.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    # Set up identity normalization
    data = {'action': torch.randn(1000, 32, 7)}
    normalizer.fit(data, mode='limits', output_max=1.0, output_min=-1.0)
    tok.set_normalizer(normalizer)

    batch = {'action': torch.randn(8, 32, 7) * 0.3}
    loss = tok(batch)

    assert loss.dim() == 0, "Loss should be scalar"
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.requires_grad, "Loss should require grad"

    # Backward should work
    loss.backward()
    print(f"PASS: test_forward_loss (loss={loss.item():.6f})")


# ── Test 8: CyclicVQ indices_to_angles round-trip ────────────────────────────

def test_cyclic_vq_indices_roundtrip():
    """indices_to_angles(quantize(angles)) should match quantized angles."""
    cvq = CyclicVQ(n_bins=[24, 12, 16])

    angles = (torch.rand(50, 3) * 2 - 1) * math.pi  # uniform in [-pi, pi]
    quantized, indices = cvq(angles)
    recovered = cvq.indices_to_angles(indices)

    assert torch.allclose(quantized, recovered, atol=1e-5), \
        "CyclicVQ indices round-trip failed"
    print("PASS: test_cyclic_vq_indices_roundtrip")


# ── Test 9: All trainable parameters receive gradients ───────────────────────

def test_all_params_get_gradients():
    """Every trainable parameter must receive gradients (DDP compatibility)."""
    tok = PolarOATTok(
        fsq_levels=[8, 5, 5, 5],
        n_bins_trans=24, n_bins_rot=12, n_bins_yaw=16,
    )
    from oat.model.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    data = {'action': torch.randn(1000, 32, 7)}
    normalizer.fit(data, mode='limits', output_max=1.0, output_min=-1.0)
    tok.set_normalizer(normalizer)

    batch = {'action': torch.randn(4, 32, 7) * 0.3}
    loss = tok(batch)
    loss.backward()

    no_grad_params = []
    for name, p in tok.named_parameters():
        if p.requires_grad and p.grad is None:
            no_grad_params.append(name)

    assert len(no_grad_params) == 0, \
        f"Parameters without gradients (DDP will fail): {no_grad_params}"
    print("PASS: test_all_params_get_gradients")


# ── Run all tests ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_polar_roundtrip()
    test_polar_roundtrip_batched()
    test_cyclic_vq_equivariance()
    test_cyclic_vq_geodesic()
    test_invariant_path_rotation_invariance()
    test_reconstruction_quality()
    test_null_tokens()
    test_vocab_sizes()
    test_forward_loss()
    test_cyclic_vq_indices_roundtrip()
    test_all_params_get_gradients()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    # Print integration notes
    print("\n" + "=" * 60)
    print("INTEGRATION NOTES FOR POLICY CHANGES")
    print("=" * 60)
    print("""
PolarOATTok exposes a factored token interface. To integrate with the AR policy:

1. MULTIPLE CLASSIFICATION HEADS:
   The policy needs one head per factored token type:
     - inv_head:       Linear(n_emb, vocab_sizes['inv'])        # 960
     - theta_trans_head: Linear(n_emb, vocab_sizes['theta_trans']) # 25 (+NULL)
     - theta_rot_head:  Linear(n_emb, vocab_sizes['theta_rot'])   # 13 (+NULL)
     - yaw_head:       Linear(n_emb, vocab_sizes['yaw'])         # 16

2. MULTIPLE EMBEDDING TABLES:
   For encoding past tokens back into the transformer:
     - inv_emb:       Embedding(vocab_sizes['inv'], n_emb)
     - theta_trans_emb: Embedding(vocab_sizes['theta_trans'], n_emb)
     - theta_rot_emb:  Embedding(vocab_sizes['theta_rot'], n_emb)
     - yaw_emb:       Embedding(vocab_sizes['yaw'], n_emb)
   At each position, sum or concat all 4 embeddings.

3. TOTAL LOSS:
   loss = CE(inv_logits, inv_targets)
        + CE(theta_trans_logits, theta_trans_targets)
        + CE(theta_rot_logits, theta_rot_targets)
        + CE(yaw_logits, yaw_targets)
   Optionally with per-head weights.

4. GENERATION:
   At each AR step, predict all 4 heads in parallel (they share the
   same transformer hidden state for that position).

5. tokenize() returns Dict[str, Tensor] instead of a single Tensor.
   detokenize() accepts Dict[str, Tensor].

6. latent_horizon = sample_horizon (32), not 8 as in OATTok.
   The AR sequence length changes from 8 to 32 token positions.
""")
