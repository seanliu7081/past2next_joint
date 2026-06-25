"""Unit tests for PolarOATTok v2."""

import torch
import torch.nn.functional as F
import math

from oat.tokenizer.polar.polar_decompose import PolarDecompose
from oat.tokenizer.polar_v2.cyclic_vq_product import CyclicVQProduct
from oat.tokenizer.polar_v2.tokenizer import PolarOATTokV2
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ
from oat.model.common.normalizer import LinearNormalizer


def _make_tokenizer():
    """Create PolarOATTokV2 with default config for testing."""
    tok = PolarOATTokV2(
        polar_decompose=PolarDecompose(),
        inv_encoder=RegisterEncoder(
            sample_dim=4, sample_horizon=32, emb_dim=128, head_dim=32,
            depth=1, pdropout=0.0, latent_dim=4, num_registers=8,
        ),
        eq_encoder=RegisterEncoder(
            sample_dim=3, sample_horizon=32, emb_dim=128, head_dim=32,
            depth=1, pdropout=0.0, latent_dim=3, num_registers=8,
        ),
        inv_quantizer=FSQ(levels=[8, 5, 5, 5]),
        eq_quantizer=CyclicVQProduct(n_bins=[24, 12, 8]),
        decoder=SinglePassDecoder(
            sample_dim=7, sample_horizon=32, emb_dim=128, head_dim=32,
            depth=1, pdropout=0.0, token_dropout_mode='pow2',
            latent_dim=7, latent_horizon=8, use_causal_decoder=True,
        ),
        equiv_reg_weight=0.1,
        n_equiv_samples=2,
    )
    # Set up identity-ish normalizer
    normalizer = LinearNormalizer()
    data = {'action': torch.randn(500, 32, 7)}
    normalizer.fit(data, mode='limits', output_max=1.0, output_min=-1.0)
    tok.set_normalizer(normalizer)
    return tok


# ── Test 1: PolarDecompose round-trip ────────────────────────────────────────

def test_polar_roundtrip():
    pd = PolarDecompose()
    actions = torch.randn(100, 32, 7)
    inv, eq, mask = pd(actions)
    recon = pd.inverse(inv, eq)
    non_null = ~mask.any(-1)
    assert torch.allclose(actions[non_null], recon[non_null], atol=1e-5)
    print("PASS: test_polar_roundtrip")


# ── Test 2: CyclicVQ C_N equivariance ───────────────────────────────────────

def test_cyclic_vq_equivariance():
    cvq = CyclicVQProduct(n_bins=[24, 12, 8])
    # Use angles well away from bin boundaries to avoid boundary effects
    angles = torch.linspace(-math.pi + 0.2, math.pi - 0.2, 100).unsqueeze(-1).unsqueeze(-1)
    angles = angles.expand(100, 8, 3).clone()

    _, tokens_orig = cvq(angles)

    # Rotate dim 0 by exactly 1 bin = 2*pi/24
    delta = 2 * math.pi / 24
    rotated = angles.clone()
    rotated[..., 0] = angles[..., 0] + delta
    _, tokens_rot = cvq(rotated)

    # Decompose product indices to per-dim indices for dim 0
    idx_orig_0 = tokens_orig // cvq._basis[0] % 24
    idx_rot_0 = tokens_rot // cvq._basis[0] % 24
    expected = (idx_orig_0 + 1) % 24

    match_rate = (idx_rot_0 == expected).float().mean()
    assert match_rate > 0.95, f"C_N equivariance: only {match_rate:.1%} match"
    print(f"PASS: test_cyclic_vq_equivariance ({match_rate:.1%} exact match)")


# ── Test 3: CyclicVQ product index round-trip ────────────────────────────────

def test_cyclic_vq_index_roundtrip():
    cvq = CyclicVQProduct(n_bins=[24, 12, 8])
    assert cvq.codebook_size == 24 * 12 * 8  # = 2304

    all_indices = torch.arange(cvq.codebook_size)
    embeddings = cvq.indices_to_embedding(all_indices)
    _, reconstructed = cvq(embeddings)
    assert torch.all(reconstructed == all_indices), \
        f"Round-trip failed: {(reconstructed != all_indices).sum()} mismatches"
    print(f"PASS: test_cyclic_vq_index_roundtrip (codebook_size={cvq.codebook_size})")


# ── Test 4: Invariance under rotation ────────────────────────────────────────

def test_inv_tokens_rotation_invariance():
    tok = _make_tokenizer()
    tok.eval()
    actions = torch.randn(4, 32, 7) * 0.3

    with torch.no_grad():
        tokens_orig = tok.tokenize(actions)

    phi = torch.rand(1) * 2 * math.pi
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    rotated = PolarOATTokV2._rotate_actions(actions, cos_phi, sin_phi)

    with torch.no_grad():
        tokens_rot = tok.tokenize(rotated)

    assert torch.all(tokens_orig['inv'] == tokens_rot['inv']), \
        "Invariant tokens changed under rotation!"
    print("PASS: test_inv_tokens_rotation_invariance")


# ── Test 5: Shapes and vocab sizes ──────────────────────────────────────────

def test_shapes_and_vocab():
    tok = _make_tokenizer()
    tok.eval()
    actions = torch.randn(4, 32, 7) * 0.3

    with torch.no_grad():
        tokens = tok.tokenize(actions)
    assert tokens['inv'].shape == (4, 8), f"inv shape: {tokens['inv'].shape}"
    assert tokens['eq'].shape == (4, 8), f"eq shape: {tokens['eq'].shape}"

    with torch.no_grad():
        recon = tok.detokenize(tokens)
    assert recon.shape == (4, 32, 7), f"recon shape: {recon.shape}"

    assert tok.vocab_sizes == {'inv': 1000, 'eq': 2304}
    assert tok.latent_horizon == 8
    print(f"PASS: test_shapes_and_vocab (vocab_sizes={tok.vocab_sizes})")


# ── Test 6: Forward loss and gradient flow ───────────────────────────────────

def test_forward_loss():
    tok = _make_tokenizer()
    tok.train()

    # The decoder's LinearHead uses zero-init by default (same as original OATTok).
    # This means encoder gradients are zero at init — they start flowing once the
    # head weights become non-zero after a few optimizer steps.
    # To verify gradient flow works, initialize head with xavier weights.
    torch.nn.init.xavier_uniform_(tok.decoder.head.proj.weight)

    batch = {'action': torch.randn(4, 32, 7) * 0.3}
    loss = tok(batch)

    assert loss.dim() == 0, "Loss should be scalar"
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.requires_grad, "Loss should require grad"

    loss.backward()

    # Check gradients flow to both encoders
    inv_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in tok.inv_encoder.parameters())
    eq_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in tok.eq_encoder.parameters())
    dec_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in tok.decoder.parameters())

    assert inv_has_grad, "inv_encoder has no gradients!"
    assert eq_has_grad, "eq_encoder has no gradients!"
    assert dec_has_grad, "decoder has no gradients!"

    print(f"PASS: test_forward_loss (loss={loss.item():.4f})")


# ── Test 7: Equiv reg loss not zero ──────────────────────────────────────────

def test_equiv_reg():
    tok = _make_tokenizer()
    tok.train()
    # Xavier-init head so decoder output is nonzero
    torch.nn.init.xavier_uniform_(tok.decoder.head.proj.weight)

    actions = torch.randn(4, 32, 7) * 0.3
    nactions = tok.normalizer['action'].normalize(actions)

    loss_equiv = tok._equiv_reg_loss(nactions)
    assert loss_equiv.item() > 0, "Equiv reg loss should be nonzero for untrained model"
    print(f"PASS: test_equiv_reg (loss_equiv={loss_equiv.item():.4f})")


# ── Test 8: All trainable params get gradients (DDP compat) ─────────────────

def test_all_params_get_gradients():
    tok = _make_tokenizer()
    tok.train()
    # Xavier-init the decoder head so gradients flow through to encoders
    torch.nn.init.xavier_uniform_(tok.decoder.head.proj.weight)

    batch = {'action': torch.randn(4, 32, 7) * 0.3}
    loss = tok(batch)
    loss.backward()

    no_grad = [n for n, p in tok.named_parameters()
               if p.requires_grad and p.grad is None]
    assert len(no_grad) == 0, f"Params without grad (DDP will fail): {no_grad}"
    print("PASS: test_all_params_get_gradients")


# ── Test 9: AR model works with v2 vocab sizes ──────────────────────────────

def test_ar_model_integration():
    from oat.model.autoregressive.polar_transformer_cache import PolarAutoRegressiveModel

    vocab_sizes = {'inv': 1000, 'eq': 2304}
    model = PolarAutoRegressiveModel(
        vocab_sizes=vocab_sizes,
        max_seq_len=9, max_cond_len=2, cond_dim=64,
        n_layer=2, n_head=2, n_emb=64,
    )

    B = 2
    token_dict = {
        'inv': torch.randint(0, 1000, (B, 8)),
        'eq': torch.randint(0, 2304, (B, 8)),
    }
    cond = torch.randn(B, 2, 64)

    # Training forward
    logits = model(token_dict, cond)
    assert logits['inv'].shape == (B, 8, 1001)
    assert logits['eq'].shape == (B, 8, 2305)

    # Loss is sum of 2 CEs — should be roughly log(1001) + log(2305) ≈ 14.6
    total_loss = sum(
        F.cross_entropy(logits[n].reshape(-1, logits[n].size(-1)), token_dict[n].reshape(-1))
        for n in vocab_sizes
    )
    assert 10 < total_loss.item() < 25, f"Loss out of expected range: {total_loss.item()}"

    # Generation
    model.eval()
    prefix = {n: torch.full((B, 1), vs, dtype=torch.long) for n, vs in vocab_sizes.items()}
    gen = model.generate(prefix, cond, max_new_tokens=8, temperature=1.0, top_k=10)
    assert gen['inv'].shape == (B, 9)
    assert gen['eq'].shape == (B, 9)

    print(f"PASS: test_ar_model_integration (loss={total_loss.item():.2f})")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_polar_roundtrip()
    test_cyclic_vq_equivariance()
    test_cyclic_vq_index_roundtrip()
    test_inv_tokens_rotation_invariance()
    test_shapes_and_vocab()
    test_forward_loss()
    test_equiv_reg()
    test_all_params_get_gradients()
    test_ar_model_integration()
    print("\n" + "=" * 60)
    print("ALL 9 TESTS PASSED")
    print("=" * 60)
