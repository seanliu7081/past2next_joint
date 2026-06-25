"""Correctness tests for the ZHill (Product Hilbert FSQ) tokenizer."""

import torch
import sys
sys.path.insert(0, "/workspace/oat")

from oat.tokenizer.zhill.quantizer.hilbert import build_hilbert_lut
from oat.tokenizer.zhill.quantizer.product_hilbert_fsq import ProductHilbertFSQ
from oat.tokenizer.oat.quantizer.fsq import FSQ


def test_lut_roundtrip():
    """Test 1: LUT roundtrip for various grid sizes."""
    print("Test 1: LUT roundtrip ... ", end="")
    for L0, L1 in [(5, 5), (8, 5), (8, 8), (3, 7), (1, 1), (2, 3)]:
        g2h, h2g = build_hilbert_lut(L0, L1)
        assert g2h.shape == (L0, L1), f"g2h shape mismatch for ({L0},{L1})"
        assert h2g.shape == (L0 * L1, 2), f"h2g shape mismatch for ({L0},{L1})"
        for idx in range(L0 * L1):
            x, y = h2g[idx].tolist()
            assert g2h[x, y].item() == idx, (
                f"Roundtrip failed at idx={idx} for ({L0},{L1})"
            )
    print("PASSED")


def test_codebook_size_preservation():
    """Test 2: Codebook size is identical between FSQ and ProductHilbertFSQ."""
    print("Test 2: Codebook size preservation ... ", end="")
    for levels in [[8, 5, 5, 5], [3, 3, 3], [8, 8], [5, 5, 5, 5, 5]]:
        fsq = FSQ(levels=levels)
        phfsq = ProductHilbertFSQ(levels=levels)
        assert fsq.codebook_size == phfsq.codebook_size, (
            f"Codebook size mismatch for levels={levels}: "
            f"FSQ={fsq.codebook_size}, PHFSQ={phfsq.codebook_size}"
        )
    print("PASSED")


def test_quantization_output_identity():
    """Test 3: Quantized values (zhat) are identical — only token indices differ."""
    print("Test 3: Quantization output identity ... ", end="")
    levels = [8, 5, 5, 5]
    fsq = FSQ(levels=levels)
    phfsq = ProductHilbertFSQ(levels=levels)
    fsq.eval()
    phfsq.eval()

    z = torch.randn(4, 8, 4)
    fsq_quant, fsq_tokens = fsq.forward_z(z)
    phfsq_quant, phfsq_tokens = phfsq.forward_z(z)
    assert torch.equal(fsq_quant, phfsq_quant), "Quantized values must be identical!"
    # Tokens differ (different indexing) — this is expected
    print("PASSED")


def test_index_roundtrip():
    """Test 4: Index roundtrip (codes_to_indices -> indices_to_embedding)."""
    print("Test 4: Index roundtrip ... ", end="")
    for levels in [[8, 5, 5, 5], [3, 3, 3], [8, 8], [5, 5, 5, 5, 5]]:
        phfsq = ProductHilbertFSQ(levels=levels)
        phfsq.eval()

        z = torch.randn(4, 8, len(levels))
        quant, tokens = phfsq.forward_z(z)
        reconstructed = phfsq.indices_to_embedding(tokens)
        assert torch.equal(quant, reconstructed), (
            f"indices_to_embedding must recover exact quantized values for levels={levels}"
        )
    print("PASSED")


def test_all_indices_covered():
    """Test 4b: Every index in [0, codebook_size) is reachable."""
    print("Test 4b: All indices covered ... ", end="")
    levels = [8, 5, 5, 5]
    phfsq = ProductHilbertFSQ(levels=levels)
    all_indices = torch.arange(phfsq.codebook_size)
    embeddings = phfsq.indices_to_embedding(all_indices)
    recovered = phfsq.codes_to_indices(embeddings)
    assert torch.equal(recovered.long(), all_indices.long()), "Not all indices round-trip correctly"
    print("PASSED")


def test_hilbert_locality():
    """Test 6: Hilbert locality — consecutive indices map to nearby grid points."""
    print("Test 6: Hilbert locality check ... ", end="")
    for L0, L1 in [(8, 5), (5, 5), (8, 8)]:
        g2h, h2g = build_hilbert_lut(L0, L1)
        for idx in range(h2g.shape[0] - 1):
            x0, y0 = h2g[idx].tolist()
            x1, y1 = h2g[idx + 1].tolist()
            dist = abs(x0 - x1) + abs(y0 - y1)
            assert dist <= 2, (
                f"Hilbert locality violated for ({L0},{L1}): "
                f"idx {idx}->{idx+1}, dist={dist}"
            )
    print("PASSED")


def test_implicit_codebook_shape():
    """Test: implicit_codebook has correct shape."""
    print("Test 7: Implicit codebook shape ... ", end="")
    levels = [8, 5, 5, 5]
    phfsq = ProductHilbertFSQ(levels=levels)
    assert phfsq.implicit_codebook.shape == (1000, 4), (
        f"Expected (1000, 4), got {phfsq.implicit_codebook.shape}"
    )
    print("PASSED")


def test_odd_dimensions():
    """Test: Odd number of dimensions (remainder dim)."""
    print("Test 8: Odd dimensions (remainder) ... ", end="")
    levels = [3, 3, 3]  # 2 paired + 1 remainder
    phfsq = ProductHilbertFSQ(levels=levels)
    phfsq.eval()
    assert phfsq.has_remainder
    assert phfsq.num_pairs == 1
    assert phfsq.codebook_size == 27

    z = torch.randn(2, 4, 3)
    quant, tokens = phfsq.forward_z(z)
    reconstructed = phfsq.indices_to_embedding(tokens)
    assert torch.equal(quant, reconstructed), "Roundtrip failed for odd dimensions"
    print("PASSED")


if __name__ == "__main__":
    test_lut_roundtrip()
    test_codebook_size_preservation()
    test_quantization_output_identity()
    test_index_roundtrip()
    test_all_indices_covered()
    test_hilbert_locality()
    test_implicit_codebook_shape()
    test_odd_dimensions()
    print("\nAll tests passed!")
