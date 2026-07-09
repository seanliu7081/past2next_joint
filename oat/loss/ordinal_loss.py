"""
Ordinal-aware loss functions for discrete token prediction with FSQ.

FSQ tokens have a natural distance/ordering structure. These losses
exploit this structure to penalize "far wrong" predictions more than
"near wrong" predictions.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List


def decompose_indices(indices: Tensor, levels: Tensor, basis: Tensor) -> List[Tensor]:
    """
    Decompose flat codebook indices into per-dimension codes.

    Args:
        indices: [...] flat indices in {0, ..., prod(levels)-1}
        levels: tensor [K] e.g. [8, 5, 5, 5]
        basis: tensor [K] e.g. [1, 8, 40, 200]

    Returns:
        list of K tensors, each [...] with per-dim codes in {0, ..., L_k-1}
    """
    per_dim = []
    for k in range(len(levels)):
        codes_k = (indices // basis[k]) % levels[k]
        per_dim.append(codes_k.long())
    return per_dim


def compose_indices(per_dim_codes: List[Tensor], basis: Tensor) -> Tensor:
    """
    Inverse of decompose_indices: per-dim codes -> flat index.

    Args:
        per_dim_codes: list of K tensors, each [...] integer codes
        basis: tensor [K]

    Returns:
        [...] flat indices
    """
    result = torch.zeros_like(per_dim_codes[0], dtype=torch.long)
    for code, b in zip(per_dim_codes, basis):
        result = result + code.long() * b.long()
    return result


def distance_weighted_ce_loss(
    logits: Tensor,
    targets: Tensor,
    codebook_embeddings: Tensor,
) -> Tensor:
    """
    Expected squared distance under the predicted distribution.

    loss = sum_j p(j) * ||embed(j) - embed(target)||^2

    This acts as a soft penalty: if the model puts probability on
    tokens near the target, the loss is small; if it puts probability
    on distant tokens, the loss is large.

    Args:
        logits: [N, V] raw logits (V = codebook_size, NOT including BOS)
        targets: [N] integer target indices in {0, ..., V-1}
        codebook_embeddings: [V, D] embedding for each codebook index

    Returns:
        Scalar loss (mean over batch)

    Notes:
        - Caller must exclude the BOS logit column before passing logits
        - Gradients flow through softmax(logits) -> differentiable
    """
    # probs: [N, V]
    probs = F.softmax(logits, dim=-1)

    # target embeddings: [N, D]
    target_emb = codebook_embeddings[targets]

    # squared distances from each codebook entry to the target: [N, V]
    # codebook_embeddings is [V, D], target_emb is [N, D]
    # distances[i, j] = ||embed(j) - embed(target_i)||^2
    distances = (codebook_embeddings.unsqueeze(0) - target_emb.unsqueeze(1)).pow(2).sum(dim=-1)

    # expected distance under predicted distribution
    loss = (probs * distances).sum(dim=-1).mean()
    return loss


def emd_loss_1d(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Earth Mover's Distance (Wasserstein-1) for a single 1D ordered distribution.

    Closed-form for 1D: W1(P, Q) = sum_i |CDF_P(i) - CDF_Q(i)|

    Args:
        logits: [N, L] raw logits for L ordered categories
        targets: [N] integer targets in {0, ..., L-1}

    Returns:
        Scalar loss (mean over batch)
    """
    N, L = logits.shape

    # predicted CDF
    probs = F.softmax(logits, dim=-1)       # [N, L]
    pred_cdf = probs.cumsum(dim=-1)          # [N, L]

    # target CDF: step function jumping at the target index
    target_onehot = torch.zeros_like(probs)
    target_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    target_cdf = target_onehot.cumsum(dim=-1)  # [N, L]

    # W1 = sum of |CDF differences|
    loss = (pred_cdf - target_cdf).abs().sum(dim=-1).mean()
    return loss


def per_dim_ce_loss(
    logits_per_dim: List[Tensor],
    targets_per_dim: List[Tensor],
) -> Tensor:
    """
    Standard CE applied independently per FSQ dimension, averaged.

    Args:
        logits_per_dim: list of K tensors, each [N, L_k]
        targets_per_dim: list of K tensors, each [N] in {0, ..., L_k-1}

    Returns:
        Scalar loss
    """
    losses = []
    for logits_k, targets_k in zip(logits_per_dim, targets_per_dim):
        losses.append(F.cross_entropy(logits_k, targets_k))
    return torch.stack(losses).mean()


def per_dim_emd_loss(
    logits_per_dim: List[Tensor],
    targets_per_dim: List[Tensor],
) -> Tensor:
    """
    EMD loss applied independently per FSQ dimension, averaged.

    Args:
        logits_per_dim: list of K tensors, each [N, L_k]
        targets_per_dim: list of K tensors, each [N] in {0, ..., L_k-1}

    Returns:
        Scalar loss
    """
    losses = []
    for logits_k, targets_k in zip(logits_per_dim, targets_per_dim):
        losses.append(emd_loss_1d(logits_k, targets_k))
    return torch.stack(losses).mean()
