"""
SpectralBasisEncoder: DCT-initialized learnable orthogonal basis encoder.

Drop-in replacement for RegisterEncoder in the OAT tokenizer pipeline.

Design principles:
  1. Frequency-domain compression: DCT basis captures 96.6% energy at K=3,
     99.1% at K=8 for typical robot action trajectories.
  2. Token independence (ActionCodec, 2026): No inter-token self-attention.
     Each latent token is produced by an independent inner product with a
     learned basis function, maximizing VLA robustness to token perturbation.
  3. Natural coarse-to-fine ordering: Low-frequency basis functions carry more
     energy, aligning with OAT's nested dropout for progressive decoding.
  4. High overlap rate: Smooth basis functions ensure temporally adjacent
     action chunks map to similar latent codes, improving AR policy training.

Interface contract (identical to RegisterEncoder):
  Input:  sample [B, T, sample_dim]     (T=sample_horizon, e.g. 32)
  Output: latents [B, K, latent_dim]    (K=num_basis, e.g. 8; latent_dim=4 for FSQ)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


def build_dct_basis(num_basis: int, sample_horizon: int) -> torch.Tensor:
    """
    Construct the first K basis vectors of the Type-II DCT, L2-normalized.

    The DCT-II basis for an N-point signal:
        phi_0[n] = 1 / sqrt(N)
        phi_k[n] = sqrt(2/N) * cos(pi * k * (2n+1) / (2N)),   k >= 1

    These are eigenfunctions of the discrete Laplacian on [0, N-1] with
    Neumann boundary conditions, and are the asymptotically optimal basis
    for stationary Gaussian processes (equivalent to KLT).

    Args:
        num_basis: Number of basis vectors K.
        sample_horizon: Length of the time axis T (e.g. 32).

    Returns:
        Tensor of shape [K, T], orthonormal rows.
    """
    n = torch.arange(sample_horizon, dtype=torch.float32)
    basis = []
    for k in range(num_basis):
        if k == 0:
            bk = torch.ones(sample_horizon) / math.sqrt(sample_horizon)
        else:
            bk = math.sqrt(2.0 / sample_horizon) * torch.cos(
                math.pi * k * (2 * n + 1) / (2 * sample_horizon)
            )
        basis.append(bk)
    return torch.stack(basis, dim=0)  # [K, T]


class SpectralBasisEncoder(nn.Module):
    """
    Encode action chunks via learned spectral projection + linear mapping.

    Pipeline:
        action [B, T, D]
        -> spectral coeffs:  c_k = sum_t phi_k(t) * x(t)   -> [B, K, D]
        -> linear projection: z_k = W * c_k + b             -> [B, K, latent_dim]
        -> layer norm (per-token)                            -> [B, K, latent_dim]
        -> (output to FSQ quantizer)

    Args:
        sample_dim:      Action dimensionality D (e.g. 7 for pos+rot+grip).
        sample_horizon:  Temporal length T of the action chunk (e.g. 32).
        latent_dim:      Output dimension per token (must match FSQ dim, e.g. 4).
        num_basis:       Number of basis functions K (replaces num_registers).
        learnable_basis: If True, basis functions are nn.Parameter (gradient-updated).
                         If False, basis is a fixed buffer (pure DCT, no learning).
        ortho_weight:    Weight for soft orthogonality regularization loss.
                         Set to 0.0 to disable. Recommended: 0.01-0.1.
    """

    def __init__(
        self,
        # sample attrs (same interface as RegisterEncoder)
        sample_dim: int,
        sample_horizon: int,
        # latent args
        latent_dim: int,
        num_basis: int,
        # spectral-specific args
        learnable_basis: bool = True,
        ortho_weight: float = 0.01,
    ):
        super().__init__()

        # -- Basis functions [K, T] ------------------------------------------
        dct_basis = build_dct_basis(num_basis, sample_horizon)
        if learnable_basis:
            self.basis = nn.Parameter(dct_basis)
        else:
            self.register_buffer("basis", dct_basis)

        # -- Coefficient -> latent projection ---------------------------------
        # Maps per-token coefficients from action-space dim (D) to latent dim.
        # This is where cross-dimension mixing happens (position <-> rotation).
        # self.coeff_proj = nn.Linear(sample_dim, latent_dim)
        # self.coeff_proj = nn.Conv1d(sample_dim, latent_dim, kernel_size=1)
        self.coeff_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sample_dim, 32),
                nn.GELU(),
                nn.Linear(32, latent_dim),
            )
            for _ in range(num_basis)
        ])

        # -- Per-token normalization ------------------------------------------
        # LayerNorm centers and scales each token's latent vector,
        # producing a distribution more amenable to FSQ quantization.
        # self.latent_norm = nn.LayerNorm(latent_dim)

        # -- Store config -----------------------------------------------------
        self.sample_dim = sample_dim
        self.sample_horizon = sample_horizon
        self.latent_dim = latent_dim
        self.num_basis = num_basis  # analogous to num_registers
        self.ortho_weight = ortho_weight
        self.learnable_basis = learnable_basis

        # Keep a frozen copy of DCT basis for analysis
        self.register_buffer("_dct_basis_ref", dct_basis.clone())

        self._log_init()

    def _log_init(self):
        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"SpectralBasisEncoder initialized:\n"
            f"  basis: [{self.num_basis}, {self.sample_horizon}] "
            f"({'learnable' if self.learnable_basis else 'fixed DCT'})\n"
            f"  projection: {self.sample_dim} -> {self.latent_dim}\n"
            f"  ortho_weight: {self.ortho_weight}\n"
            f"  total params: {n_params:,}"
        )

    # -- Forward --------------------------------------------------------------

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample: Normalized actions [B, T, D].

        Returns:
            latents: [B, K, latent_dim], ready for FSQ quantization.
        """
        # Spectral projection: inner product with each basis function
        # basis: [K, T],  sample: [B, T, D]  ->  coeffs: [B, K, D]
        coeffs = torch.einsum("kt, btd -> bkd", self.basis, sample)

        # Project action-space coefficients to latent space
        # [B, K, D] -> [B, K, latent_dim]
        # latents = self.coeff_proj(coeffs)
        latents = torch.stack([
            self.coeff_projs[k](coeffs[:, k])
            for k in range(self.num_basis)
        ], dim=1)  # [B, K, latent_dim]

        # Per-token normalization for FSQ-friendly distribution
        # latents = self.latent_norm(latents)

        return latents

    # -- Auxiliary losses -----------------------------------------------------

    def ortho_loss(self) -> torch.Tensor:
        """
        Soft orthogonality regularization on the basis matrix.

        Penalizes deviation of Phi @ Phi^T from identity:
            L_ortho = || Phi @ Phi^T - I ||^2_F / K^2

        This encourages the basis to stay near the Stiefel manifold V_K(R^T)
        without enforcing a hard constraint, allowing the optimizer to find
        a good trade-off between orthogonality and reconstruction quality.

        Returns:
            Scalar loss tensor (0 if basis is not learnable or weight is 0).
        """
        if not self.learnable_basis or self.ortho_weight == 0.0:
            return torch.tensor(0.0, device=self.basis.device)

        gram = self.basis @ self.basis.T  # [K, K]
        eye = torch.eye(self.num_basis, device=gram.device, dtype=gram.dtype)
        loss = F.mse_loss(gram, eye)
        return self.ortho_weight * loss

    def tcl_loss(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negatives: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Time Contrastive Loss (inspired by ActionCodec).

        Encourages temporally adjacent action chunks to have similar latent
        representations, which directly increases the overlap rate (OR).

        Args:
            z_anchor:    Pooled latent of current chunk [B, d].
            z_positive:  Pooled latent of adjacent chunk [B, d].
            z_negatives: Pooled latents of other chunks  [B, N, d] or [N, d].

        Returns:
            Scalar InfoNCE loss.
        """
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_positive = F.normalize(z_positive, dim=-1)
        z_negatives = F.normalize(z_negatives, dim=-1)

        # Positive similarity
        pos_sim = (z_anchor * z_positive).sum(dim=-1) / temperature  # [B]

        # Negative similarities
        if z_negatives.dim() == 2:
            neg_sim = (z_anchor @ z_negatives.T) / temperature  # [B, N]
        else:
            neg_sim = torch.bmm(
                z_negatives, z_anchor.unsqueeze(-1)
            ).squeeze(-1) / temperature  # [B, N]

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [B, 1+N]
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def pool_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Mean-pool token-level latents into a single vector for contrastive loss."""
        return latents.mean(dim=1)

    # -- Analysis utilities ---------------------------------------------------

    @torch.no_grad()
    def basis_analysis(self) -> Dict[str, Any]:
        """
        Compare learned basis against reference DCT basis.

        Returns dict with:
          cosine_similarities: [K] cosine sim between learned and DCT per basis
          gram_matrix:         [K, K] current Gram matrix (ideal: identity)
          gram_offdiag_norm:   scalar, Frobenius norm of off-diagonal elements
          basis_norms:         [K] L2 norm of each basis vector
          energy_ordered:      whether norms are monotonically non-increasing
        """
        basis = self.basis.detach()
        ref = self._dct_basis_ref

        # Per-basis cosine similarity with DCT reference
        cos_sims = F.cosine_similarity(basis, ref, dim=-1)  # [K]

        # Gram matrix analysis
        gram = basis @ basis.T
        eye = torch.eye(self.num_basis, device=gram.device)
        offdiag = gram - torch.diag(gram.diag())
        gram_offdiag_norm = offdiag.norm().item()

        # Basis vector norms
        norms = basis.norm(dim=-1)  # [K]

        # Energy ordering check (important for nested dropout)
        energy_ordered = all(
            norms[i].item() >= norms[i + 1].item() - 1e-6
            for i in range(len(norms) - 1)
        )

        return {
            "cosine_similarities": cos_sims.cpu(),
            "gram_matrix": gram.cpu(),
            "gram_offdiag_norm": gram_offdiag_norm,
            "basis_norms": norms.cpu(),
            "energy_ordered": energy_ordered,
        }