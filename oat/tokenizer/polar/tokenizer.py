import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.base_tokenizer import BaseTokenizer
from oat.tokenizer.oat.quantizer.fsq import FSQ
from oat.tokenizer.polar.polar_decompose import PolarDecompose
from oat.tokenizer.polar.cyclic_vq import CyclicVQ


class InvEncoder(nn.Module):
    """MLP encoder for the 4D invariant subspace -> latent for FSQ.

    Any function of SO(2)-invariant inputs is still SO(2)-invariant,
    so this MLP cannot break invariance.

    Args:
        input_dim: Invariant subspace dimension (4).
        hidden_dim: Hidden layer width.
        num_layers: Number of hidden layers.
        output_dim: FSQ latent dimension (= len(fsq_levels)).
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128,
                 num_layers: int = 2, output_dim: int = 4):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU()])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolarDecoder(nn.Module):
    """Decoder: maps quantized representations -> 7D Cartesian action.

    Outputs Cartesian directly (NOT polar). This is the intentional relaxation
    allowing the decoder to learn workspace boundary corrections, gravity-induced
    asymmetries, and task-specific biases.

    During training, the invariant path uses a linear projection from the FSQ
    quantized latent (continuous, with STE gradient) so that encoder gradients
    flow through. The equivariant path uses embedding lookups (CyclicVQ has no
    learnable encoder, so no gradient issue).

    During inference (detokenize), all paths use embedding lookups from indices.

    Args:
        fsq_dim: Dimensionality of FSQ latent (= len(fsq_levels)).
        vocab_inv: Invariant FSQ codebook size (for inference embedding table).
        n_bins_trans: Number of theta_trans bins (+ 1 NULL).
        n_bins_rot: Number of theta_rot bins (+ 1 NULL).
        n_bins_yaw: Number of dyaw bins.
        embed_dim: Per-token embedding dimension.
        hidden_dim: Decoder MLP hidden width.
        num_layers: Number of decoder MLP layers.
        output_dim: Output dimension (7 for Cartesian actions).
    """

    def __init__(
        self,
        fsq_dim: int,
        vocab_inv: int,
        n_bins_trans: int,
        n_bins_rot: int,
        n_bins_yaw: int,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 7,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fsq_dim = fsq_dim

        # Invariant: linear projection from continuous FSQ latent.
        # Used in both training (with STE gradient from encoder) and inference
        # (from FSQ.indices_to_embedding). Single path avoids unused-parameter
        # issues with DDP.
        self.inv_proj = nn.Linear(fsq_dim, embed_dim)

        # Equivariant: embedding tables (CyclicVQ has no learnable encoder)
        self.theta_trans_embedding = nn.Embedding(n_bins_trans + 1, embed_dim)  # +1 NULL
        self.theta_rot_embedding = nn.Embedding(n_bins_rot + 1, embed_dim)     # +1 NULL
        self.yaw_embedding = nn.Embedding(n_bins_yaw, embed_dim)

        total_embed = embed_dim * 4  # inv + 3 angular embeddings
        layers = []
        in_d = total_embed
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU()])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        token_indices: Dict[str, torch.Tensor],
        inv_quant: torch.Tensor,
    ) -> torch.Tensor:
        """Decode to 7D Cartesian actions.

        Args:
            token_indices: Dict with keys 'theta_trans', 'theta_rot', 'yaw'.
            inv_quant: (*, fsq_dim) continuous FSQ quantized codes.
                       During training this carries STE gradients from the encoder.
                       During inference, obtain from FSQ.indices_to_embedding().

        Returns:
            actions: (*, 7) reconstructed Cartesian actions.
        """
        e_inv = self.inv_proj(inv_quant)

        e_tt = self.theta_trans_embedding(token_indices['theta_trans'])
        e_tr = self.theta_rot_embedding(token_indices['theta_rot'])
        e_yaw = self.yaw_embedding(token_indices['yaw'])

        cat = torch.cat([e_inv, e_tt, e_tr, e_yaw], dim=-1)
        return self.mlp(cat)


class PolarOATTok(BaseTokenizer):
    """Geometry-aware action tokenizer using polar decomposition.

    Decomposes 7D robot actions into:
      - Invariant subspace: (r_trans, dz, r_rot, grip) -> MLP -> FSQ (ordinal)
      - Equivariant subspace: (theta_trans, theta_rot, dyaw) -> CyclicVQ (geodesic, no encoder)

    The equivariant path has NO learnable encoder — angles go directly to CyclicVQ.
    The decoder outputs Cartesian 7D directly (intentional relaxation).
    Near-zero motion (r ~ 0) uses special NULL tokens for ill-defined angles.

    Args:
        fsq_levels: FSQ level list for invariant subspace, e.g. [8, 10, 4, 3].
        n_bins_trans: Number of uniform bins for theta_trans.
        n_bins_rot: Number of uniform bins for theta_rot.
        n_bins_yaw: Number of uniform bins for dyaw.
        inv_encoder_hidden: Hidden dim for invariant encoder MLP.
        inv_encoder_layers: Number of hidden layers in invariant encoder.
        decoder_hidden: Hidden dim for decoder MLP.
        decoder_layers: Number of hidden layers in decoder.
        embed_dim: Per-token embedding dimension in decoder.
        sample_horizon: Action chunk length (T). None for per-step mode.
    """

    def __init__(
        self,
        fsq_levels: List[int] = [8, 10, 4, 3],
        n_bins_trans: int = 24,
        n_bins_rot: int = 12,
        n_bins_yaw: int = 16,
        inv_encoder_hidden: int = 128,
        inv_encoder_layers: int = 2,
        decoder_hidden: int = 256,
        decoder_layers: int = 2,
        embed_dim: int = 32,
        sample_horizon: int = 32,
    ):
        super().__init__()

        self.sample_horizon = sample_horizon

        # --- Deterministic polar decomposition (no params) ---
        self.polar_decompose = PolarDecompose()

        # --- Invariant path: MLP encoder -> FSQ ---
        self.inv_encoder = InvEncoder(
            input_dim=4,
            hidden_dim=inv_encoder_hidden,
            num_layers=inv_encoder_layers,
            output_dim=len(fsq_levels),
        )
        self.quantizer = FSQ(levels=fsq_levels)

        # --- Equivariant path: CyclicVQ (no encoder, fixed bins) ---
        self.cyclic_vq = CyclicVQ(n_bins=[n_bins_trans, n_bins_rot, n_bins_yaw])

        # --- Decoder: token embeddings -> 7D Cartesian ---
        self.decoder = PolarDecoder(
            fsq_dim=len(fsq_levels),
            vocab_inv=self.quantizer.codebook_size,
            n_bins_trans=n_bins_trans,
            n_bins_rot=n_bins_rot,
            n_bins_yaw=n_bins_yaw,
            embed_dim=embed_dim,
            hidden_dim=decoder_hidden,
            num_layers=decoder_layers,
            output_dim=7,
        )

        # --- Normalizer ---
        self.normalizer = LinearNormalizer()

        # Store config for vocab_sizes
        self._n_bins_trans = n_bins_trans
        self._n_bins_rot = n_bins_rot
        self._n_bins_yaw = n_bins_yaw

    @property
    def vocab_sizes(self) -> Dict[str, int]:
        """Vocab size for each factored head (used by the AR policy)."""
        return {
            'inv': self.quantizer.codebook_size,
            'theta_trans': self._n_bins_trans + 1,  # +1 for NULL
            'theta_rot': self._n_bins_rot + 1,      # +1 for NULL
            'yaw': self._n_bins_yaw,
        }

    @property
    def effective_vocab_size(self) -> int:
        """Product of all vocab sizes — total expressiveness."""
        return math.prod(self.vocab_sizes.values())

    @property
    def codebook_size(self) -> int:
        """Effective vocab for compatibility with policy code."""
        return self.effective_vocab_size

    @property
    def latent_horizon(self) -> int:
        """Number of token positions. Each timestep produces one set of factored tokens."""
        return self.sample_horizon

    def get_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _encode_step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Encode per-timestep actions into factored tokens.

        Args:
            actions: (*, 7) normalized actions.

        Returns:
            latents: dict of quantized latent values.
            tokens: dict of token indices.
        """
        # Polar decomposition
        invariant, equivariant, null_mask = self.polar_decompose(actions)

        # Invariant path: MLP -> FSQ
        inv_latent = self.inv_encoder(invariant)        # (*, fsq_dim)
        inv_quant, inv_tokens = self.quantizer.forward_z(inv_latent)  # (*, fsq_dim), (*, )

        # Equivariant path: direct CyclicVQ (no encoder)
        eq_quant, eq_tokens = self.cyclic_vq(equivariant, null_mask=null_mask)

        latents = {
            'inv': inv_quant,       # (*, fsq_dim)
            'eq': eq_quant,         # (*, 3)
        }
        tokens = {
            'inv': inv_tokens,              # (*, )
            'theta_trans': eq_tokens[..., 0],  # (*, )
            'theta_rot': eq_tokens[..., 1],    # (*, )
            'yaw': eq_tokens[..., 2],          # (*, )
        }
        return latents, tokens

    def _decode_tokens(
        self,
        tokens: Dict[str, torch.Tensor],
        inv_quant: torch.Tensor,
    ) -> torch.Tensor:
        """Decode factored token indices to 7D Cartesian.

        Args:
            tokens: dict of token indices.
            inv_quant: (*, fsq_dim) continuous FSQ quantized codes.

        Returns:
            actions: (*, 7) reconstructed normalized actions.
        """
        return self.decoder(tokens, inv_quant=inv_quant)

    def encode(self, samples: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Encode action samples into factored tokens.

        Args:
            samples: (B, T, 7) raw actions.

        Returns:
            latents: dict of quantized latent tensors.
            tokens: dict of token index tensors, each (B, T).
        """
        nsamples = self.normalizer['action'].normalize(samples)
        return self._encode_step(nsamples)

    def decode(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode factored token indices to raw action space.

        Args:
            tokens: dict of token indices, each (B, T).

        Returns:
            samples: (B, T, 7) reconstructed actions in original scale.
        """
        # Recover continuous FSQ codes from discrete indices
        inv_quant = self.quantizer.indices_to_embedding(tokens['inv'])
        nsamples = self._decode_tokens(tokens, inv_quant=inv_quant)
        return self.normalizer['action'].unnormalize(nsamples)

    def forward(self, batch: dict) -> torch.Tensor:
        """Training forward pass. Returns scalar MSE loss.

        Args:
            batch: dict with 'action' key, shape (B, T, 7).

        Returns:
            loss: scalar reconstruction MSE in normalized space.
        """
        samples = batch['action']
        nsamples = self.normalizer['action'].normalize(samples)

        latents, tokens = self._encode_step(nsamples)
        # Pass inv_quant (continuous, STE gradient) so encoder gets gradients
        recons = self._decode_tokens(tokens, inv_quant=latents['inv'])

        loss = F.mse_loss(recons, nsamples)
        return loss

    def autoencode(self, samples: torch.Tensor) -> torch.Tensor:
        """Full encode-decode round trip.

        Args:
            samples: (B, T, 7) raw actions.

        Returns:
            recons: (B, T, 7) reconstructed actions in original scale.
        """
        latents, tokens = self.encode(samples)
        # Use continuous latents from encode (not re-derived from indices)
        inv_quant = latents['inv']
        nsamples = self._decode_tokens(tokens, inv_quant=inv_quant)
        return self.normalizer['action'].unnormalize(nsamples)

    def tokenize(self, samples: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Tokenize actions into factored token indices.

        Args:
            samples: (B, T, 7) raw actions.

        Returns:
            tokens: dict of token indices, each (B, T).
        """
        _, tokens = self.encode(samples)
        return tokens

    def detokenize(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct actions from factored token indices.

        Args:
            tokens: dict of token indices, each (B, T).

        Returns:
            samples: (B, T, 7) reconstructed actions in original scale.
        """
        # Recover continuous FSQ codes from discrete indices
        inv_quant = self.quantizer.indices_to_embedding(tokens['inv'])
        nsamples = self._decode_tokens(tokens, inv_quant=inv_quant)
        return self.normalizer['action'].unnormalize(nsamples)
