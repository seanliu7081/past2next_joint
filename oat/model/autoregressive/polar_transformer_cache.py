"""PolarAutoRegressiveModel: Multi-head factored AR model for PolarOATTok.

Instead of a single embedding + single classification head, uses:
- Summed per-subspace embeddings for input
- Parallel per-subspace classification heads for output
- No weight tying (incompatible with summed embeddings + multiple heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from oat.model.common.module_attr_mixin import ModuleAttrMixin
from oat.model.autoregressive.transformer_cache import (
    Block, RMSNorm, MLP,
)


class PolarAutoRegressiveModel(ModuleAttrMixin):
    """Autoregressive transformer with factored multi-head input/output.

    Each subspace (inv, theta_trans, theta_rot, yaw) has:
    - Its own embedding table (inputs are summed)
    - Its own classification head (outputs are independent)

    Args:
        vocab_sizes: Dict mapping subspace name -> vocab size (excluding BOS).
        max_seq_len: Maximum token sequence length (including BOS position).
        max_cond_len: Maximum conditioning sequence length.
        cond_dim: Conditioning feature dimension.
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads.
        n_emb: Embedding / hidden dimension.
        p_drop_emb: Embedding dropout probability.
        p_drop_attn: Attention dropout probability.
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        max_seq_len: int,
        max_cond_len: int,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_emb = n_emb
        self.subspace_names = list(vocab_sizes.keys())
        self.vocab_sizes = vocab_sizes

        # Per-subspace embedding tables (+1 for BOS token per subspace)
        self.tok_embs = nn.ModuleDict({
            name: nn.Embedding(vs + 1, n_emb)  # +1 for BOS
            for name, vs in vocab_sizes.items()
        })
        # BOS id per subspace = vocab_size (the extra index)
        self.bos_ids = {name: vs for name, vs in vocab_sizes.items()}

        # Positional embedding (shared across subspaces)
        self.tok_pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_emb))

        # Conditioning
        self.cond_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_len, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.Mish(),
            nn.Linear(4 * n_emb, n_emb),
        )

        # Transformer blocks (shared backbone)
        self.blocks = nn.ModuleList([
            Block(n_head, n_emb, p_drop_attn) for _ in range(n_layer)
        ])

        # Final normalization
        self.ln_f = RMSNorm(n_emb)

        # Per-subspace classification heads (NO weight tying)
        self.heads = nn.ModuleDict({
            name: nn.Linear(n_emb, vs + 1, bias=False)  # +1 for BOS (consistent dim)
            for name, vs in vocab_sizes.items()
        })

        self.init_weights_sp()

    def init_weights_sp(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, RMSNorm):
                nn.init.constant_(m.weight, 1.0)

    def embed_tokens(self, token_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum per-subspace embeddings into a single hidden state.

        Args:
            token_dict: {name: (B, T) LongTensor} for each subspace.

        Returns:
            (B, T, n_emb) summed embeddings.
        """
        embeddings = [self.tok_embs[name](token_dict[name]) for name in self.subspace_names]
        return sum(embeddings)

    def forward(
        self,
        token_dict: Dict[str, torch.Tensor],
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            token_dict: {name: (B, T) LongTensor} per subspace (with BOS prepended).
            cond: (B, T_cond, cond_dim) observation features.

        Returns:
            logits_dict: {name: (B, T, vocab_size+1)} logits per head.
        """
        T_tok = token_dict[self.subspace_names[0]].shape[1]
        T_cond = cond.shape[1]

        # Token embedding (sum of all subspace embeddings)
        tok_emb = self.embed_tokens(token_dict)
        pos_emb = self.tok_pos_emb[:, :T_tok, :]
        x = self.drop(tok_emb + pos_emb)

        # Condition embedding
        cond_emb = self.cond_emb(cond)
        cond_pos_emb = self.cond_pos_emb[:, :T_cond, :]
        memory = self.drop(cond_emb + cond_pos_emb)
        memory = self.encoder(memory)

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x, memory)

        x = self.ln_f(x)

        # Per-subspace logits
        logits_dict = {name: self.heads[name](x) for name in self.subspace_names}
        return logits_dict

    @torch.inference_mode()
    def generate(
        self,
        prefix_dict: Dict[str, torch.Tensor],
        cond: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive generation with KV caching.

        At each step, all subspace heads are sampled in parallel (C1: independent).

        Args:
            prefix_dict: {name: (B, T_pre) LongTensor} — typically just BOS tokens.
            cond: (B, T_cond, cond_dim) observation features.
            max_new_tokens: Number of new token positions to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering per head.

        Returns:
            out_dict: {name: (B, T_pre + max_new_tokens) LongTensor} generated tokens.
        """
        # --- Pre-compute condition KV cache ---
        T_cond = cond.shape[1]
        cond_emb = self.cond_emb(cond)
        cond_pos_emb = self.cond_pos_emb[:, :T_cond, :]
        memory = self.drop(cond_emb + cond_pos_emb)
        memory = self.encoder(memory)
        B_mem, T_mem = memory.shape[:2]

        memory_kv_cache = []
        for block in self.blocks:
            k_mem, v_mem = block.cross_attn.kv_proj(memory).split(self.n_emb, dim=2)
            k_mem, v_mem = map(
                lambda t: t.view(B_mem, T_mem, self.n_head, self.n_emb // self.n_head).transpose(1, 2),
                (k_mem, v_mem),
            )
            memory_kv_cache.append((k_mem, v_mem))

        # --- Phase 1: Process prefix to fill KV cache ---
        T_pre = prefix_dict[self.subspace_names[0]].shape[1]
        tok_emb = self.embed_tokens(prefix_dict)
        pos_emb = self.tok_pos_emb[:, :T_pre, :]
        x = self.drop(tok_emb + pos_emb)

        past_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.n_layer
        for i, block in enumerate(self.blocks):
            x, present = block(x, memory, layer_past=None, memory_kv_cache=memory_kv_cache[i])
            past_key_values[i] = present

        x = self.ln_f(x[:, -1:, :])
        logits_dict = {name: self.heads[name](x) for name in self.subspace_names}

        # --- Phase 2: Autoregressive generation ---
        out_dict = {name: prefix_dict[name].clone() for name in self.subspace_names}

        for i in range(max_new_tokens):
            # Sample from all heads simultaneously
            next_tokens = {}
            for name in self.subspace_names:
                head_logits = logits_dict[name].squeeze(1)  # (B, vocab+1)
                if temperature > 0:
                    head_logits = head_logits / temperature
                    if top_k is not None:
                        k = min(top_k, head_logits.size(-1))
                        v, _ = torch.topk(head_logits, k)
                        head_logits[head_logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(head_logits, dim=-1)
                    next_tokens[name] = torch.multinomial(probs, num_samples=1)  # (B, 1)
                else:
                    next_tokens[name] = torch.argmax(head_logits, dim=-1, keepdim=True)

            # Append to output
            for name in self.subspace_names:
                out_dict[name] = torch.cat((out_dict[name], next_tokens[name]), dim=1)

            # Check if done
            if out_dict[self.subspace_names[0]].shape[1] >= T_pre + max_new_tokens:
                break

            # Embed new tokens (sum) for next step
            x = sum(self.tok_embs[name](next_tokens[name]) for name in self.subspace_names)
            current_pos = T_pre + i
            pos_emb = self.tok_pos_emb[:, current_pos:current_pos + 1, :]
            x = self.drop(x + pos_emb)

            # Forward through blocks with KV cache
            for layer_idx, block in enumerate(self.blocks):
                x, present = block(
                    x, memory,
                    layer_past=past_key_values[layer_idx],
                    memory_kv_cache=memory_kv_cache[layer_idx],
                )
                past_key_values[layer_idx] = present

            x = self.ln_f(x)
            logits_dict = {name: self.heads[name](x) for name in self.subspace_names}

        return out_dict
