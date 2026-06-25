import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from oat.model.common.module_attr_mixin import ModuleAttrMixin


# ─── Building Blocks ─────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class MLP(nn.Module):
    def __init__(self, n_emb: int, p_drop: float):
        super().__init__()
        self.c_fc   = nn.Linear(n_emb, 4 * n_emb)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class SelfAttention(nn.Module):
    """Bidirectional (full) self-attention."""
    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        assert n_emb % n_head == 0
        self.c_attn  = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.c_proj  = nn.Linear(n_emb, n_emb, bias=False)
        self.resid_dropout = nn.Dropout(p_drop_attn)
        self.p_attn_dropout = p_drop_attn
        self.n_head  = n_head
        self.head_dim = n_emb // n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, T, self.n_head, self.head_dim).transpose(1, 2),
            (q, k, v),
        )
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.p_attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class CrossAttention(nn.Module):
    """Cross-attention to observation memory."""
    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        assert n_emb % n_head == 0
        self.q_proj  = nn.Linear(n_emb, n_emb, bias=False)
        self.kv_proj = nn.Linear(n_emb, 2 * n_emb, bias=False)
        self.c_proj  = nn.Linear(n_emb, n_emb, bias=False)
        self.resid_dropout = nn.Dropout(p_drop_attn)
        self.p_attn_dropout = p_drop_attn
        self.n_head  = n_head
        self.n_emb   = n_emb
        self.head_dim = n_emb // n_head

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        B_m, T_m, _ = memory.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(memory).split(self.n_emb, dim=2)
        k, v = map(
            lambda t: t.view(B_m, T_m, self.n_head, self.head_dim).transpose(1, 2),
            (k, v),
        )
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.p_attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    """Transformer block: self-attn → cross-attn → MLP (all pre-norm)."""
    def __init__(self, n_head: int, n_emb: int, p_drop_attn: float):
        super().__init__()
        self.ln_1 = RMSNorm(n_emb)
        self.attn  = SelfAttention(n_head, n_emb, p_drop_attn)
        self.ln_2 = RMSNorm(n_emb)
        self.cross_attn = CrossAttention(n_head, n_emb, p_drop_attn)
        self.ln_3 = RMSNorm(n_emb)
        self.mlp  = MLP(n_emb, p_drop_attn)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), memory)
        x = x + self.mlp(self.ln_3(x))
        return x


# ─── MaskGIT Model ───────────────────────────────────────────────────────────

class MaskGITModel(ModuleAttrMixin):
    """
    Masked Generative Image Transformer (MaskGIT) adapted for action-token
    prediction.  Replaces the left-to-right autoregressive model with
    bidirectional attention and iterative parallel decoding.
    """

    def __init__(
        self,
        vocab_size: int,
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
        self.n_layer   = n_layer
        self.n_head    = n_head
        self.n_emb     = n_emb
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # ── embeddings ──
        self.tok_emb      = nn.Embedding(vocab_size, n_emb)
        self.mask_emb     = nn.Parameter(torch.zeros(n_emb))
        self.pos_emb      = nn.Parameter(torch.zeros(1, max_seq_len, n_emb))
        self.cond_emb     = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_len, n_emb))
        self.drop         = nn.Dropout(p_drop_emb)

        # ── condition encoder (matches AR model) ──
        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.Mish(),
            nn.Linear(4 * n_emb, n_emb),
        )

        # ── transformer blocks ──
        self.blocks = nn.ModuleList(
            [Block(n_head, n_emb, p_drop_attn) for _ in range(n_layer)]
        )

        # ── output head ──
        self.ln_f = RMSNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab_size, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    # ── initialisation ──────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, RMSNorm):
                nn.init.constant_(m.weight, 1.0)
        nn.init.normal_(self.mask_emb, std=0.02)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _encode_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Encode observation features into memory (computed once per sample)."""
        T_cond = cond.shape[1]
        cond_emb = self.cond_emb(cond)
        memory = self.drop(cond_emb + self.cond_pos_emb[:, :T_cond, :])
        return self.encoder(memory)

    def _embed_tokens(
        self, tokens: torch.LongTensor, mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Embed tokens, replacing masked positions with the learnable [MASK]."""
        tok_emb = self.tok_emb(tokens)                          # (B, T, n_emb)
        mask_exp = mask.unsqueeze(-1).expand_as(tok_emb)        # (B, T, n_emb)
        tok_emb = torch.where(mask_exp, self.mask_emb, tok_emb)
        T = tokens.shape[1]
        return self.drop(tok_emb + self.pos_emb[:, :T, :])

    @staticmethod
    def _unmask_schedule(seq_len: int, num_steps: int) -> List[int]:
        """
        Cosine schedule: returns a list of length *num_steps* where entry *i*
        is the number of tokens that should **remain masked** after step *i*.
        Guaranteed monotonically decreasing and ending at 0.
        """
        raw = []
        for s in range(1, num_steps + 1):
            ratio = math.cos(math.pi / 2 * s / num_steps)
            raw.append(max(0, round(seq_len * ratio)))
        raw[-1] = 0  # ensure all unmasked at final step

        # ensure monotonically decreasing & at least 1 token unmasked per step
        for i in range(len(raw)):
            prev = seq_len if i == 0 else raw[i - 1]
            raw[i] = min(raw[i], max(prev - 1, 0))
            raw[i] = max(raw[i], 0)
        return raw

    # ── forward (training) ──────────────────────────────────────────────────

    def forward(
        self,
        tokens: torch.LongTensor,
        cond: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            tokens: (B, T)  ground-truth token ids (valid at all positions)
            cond:   (B, T_cond, cond_dim)  observation features
            mask:   (B, T)  True at positions to be predicted

        Returns:
            logits: (B, T, vocab_size)
        """
        memory = self._encode_cond(cond)
        x = self._embed_tokens(tokens, mask)
        for block in self.blocks:
            x = block(x, memory)
        return self.head(self.ln_f(x))

    # ── generate (inference) ────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        cond: torch.Tensor,
        seq_len: int,
        num_steps: int = 8,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        cfg_scale: float = 0.0,
        uncond_cond: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        """
        Iterative parallel decoding (MaskGIT).

        Args:
            cond:       (B, T_cond, cond_dim)
            seq_len:    number of tokens to generate
            num_steps:  number of unmasking iterations
            temperature: sampling temperature
            top_k:      optional top-k filtering
            cfg_scale:  classifier-free guidance scale (0 = disabled)
            uncond_cond: unconditional observation features for CFG
                         (required when cfg_scale > 0)

        Returns:
            tokens: (B, seq_len)  generated token ids
        """
        B = cond.shape[0]
        device = cond.device

        # pre-compute observation memory (static across steps)
        memory = self._encode_cond(cond)
        if cfg_scale > 0.0:
            assert uncond_cond is not None, \
                "Must provide uncond_cond when cfg_scale > 0"
            uncond_memory = self._encode_cond(uncond_cond)

        schedule = self._unmask_schedule(seq_len, num_steps)

        tokens = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)

        for step, n_remain in enumerate(schedule):
            # forward with pre-computed memory
            x = self._embed_tokens(tokens, mask)
            for block in self.blocks:
                x = block(x, memory)
            logits = self.head(self.ln_f(x))            # (B, T, V)

            # classifier-free guidance
            if cfg_scale > 0.0:
                x_u = self._embed_tokens(tokens, mask)
                for block in self.blocks:
                    x_u = block(x_u, uncond_memory)
                logits_u = self.head(self.ln_f(x_u))
                logits = logits_u + (1 + cfg_scale) * (logits - logits_u)

            # ── sample ──
            scaled_logits = logits / temperature if temperature > 0 else logits
            if top_k is not None:
                v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)), dim=-1)
                scaled_logits[scaled_logits < v[..., [-1]]] = -float("Inf")

            if temperature > 0:
                probs = F.softmax(scaled_logits, dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, self.vocab_size), 1
                ).view(B, seq_len)
            else:
                sampled = scaled_logits.argmax(dim=-1)

            # confidence = probability of the sampled token (temp-free)
            probs_clean = F.softmax(logits, dim=-1)
            confidences = torch.gather(
                probs_clean, -1, sampled.unsqueeze(-1)
            ).squeeze(-1)

            # fill in already-unmasked positions
            sampled     = torch.where(mask, sampled, tokens)
            confidences = torch.where(
                mask, confidences,
                torch.tensor(float("inf"), device=device),
            )

            if n_remain == 0:
                # last step – unmask everything
                tokens = sampled
                mask.fill_(False)
                break

            # keep the *n_remain* lowest-confidence masked positions masked
            sorted_idx = confidences.argsort(dim=-1)          # ascending
            new_mask = torch.zeros_like(mask)
            new_mask.scatter_(1, sorted_idx[:, :n_remain], True)

            # update tokens at newly-unmasked positions
            tokens = torch.where(mask & ~new_mask, sampled, tokens)
            mask = new_mask

        return tokens