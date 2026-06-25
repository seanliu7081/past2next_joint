import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.maskgit.maskgit import MaskGITModel


class MaskGITPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: OATTok,
        n_action_steps: int,
        n_obs_steps: int,
        # model architecture
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        # generation
        num_gen_steps: int = 8,
        temperature: float = 1.0,
        topk: int = 10,
        cfg_scale: float = 0.0,
        cfg_dropout_rate: float = 0.0,
    ):
        super().__init__()

        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta["obs"].items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)
            _type = attr["type"]
            if _type in modalities:
                obs_ports.append(key)

        # freeze action tokenizer
        for param in action_tokenizer.parameters():
            param.requires_grad_(False)
        action_tokenizer.eval()

        # create MaskGIT model
        codebook_size = action_tokenizer.quantizer.codebook_size
        latent_horizon = action_tokenizer.latent_horizon

        model = MaskGITModel(
            vocab_size=codebook_size,
            max_seq_len=latent_horizon,
            max_cond_len=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.action_tokenizer = action_tokenizer
        self.model = model
        self.max_seq_len = latent_horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim

        # generation params
        self.num_gen_steps = num_gen_steps
        self.temperature = temperature
        self.topk = topk
        self.cfg_scale = cfg_scale
        self.cfg_dropout_rate = cfg_dropout_rate

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(
            p.numel() for p in obs_encoder.parameters() if p.requires_grad
        )
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params
        num_tok_params = sum(p.numel() for p in action_tokenizer.parameters())
        num_trainable_tok_params = sum(
            p.numel() for p in action_tokenizer.parameters() if p.requires_grad
        )
        tok_trainable_ratio = num_trainable_tok_params / num_tok_params
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        model_trainable_ratio = num_trainable_model_params / num_model_params
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params / 1e6:.1f}M "
            f"({obs_trainable_ratio:.5%} trainable)\n"
            f"  act tok: {num_tok_params / 1e6:.1f}M "
            f"({tok_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params / 1e6:.1f}M "
            f"({model_trainable_ratio:.5%} trainable)\n"
        )

    # ── BasePolicy interface ────────────────────────────────────────────────

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities

    def get_observation_ports(self):
        return self.obs_ports

    def get_policy_name(self):
        base_name = "maskgitpolicy_"
        for modality in self.modalities:
            if modality != "state":
                base_name += modality + "|"
        return base_name[:-1]

    def create_dummy_observation(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device,
        )

    def set_normalizer(self, normalizer):
        self.obs_encoder.set_normalizer(normalizer)

    def get_optimizer(
        self,
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """AdamW with separate LR for encoder vs policy, weight-decay on ≥2-D params."""
        encoder_decay, encoder_nodecay = [], []
        for name, param in self.obs_encoder.named_parameters():
            if not param.requires_grad:
                continue
            (encoder_decay if param.dim() >= 2 else encoder_nodecay).append(param)

        policy_decay, policy_nodecay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (policy_decay if param.dim() >= 2 else policy_nodecay).append(param)

        optim_groups = [
            {"params": policy_decay,    "lr": policy_lr,  "weight_decay": weight_decay},
            {"params": policy_nodecay,  "lr": policy_lr,  "weight_decay": 0.0},
            {"params": encoder_decay,   "lr": obs_enc_lr, "weight_decay": weight_decay},
            {"params": encoder_nodecay, "lr": obs_enc_lr, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, betas=betas)

    # ── Training ────────────────────────────────────────────────────────────

    def forward(self, batch) -> torch.Tensor:
        """
        Masked-generative training step.

        1. Tokenize ground-truth actions  →  token_ids  (B, L)
        2. Sample a cosine mask ratio per sample, create random mask
        3. Forward through MaskGIT with bidirectional attention
        4. Cross-entropy loss on masked positions only
        """
        # tokenize actions (frozen tokenizer)
        with torch.no_grad():
            action_tokens = self.action_tokenizer.tokenize(batch["action"])  # (B, L)

        B, T = action_tokens.shape
        device = action_tokens.device

        # encode observations
        features = self.obs_encoder(batch["obs"])   # (B, To, d)

        # ── classifier-free guidance dropout ──
        if self.cfg_dropout_rate > 0.0 and self.training:
            drop = torch.rand(B, device=device) < self.cfg_dropout_rate
            features = features * (~drop).float().view(B, 1, 1)

        # ── random masking with cosine schedule ──
        # r ~ U(0,1),  mask_ratio = cos(r * π/2)  →  biased toward high mask ratio
        r = torch.rand(B, device=device)
        mask_ratios = torch.cos(r * (math.pi / 2))
        num_to_mask = (mask_ratios * T).ceil().clamp(min=1, max=T).long()  # (B,)

        # vectorised random mask: rank random noise, mask the lowest-rank positions
        noise = torch.rand(B, T, device=device)
        ranks = noise.argsort(dim=-1).argsort(dim=-1)          # rank per position
        mask = ranks < num_to_mask.unsqueeze(-1)                # (B, T)  True = masked

        # ── forward & loss ──
        logits = self.model(action_tokens, features, mask)      # (B, T, V)
        loss = F.cross_entropy(
            logits[mask],               # (num_masked, V)
            action_tokens[mask],        # (num_masked,)
        )
        return loss

    # ── Inference ───────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        num_gen_steps: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        cfg_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if num_gen_steps is None:
            num_gen_steps = self.num_gen_steps
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        # encode observation
        features = self.obs_encoder(obs_dict)       # (B, To, d)
        B = features.shape[0]

        # prepare unconditional features for CFG
        uncond_cond = None
        if cfg_scale > 0.0:
            uncond_cond = torch.zeros_like(features)

        # iterative parallel decoding
        action_tokens = self.model.generate(
            cond=features,
            seq_len=self.max_seq_len,
            num_steps=num_gen_steps,
            temperature=temperature,
            top_k=topk,
            cfg_scale=cfg_scale,
            uncond_cond=uncond_cond,
        )                                           # (B, L)

        # decode tokens → continuous actions
        action_pred = self.action_tokenizer.detokenize(
            tokens=action_tokens,
        )

        # receding horizon
        action = action_pred[:, : self.n_action_steps]

        return {
            "action": action,
            "action_pred": action_pred,
        }