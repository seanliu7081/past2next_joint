"""PolarOATPolicy: Multi-head factored AR policy for PolarOATTok.

Predicts factored tokens (inv, theta_trans, theta_rot, yaw) in parallel
from independent classification heads sharing a single transformer backbone.
Vanilla version — observation-conditioned only, no past action conditioning.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.autoregressive.polar_transformer_cache import PolarAutoRegressiveModel


class PolarOATPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer,  # PolarOATTok (duck-typed)
        n_action_steps: int,
        n_obs_steps: int,
        # policy model params
        embed_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        # policy inference params
        temperature: float = 1.0,
        topk: int = 10,
    ):
        super().__init__()

        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta['obs'].items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr['type']
            if type in modalities:
                obs_ports.append(key)

        # Freeze action tokenizer
        for param in action_tokenizer.parameters():
            param.requires_grad_(False)
        action_tokenizer.eval()

        # Get factored vocab sizes from tokenizer
        vocab_sizes = action_tokenizer.vocab_sizes  # dict
        latent_horizon = action_tokenizer.latent_horizon

        # Create multi-head AR model
        model = PolarAutoRegressiveModel(
            vocab_sizes=vocab_sizes,
            max_seq_len=latent_horizon + 1,  # +1 for BOS
            max_cond_len=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
        )

        # BOS ids per subspace (= vocab_size for that subspace)
        bos_ids = {name: vs for name, vs in vocab_sizes.items()}

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.action_tokenizer = action_tokenizer
        self.model = model
        self.vocab_sizes = vocab_sizes
        self.subspace_names = list(vocab_sizes.keys())
        self.max_seq_len = latent_horizon
        self.bos_ids = bos_ids
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.topk = topk

        # Report parameter counts
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params if num_obs_params > 0 else 0
        num_tok_params = sum(p.numel() for p in action_tokenizer.parameters())
        num_trainable_tok_params = sum(p.numel() for p in action_tokenizer.parameters() if p.requires_grad)
        tok_trainable_ratio = num_trainable_tok_params / num_tok_params if num_tok_params > 0 else 0
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / num_model_params if num_model_params > 0 else 0
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params / 1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  act tok: {num_tok_params / 1e6:.1f}M ({tok_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params / 1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
            f"  vocab_sizes: {vocab_sizes}\n"
            f"  latent_horizon: {latent_horizon}\n"
        )

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities

    def get_observation_ports(self):
        return self.obs_ports

    def get_policy_name(self):
        base_name = 'polar_oatpolicy_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
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
        encoder_decay_params = []
        encoder_nodecay_params = []
        for name, param in self.obs_encoder.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                encoder_decay_params.append(param)
            else:
                encoder_nodecay_params.append(param)

        policy_decay_params = []
        policy_nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                policy_decay_params.append(param)
            else:
                policy_nodecay_params.append(param)

        optim_groups = [
            {'params': policy_decay_params, 'lr': policy_lr, 'weight_decay': weight_decay},
            {'params': policy_nodecay_params, 'lr': policy_lr, 'weight_decay': 0.0},
            {'params': encoder_decay_params, 'lr': obs_enc_lr, 'weight_decay': weight_decay},
            {'params': encoder_nodecay_params, 'lr': obs_enc_lr, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, betas=betas)
        return optimizer

    def forward(self, batch) -> torch.Tensor:
        """Training forward pass. Returns sum of per-head CE losses."""
        # Tokenize actions (frozen tokenizer) -> dict of factored indices
        with torch.no_grad():
            token_dict = self.action_tokenizer.tokenize(batch['action'])
            # {name: (B, T_latent)} for each subspace

        B = batch['action'].shape[0]
        device = batch['action'].device

        # Encode observations
        features = self.obs_encoder(batch['obs'])  # (B, To, d)

        # Prepend BOS token to each subspace, shift right for teacher forcing
        input_dict = {}
        target_dict = {}
        for name in self.subspace_names:
            bos = torch.full((B, 1), self.bos_ids[name], dtype=torch.long, device=device)
            # Input: [BOS, tok_0, tok_1, ..., tok_{T-2}]  (drop last)
            input_dict[name] = torch.cat([bos, token_dict[name]], dim=1)[:, :-1]
            # Target: [tok_0, tok_1, ..., tok_{T-1}]
            target_dict[name] = token_dict[name]

        # Forward model -> dict of logits per head
        logits_dict = self.model(input_dict, cond=features)

        # Sum of CE losses across all heads
        total_loss = torch.tensor(0.0, device=device)
        for name in self.subspace_names:
            logits = logits_dict[name]       # (B, T, vocab+1)
            targets = target_dict[name]      # (B, T)
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )
            total_loss = total_loss + loss

        return total_loss

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        else:
            use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        # Encode observations
        features = self.obs_encoder(obs_dict)  # (B, To, d)
        B = features.shape[0]

        # Create BOS prefix for each subspace
        prefix_dict = {
            name: torch.full((B, 1), self.bos_ids[name], dtype=torch.long, device=self.device)
            for name in self.subspace_names
        }

        # Autoregressive generation
        generated_dict = self.model.generate(
            prefix_dict,
            cond=features,
            max_new_tokens=use_k_tokens,
            temperature=temperature,
            top_k=topk,
        )

        # Strip BOS and clamp to valid range
        token_dict = {}
        for name in self.subspace_names:
            tokens = generated_dict[name][:, 1:]  # drop BOS
            tokens = tokens.clamp(0, self.bos_ids[name] - 1)
            token_dict[name] = tokens

        # Detokenize
        with torch.inference_mode():
            action_pred = self.action_tokenizer.detokenize(token_dict)

        # Receding horizon
        action = action_pred[:, :self.n_action_steps]

        return {
            'action': action,
            'action_pred': action_pred,
        }
