import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from oat.policy.oatpolicy import OATPolicy
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.autoregressive.transformer_cache import AutoregressiveModel


class OATPolicyJoint(OATPolicy):
    """End-to-end co-training of the action tokenizer and the AR policy backbone.

    Unlike `OATPolicy`, which loads a *frozen* tokenizer from a checkpoint, this
    policy owns a *trainable* `OATTok` built from scratch and optimizes it jointly
    with the backbone + obs encoder.  The combined loss is

        loss = recon_loss_weight * mse(decode(quant(encode(a))), a)      # trains tokenizer
             + current_ce_weight  * CE(backbone(tokens), tokens)         # trains backbone/obs

    The cross-entropy targets are the *detached* discrete FSQ token indices, so the
    CE loss never backprops into the tokenizer (FSQ indices are non-differentiable).
    The tokenizer learns solely from the reconstruction loss (STE through `quant`).

    `current_ce_weight` is a runtime-toggleable copy of `ce_loss_weight` used by the
    workspace to implement a tokenizer-only warmup phase (set it to 0.0 during warmup).
    """

    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: OATTok,
        n_action_steps: int,
        n_obs_steps: int,
        # policy model params
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        # policy inference params
        temperature: float = 1.0,
        topk: int = 10,
        # joint training params
        recon_loss_weight: float = 1.0,
        ce_loss_weight: float = 1.0,
    ):
        # Skip OATPolicy.__init__ (it freezes the tokenizer); init the nn.Module base.
        super(OATPolicy, self).__init__()

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

        # NOTE: tokenizer is intentionally NOT frozen here -- it is trained jointly.

        # create AR model
        codebook_size = action_tokenizer.quantizer.codebook_size
        latent_horizon = action_tokenizer.latent_horizon
        model = AutoregressiveModel(
            vocab_size=codebook_size + 1,  # +1 for <BOS>
            max_seq_len=latent_horizon + 1,
            max_cond_len=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
        )
        bos_id = codebook_size  # last token id for <BOS>

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.action_tokenizer = action_tokenizer
        self.model = model
        self.max_seq_len = latent_horizon
        self.bos_id = bos_id
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.topk = topk

        # joint training weights
        self.recon_loss_weight = recon_loss_weight
        self.ce_loss_weight = ce_loss_weight
        # runtime-toggleable CE weight (set to 0.0 during tokenizer-only warmup)
        self.current_ce_weight = ce_loss_weight

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params
        num_tok_params = sum(p.numel() for p in action_tokenizer.parameters())
        num_trainable_tok_params = sum(p.numel() for p in action_tokenizer.parameters() if p.requires_grad)
        tok_trainable_ratio = num_trainable_tok_params / num_tok_params
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / num_model_params
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params/1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  act tok: {num_tok_params/1e6:.1f}M ({tok_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params/1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
        )

    def get_policy_name(self):
        base_name = 'oat_joint_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]

    def set_normalizer(self, normalizer):
        # Both obs encoder and (trainable) tokenizer share the dataset normalizer so
        # the recon path (normalize) and detokenize path (unnormalize) stay consistent.
        self.obs_encoder.set_normalizer(normalizer)
        self.action_tokenizer.set_normalizer(normalizer)

    def set_ce_weight(self, weight: float):
        """Runtime toggle for the warmup phase (0.0 = tokenizer-only)."""
        self.current_ce_weight = weight

    def get_optimizer(
        self,
        policy_lr: float,
        obs_enc_lr: float,
        tokenizer_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """AdamW with weight decay for 2D params only, across backbone / obs enc / tokenizer."""
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

        tokenizer_decay_params = []
        tokenizer_nodecay_params = []
        for name, param in self.action_tokenizer.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                tokenizer_decay_params.append(param)
            else:
                tokenizer_nodecay_params.append(param)

        optim_groups = [
            {'params': policy_decay_params, 'lr': policy_lr, 'weight_decay': weight_decay},
            {'params': policy_nodecay_params, 'lr': policy_lr, 'weight_decay': 0.0},
            {'params': encoder_decay_params, 'lr': obs_enc_lr, 'weight_decay': weight_decay},
            {'params': encoder_nodecay_params, 'lr': obs_enc_lr, 'weight_decay': 0.0},
            {'params': tokenizer_decay_params, 'lr': tokenizer_lr, 'weight_decay': weight_decay},
            {'params': tokenizer_nodecay_params, 'lr': tokenizer_lr, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, betas=betas)
        return optimizer

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        action = batch['action']
        B = action.shape[0]
        device = action.device

        tok = self.action_tokenizer

        # --- tokenizer reconstruction path (needs grad: NO inference_mode) ---
        nsamples = tok.normalizer['action'].normalize(action)
        latents = tok.encoder(nsamples)
        quant, tokens = tok.quantizer(latents)   # quant: STE-differentiable, tokens: long indices
        recons = tok.decoder(quant)              # nested_dropout active in train mode (desired)
        recon_loss = F.mse_loss(recons, nsamples)

        # --- CE targets: detached discrete tokens (no backprop into tokenizer) ---
        action_tokens = tokens.detach().long()   # [B, latent_horizon]

        # --- obs conditioning + AR backbone (always computed; see workspace DDP note) ---
        features = self.obs_encoder(batch['obs'])   # [B, To, d]

        # prepend <BOS> token
        action_tokens = torch.cat([
            torch.full(
                (B, 1), self.bos_id,
                dtype=torch.long, device=device
            ),
            action_tokens
        ], dim=1)                                   # [B, latent_horizon + 1]

        logits = self.model(action_tokens[:, :-1], cond=features)
        vocab_size = logits.size(-1)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),         # (B*T, vocab_size)
            action_tokens[:, 1:].reshape(-1)        # (B*T,)
        )

        total = self.recon_loss_weight * recon_loss + self.current_ce_weight * ce_loss
        return {
            'loss': total,
            'recon_loss': recon_loss,
            'ce_loss': ce_loss,
        }
