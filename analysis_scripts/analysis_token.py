import torch
import dill
import zarr
import numpy as np

# ── 加载 BSQ tokenizer ──
payload = torch.load("/workspace/oat/tok_ckpt/bsq/ep-0840_mse-0.004.ckpt",
                     pickle_module=dill, map_location='cpu')

from oat.tokenizer.oat.tokenizer import OATTok
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.bsq import BSQ

encoder = RegisterEncoder(
    sample_dim=7, sample_horizon=32,
    emb_dim=256, head_dim=64, depth=2, pdropout=0.1,
    latent_dim=10, num_registers=8,
)
decoder = SinglePassDecoder(
    sample_dim=7, sample_horizon=32,
    emb_dim=256, head_dim=64, depth=4, pdropout=0.1,
    token_dropout_mode='pow2', latent_dim=10, latent_horizon=8,
    use_causal_decoder=True,
)
quantizer = BSQ(L=10)
tok_bsq = OATTok(encoder=encoder, decoder=decoder, quantizer=quantizer)
tok_bsq.load_state_dict(payload['state_dicts']['ema_model'])
tok_bsq.eval().cuda()

# ── 加载 FSQ tokenizer（同样方式） ──
payload_fsq = torch.load("/workspace/oat/ep-0700_mse-0.003.ckpt",
                         pickle_module=dill, map_location='cpu')

from oat.tokenizer.oat.quantizer.fsq import FSQ

encoder_fsq = RegisterEncoder(
    sample_dim=7, sample_horizon=32,
    emb_dim=256, head_dim=64, depth=2, pdropout=0.1,
    latent_dim=4, num_registers=8,
)
decoder_fsq = SinglePassDecoder(
    sample_dim=7, sample_horizon=32,
    emb_dim=256, head_dim=64, depth=4, pdropout=0.1,
    token_dropout_mode='pow2', latent_dim=4, latent_horizon=8,
    use_causal_decoder=True,
)
quantizer_fsq = FSQ(levels=[8, 5, 5, 5])
tok_fsq = OATTok(encoder=encoder_fsq, decoder=decoder_fsq, quantizer=quantizer_fsq)
tok_fsq.load_state_dict(payload_fsq['state_dicts']['ema_model'])
tok_fsq.eval().cuda()

# ── 加载数据 ──
root = zarr.open("/workspace/oat/data/libero/libero10_N500.zarr", "r")
actions_flat = torch.tensor(root['data']['action'][:], dtype=torch.float32)
episode_ends = root['meta']['episode_ends'][:]

chunks = []
prev = 0
for end in episode_ends:
    ep = actions_flat[prev:int(end)]
    for s in range(0, len(ep) - 31, 16):
        chunks.append(ep[s:s + 32])
    prev = int(end)
chunks = torch.stack(chunks).cuda()
print(f"Total chunks: {chunks.shape[0]}")

# ── 分析函数 ──
def analyze_tokens(tok, chunks, name, codebook_size):
    with torch.no_grad():
        tokens = tok.tokenize(chunks)
    print(f"\n{'='*60}")
    print(f"  {name}  (codebook={codebook_size})")
    print(f"{'='*60}")
    for k in range(8):
        t = tokens[:, k]
        n_unique = t.unique().shape[0]
        counts = torch.bincount(t, minlength=codebook_size)
        probs = counts.float() / counts.sum()
        entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
        max_entropy = torch.tensor(float(codebook_size)).log()
        top10 = counts.sort(descending=True).values[:10].sum().float() / counts.sum()
        top50 = counts.sort(descending=True).values[:50].sum().float() / counts.sum()
        print(f"  Token {k}: unique={n_unique:>5}/{codebook_size}, "
              f"H_ratio={entropy/max_entropy:.3f}, "
              f"top10={top10:.1%}, top50={top50:.1%}")

# ── 跑分析 ──
analyze_tokens(tok_bsq, chunks, "BSQ (L=10)", 1024)
analyze_tokens(tok_fsq, chunks, "FSQ [8,5,5,5]", 1000)