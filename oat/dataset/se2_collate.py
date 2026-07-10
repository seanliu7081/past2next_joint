"""Collate for paired-angle items from ``SE2AugZarrDataset(emit_angle_pair=True)``.

Phase-2 SCAFFOLD (consistency loss): not referenced by any training config.
Pass as ``collate_fn`` to the DataLoader; plain items (no ``'pair'`` key)
collate exactly like the default.
"""

from typing import Dict, List

import torch
from torch.utils.data import default_collate


def _collate_side(items: List[Dict]) -> Dict:
    """default_collate on {'obs', 'action'}; 'theta' floats -> (B,) float32."""
    out = default_collate([
        {k: v for k, v in it.items() if k not in ('theta', 'pair')}
        for it in items
    ])
    if 'theta' in items[0]:
        out['theta'] = torch.as_tensor(
            [it['theta'] for it in items], dtype=torch.float32)
    return out


def paired_angle_collate(batch: List[Dict]) -> Dict:
    out = _collate_side(batch)
    if 'pair' in batch[0]:
        out['pair'] = _collate_side([it['pair'] for it in batch])
    return out
