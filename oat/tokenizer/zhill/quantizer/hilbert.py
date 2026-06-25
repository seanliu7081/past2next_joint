import math
from typing import Tuple

import torch


def xy2d(n: int, x: int, y: int) -> int:
    """Convert 2D coordinates (x, y) on an n x n grid to Hilbert curve index.

    Args:
        n: Grid size, must be a power of 2.
        x: X coordinate in [0, n).
        y: Y coordinate in [0, n).

    Returns:
        Hilbert curve index in [0, n*n).
    """
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def d2xy(n: int, d: int) -> Tuple[int, int]:
    """Convert Hilbert curve index d back to 2D coordinates (x, y) on an n x n grid.

    Args:
        n: Grid size, must be a power of 2.
        d: Hilbert curve index in [0, n*n).

    Returns:
        Tuple (x, y) of coordinates.
    """
    x = 0
    y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) > 0 else 0
        ry = 1 if ((d & 1) ^ rx) > 0 else 0  # ry = (d & 1) ^ rx
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def build_hilbert_lut(L0: int, L1: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build lookup tables for a rectangular L0 x L1 grid using Hilbert curve ordering.

    The Hilbert curve is computed on the smallest power-of-2 square that covers both
    dimensions, then filtered to keep only in-bounds points with consecutive re-indexing.

    Args:
        L0: Size of first dimension.
        L1: Size of second dimension.

    Returns:
        grid_to_hilbert: LongTensor of shape (L0, L1). grid_to_hilbert[x, y] = Hilbert index.
        hilbert_to_grid: LongTensor of shape (L0*L1, 2). hilbert_to_grid[idx] = (x, y).
    """
    n = 1 << math.ceil(math.log2(max(L0, L1))) if max(L0, L1) > 1 else 1

    # Walk the full Hilbert curve, keeping only in-bounds points
    hilbert_to_grid_list = []
    for d in range(n * n):
        x, y = d2xy(n, d)
        if x < L0 and y < L1:
            hilbert_to_grid_list.append((x, y))

    assert len(hilbert_to_grid_list) == L0 * L1, (
        f"Expected {L0 * L1} in-bounds points, got {len(hilbert_to_grid_list)}"
    )

    hilbert_to_grid = torch.tensor(hilbert_to_grid_list, dtype=torch.long)  # (L0*L1, 2)
    grid_to_hilbert = torch.zeros(L0, L1, dtype=torch.long)

    for idx, (x, y) in enumerate(hilbert_to_grid_list):
        grid_to_hilbert[x, y] = idx

    # Sanity check: roundtrip
    for idx in range(L0 * L1):
        x, y = hilbert_to_grid[idx].tolist()
        assert grid_to_hilbert[x, y].item() == idx, (
            f"Roundtrip failed at idx={idx}: grid_to_hilbert[{x},{y}]="
            f"{grid_to_hilbert[x, y].item()}"
        )

    return grid_to_hilbert, hilbert_to_grid
