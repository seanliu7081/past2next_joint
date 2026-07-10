"""Single source of truth for the SO(2)/SE(2) block structure of the LIBERO
action and proprio spaces.

One ``BlockSpec`` list drives (i) group-compatible normalization
(``oat.equi.normalization``), (ii) the SE(2) label/proprio transforms applied by
the augmentation dataset (``oat.equi.se2_transforms``), and (iii) the
block-isotropic noise source (``oat.equi.sources``) so the three can never
disagree.

Action layout (``ACTION_DIM = 7``), per timestep:

    [dx, dy, dz, rx, ry, rz, grip]

with ``(rx, ry, rz)`` the axis-angle rotation delta of the OSC_POSE controller
(robosuite 1.4.0).

Representations under the world-frame yaw rotation R_z(theta) about the robot
base:

    rho1     -- 2-D vector pair, rotates by R(theta)
    rho0     -- scalar, invariant
    free_iso -- block of arbitrary dim with ONE tied isotropic scale. P1-safe
                under BOTH controller-frame hypotheses for the rotation delta
                (the "rot hedge"); this is the default for (rx, ry, rz).
    identity -- left untouched by normalization (unit-norm quaternions)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ACTION_DIM = 7

RHO0 = "rho0"
RHO1 = "rho1"
FREE_ISO = "free_iso"
IDENTITY = "identity"
_REPS = (RHO0, RHO1, FREE_ISO, IDENTITY)


@dataclass(frozen=True)
class BlockSpec:
    name: str
    idx: Tuple[int, ...]  # indices into the (last dim of the) vector this block owns
    rep: str              # one of RHO0 | RHO1 | FREE_ISO | IDENTITY

    def __post_init__(self):
        assert self.rep in _REPS, f"unknown rep '{self.rep}'"
        assert len(self.idx) > 0, f"block '{self.name}' is empty"
        if self.rep == RHO1:
            assert len(self.idx) == 2, f"rho1 block '{self.name}' must be a 2-dim pair"


def libero_action_blocks(world_frame_rotation: bool = False) -> List[BlockSpec]:
    """Block spec for the 7-DoF LIBERO action [dx, dy, dz, rx, ry, rz, grip].

    ``world_frame_rotation=False`` (DEFAULT, the hedge): the axis-angle rotation
    delta is one isotropic 3-dim block -- P1-correct whether the controller
    interprets it in the world frame or the EE frame. Only flip to ``True`` as
    an ablation after the controller-frame probe (M2) conclusively shows
    world-frame rotation deltas.
    """
    blocks = [
        BlockSpec("xy", (0, 1), RHO1),
        BlockSpec("z", (2,), RHO0),
    ]
    if world_frame_rotation:
        blocks += [
            BlockSpec("rot_xy", (3, 4), RHO1),
            BlockSpec("rot_z", (5,), RHO0),
        ]
    else:
        blocks += [BlockSpec("rot", (3, 4, 5), FREE_ISO)]
    blocks += [BlockSpec("grip", (6,), RHO0)]
    assert_blocks_cover(blocks, ACTION_DIM)
    return blocks


def libero_obs_blocks() -> Dict[str, List[BlockSpec]]:
    """Block specs for the low-dim LIBERO proprio observation keys.

    Note on ``robot0_eef_pos``: the world EEF position rotates about the robot
    base point ``p_base != 0``, so in normalized space the group action is
    orthogonal-linear PLUS a constant offset. The zero-centered shared xy scale
    here is the Phase-1 hedge; the augmentation labels themselves stay exact
    because the dataset rotates the RAW position about ``p_base``.

    ``robot0_eef_quat`` is a unit quaternion (xyzw): normalization is identity
    (scale 1, offset 0) -- any per-dim affine would break the quaternion group
    action.

    ``robot0_joint_pos`` (present in the zarr but unused by the libero10
    policy): joint 1 shifts by theta under the group action, which is not
    linearly equivariant -- kept per-dim rho0 as a documented residual.
    """
    return {
        "robot0_eef_pos": [
            BlockSpec("xy", (0, 1), RHO1),
            BlockSpec("z", (2,), RHO0),
        ],
        "robot0_eef_quat": [BlockSpec("quat", (0, 1, 2, 3), IDENTITY)],
        "robot0_gripper_qpos": [BlockSpec("g", (0, 1), RHO0)],
        "task_uid": [BlockSpec("uid", (0,), RHO0)],
        "robot0_joint_pos": [BlockSpec("joints", tuple(range(7)), RHO0)],
    }


def assert_blocks_cover(blocks: Sequence[BlockSpec], dim: int) -> None:
    """Assert blocks are non-overlapping and cover exactly ``range(dim)``."""
    covered = np.zeros(dim, dtype=bool)
    for b in blocks:
        idx = np.asarray(b.idx, dtype=int)
        assert idx.min() >= 0 and idx.max() < dim, (
            f"block '{b.name}' indices {b.idx} out of range for dim={dim}"
        )
        assert not covered[idx].any(), f"block '{b.name}' overlaps an earlier block"
        covered[idx] = True
    missing = np.nonzero(~covered)[0].tolist()
    assert not missing, f"blocks do not cover all {dim} dims; missing {missing}"


def action_rho_matrix(theta: float, rotate_rotation: bool) -> np.ndarray:
    """The exact group action rho_A(theta) on the 7-dim action, as a matrix.

    The action is fixed by the physics/controller, NOT by the block
    decomposition: (dx, dy) always rotates; the axis-angle delta (rx, ry, rz)
    transforms as a world vector -- i.e. (rx, ry) rotates -- iff the controller
    interprets it in the world frame (``rotate_rotation=True``, gated by the
    M2 controller-frame probe). dz, rz, grip are invariant.

    The block decompositions in :func:`libero_action_blocks` must be
    *compatible* with this matrix (block-diagonal w.r.t. the blocks, orthogonal
    on each rotating block) -- asserted in tests.
    """
    c, s = np.cos(theta), np.sin(theta)
    rho = np.eye(ACTION_DIM)
    rho[0, 0], rho[0, 1] = c, -s
    rho[1, 0], rho[1, 1] = s, c
    if rotate_rotation:
        rho[3, 3], rho[3, 4] = c, -s
        rho[4, 3], rho[4, 4] = s, c
    return rho


def to_noise_blocks(
    blocks: Sequence[BlockSpec],
    scales: Optional[Dict[str, float]] = None,
    ranges: Optional[Sequence[float]] = None,
):
    """Adapter to the existing diffusion-side ``equi_noise.NoiseBlock`` spec.

    Provided for interop/equivalence tests only -- the flow path does not
    depend on ``oat.model.diffusion.equi_noise``.
    """
    from oat.model.diffusion.equi_noise import NoiseBlock

    kind_of_rep = {RHO1: "so2_vec", RHO0: "scalar", FREE_ISO: "free", IDENTITY: "scalar"}
    s = scales or {}

    def rng(idx):
        if ranges is None:
            return None
        return [float(ranges[i]) for i in idx]

    return [
        NoiseBlock(
            b.name,
            list(b.idx),
            kind=kind_of_rep[b.rep],
            scale=float(s.get(b.name, 1.0)),
            ranges=rng(b.idx) if b.rep == RHO1 else None,
        )
        for b in blocks
    ]
