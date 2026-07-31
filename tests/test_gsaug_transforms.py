"""Tests for the sim-free GS math modules (plan §6.5, M5 gate):
``oat.gsaug.sh_rotation``, ``oat.gsaug.components``, ``oat.gsaug.gaussian_asset``.

The brute-force SH checks evaluate coefficients through a local numpy port of
gsplat's real-SH basis (``_eval_sh_bases_fast``, bands l=0..3 in the 3DGS
m=-l..l layout) so the rotation functions are pinned to the EXACT basis the
rasterizer uses — "rotating a component by R" must satisfy f_rot(d) = f(R^T d)
(G5). CPU-only: no gsplat, no CUDA, no sim.
"""

import math

import numpy as np
import pytest
import torch

from oat.equi.se2_transforms import quat_mul_wxyz
from oat.gsaug.components import (
    PosedComponent,
    WorldGaussians,
    quat_conj,
    quat_mul,
    quat_normalize,
    quat_to_R,
)
from oat.gsaug.gaussian_asset import EXPECTED_CONVENTIONS, GaussianAsset
from oat.gsaug.sh_rotation import (
    rotate_sh_l1,
    rotate_sh_so3,
    rotate_sh_z,
    sh_degree_of,
)

# ── tiny real-SH evaluator (numpy port of gsplat _eval_sh_bases_fast, l<=3) ──
# Constants verbatim from gsplat/cuda/_torch_impl.py; layout: bands l=0..3,
# m=-l..l, indices 0..15.


def eval_sh_bases(dirs: np.ndarray) -> np.ndarray:
    """(D,3) unit dirs -> (D,16) basis values in the gsplat/3DGS layout."""
    d = np.asarray(dirs, dtype=np.float64)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    out = np.empty((d.shape[0], 16), dtype=np.float64)
    out[:, 0] = 0.2820947917738781
    fA = -0.48860251190292
    out[:, 1] = fA * y
    out[:, 2] = -fA * z
    out[:, 3] = fA * x
    z2 = z * z
    fB = -1.092548430592079 * z
    fA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2.0 * x * y
    out[:, 4] = fA * fS1
    out[:, 5] = fB * y
    out[:, 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    out[:, 7] = fB * x
    out[:, 8] = fA * fC1
    fC = -2.285228997322329 * z2 + 0.4570457994644658
    fB = 1.445305721320277 * z
    fA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    out[:, 9] = fA * fS2
    out[:, 10] = fB * fS1
    out[:, 11] = fC * y
    out[:, 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    out[:, 13] = fC * x
    out[:, 14] = fB * fC1
    out[:, 15] = fA * fC2
    return out


def eval_sh(sh: torch.Tensor, dirs: np.ndarray) -> np.ndarray:
    """Evaluate (N,K,3) coefficients at (D,3) dirs -> (N,D,3) function values."""
    K = sh.shape[-2]
    bases = eval_sh_bases(dirs)[:, :K]                    # (D,K)
    c = sh.detach().cpu().double().numpy()                # (N,K,3)
    return np.einsum("dk,nkc->ndc", bases, c)


def unit_dirs(n=2000, seed=0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rodrigues(axis, angle) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * Kx + (1.0 - math.cos(angle)) * Kx @ Kx


def aa_quat(axis, angle) -> np.ndarray:
    """wxyz quaternion for a rotation of ``angle`` about ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * a])


def rand_sh(n, deg, seed=0, scale=1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, (deg + 1) ** 2, 3, generator=g) * scale


# ── rotate_sh_z ──────────────────────────────────────────────────────────────

def test_rotate_sh_z_theta0_identity():
    sh = rand_sh(4, 3, seed=1)
    torch.testing.assert_close(rotate_sh_z(sh, 0.0), sh, rtol=0.0, atol=1e-7)


def test_rotate_sh_z_composition():
    sh = rand_sh(4, 3, seed=2)
    a, b = 0.41, -1.13
    ab = rotate_sh_z(rotate_sh_z(sh, a), b)
    torch.testing.assert_close(ab, rotate_sh_z(sh, a + b), rtol=0.0, atol=1e-5)


def test_rotate_sh_z_matches_l1_on_deg1():
    sh = rand_sh(5, 1, seed=3)
    theta = 0.87
    out_z = rotate_sh_z(sh, theta)
    out_l1 = rotate_sh_l1(sh, rodrigues([0, 0, 1], theta))
    torch.testing.assert_close(out_z, out_l1, rtol=0.0, atol=1e-6)


def test_rotate_sh_z_brute_force_deg3():
    # f_rot(d) == f(R_z(theta)^T d) on a dense direction grid (G5 convention).
    sh = rand_sh(3, 3, seed=4)
    theta = 0.7321
    dirs = unit_dirs()
    Rz = rodrigues([0, 0, 1], theta)
    f_rot = eval_sh(rotate_sh_z(sh, theta), dirs)
    f_ref = eval_sh(sh, dirs @ Rz)      # rows of dirs @ R are R^T d
    np.testing.assert_allclose(f_rot, f_ref, atol=1e-5)


# ── rotate_sh_l1 ─────────────────────────────────────────────────────────────

def test_rotate_sh_l1_identity():
    sh = rand_sh(4, 1, seed=5)
    torch.testing.assert_close(rotate_sh_l1(sh, np.eye(3)), sh,
                               rtol=0.0, atol=1e-7)


def test_rotate_sh_l1_composition():
    sh = rand_sh(4, 1, seed=6)
    R1 = rodrigues([0.3, -0.5, 0.8], 1.1)
    R2 = rodrigues([-0.6, 0.2, 0.4], 0.7)
    out = rotate_sh_l1(rotate_sh_l1(sh, R1), R2)
    torch.testing.assert_close(out, rotate_sh_l1(sh, R2 @ R1),
                               rtol=0.0, atol=1e-5)


def test_rotate_sh_l1_brute_force_random_R():
    # The l=1 basis is a SIGNED permutation (-y, +z, -x); this check pins the
    # sign conjugation in rotate_sh_l1 against the actual gsplat basis.
    sh = rand_sh(3, 1, seed=7)
    R = rodrigues([0.3, -0.5, 0.8], 1.1)   # mixes z with x/y
    dirs = unit_dirs(seed=1)
    f_rot = eval_sh(rotate_sh_l1(sh, R), dirs)
    f_ref = eval_sh(sh, dirs @ R)
    np.testing.assert_allclose(f_rot, f_ref, atol=1e-5)


def test_rotate_sh_l1_rejects_wrong_degree():
    with pytest.raises(ValueError, match="K=4"):
        rotate_sh_l1(rand_sh(2, 2, seed=8), np.eye(3))


# ── rotate_sh_so3 (production general-SO(3) path, R7) ────────────────────────

def test_rotate_sh_so3_matches_z_path_deg3():
    # For a pure z-rotation the projection path must agree with the exact
    # closed form at deg 3 — pins so3_deg3's two paths to each other.
    sh = rand_sh(4, 3, seed=20)
    theta = 1.234
    out = rotate_sh_so3(sh, rodrigues([0, 0, 1], theta))
    torch.testing.assert_close(out, rotate_sh_z(sh, theta), rtol=0.0, atol=1e-5)


def test_rotate_sh_so3_band1_matches_l1():
    # Band 1 under a random z-mixing R must reproduce the independently
    # verified signed-permutation l=1 rule.
    sh = rand_sh(5, 1, seed=21)
    R = rodrigues([0.3, -0.5, 0.8], 1.1)
    torch.testing.assert_close(rotate_sh_so3(sh, R), rotate_sh_l1(sh, R),
                               rtol=0.0, atol=1e-6)


def test_rotate_sh_so3_brute_force_deg3():
    # f_rot(d) == f(R^T d) on the dense grid for a z-mixing R at deg 3: the
    # projection is exact in the very basis the rasterizer evaluates (G5).
    sh = rand_sh(3, 3, seed=22)
    R = rodrigues([0.4, -0.7, 0.55], 1.3)   # mixes z with x/y
    dirs = unit_dirs(seed=2)
    f_rot = eval_sh(rotate_sh_so3(sh, R), dirs)
    f_ref = eval_sh(sh, dirs @ R)
    np.testing.assert_allclose(f_rot, f_ref, atol=1e-5)


def test_rotate_sh_so3_composition():
    sh = rand_sh(4, 3, seed=23)
    R1 = rodrigues([0.3, -0.5, 0.8], 1.1)
    R2 = rodrigues([-0.6, 0.2, 0.4], 0.7)
    out = rotate_sh_so3(rotate_sh_so3(sh, R1), R2)
    torch.testing.assert_close(out, rotate_sh_so3(sh, R2 @ R1),
                               rtol=0.0, atol=1e-5)


def test_sh_degree_of():
    assert sh_degree_of(rand_sh(1, 0)) == 0
    assert sh_degree_of(rand_sh(1, 3)) == 3
    with pytest.raises(ValueError, match="not"):
        sh_degree_of(torch.zeros(1, 5, 3))


# ── quaternion helpers (components) ──────────────────────────────────────────

def test_quat_mul_matches_se2_transforms():
    rng = np.random.default_rng(9)
    q1 = rng.normal(size=(64, 4))
    q2 = rng.normal(size=(64, 4))
    got = quat_mul(torch.as_tensor(q1, dtype=torch.float32),
                   torch.as_tensor(q2, dtype=torch.float32)).numpy()
    want = quat_mul_wxyz(q1, q2)   # numpy reference, same wxyz convention
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_quat_conj_inverts_unit_quats():
    rng = np.random.default_rng(10)
    q = quat_normalize(torch.as_tensor(rng.normal(size=(16, 4)),
                                       dtype=torch.float32))
    ident = quat_mul(q, quat_conj(q)).numpy()
    want = np.tile([1.0, 0, 0, 0], (16, 1))
    np.testing.assert_allclose(ident, want, atol=1e-6)


def test_quat_to_R_orthonormal_and_matches_scipy():
    rng = np.random.default_rng(11)
    q = rng.normal(size=(32, 4))
    R = quat_to_R(torch.as_tensor(q, dtype=torch.float32)).numpy()
    eye = np.einsum("nij,nkj->nik", R, R)
    np.testing.assert_allclose(eye, np.tile(np.eye(3), (32, 1, 1)), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(R), np.ones(32), atol=1e-5)

    scipy_spatial = pytest.importorskip("scipy.spatial.transform")
    q_unit = q / np.linalg.norm(q, axis=-1, keepdims=True)
    want = scipy_spatial.Rotation.from_quat(q_unit[:, [1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(R, want, atol=1e-5)


# ── PosedComponent ───────────────────────────────────────────────────────────

def make_component(mode, deg, n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return PosedComponent(
        "test",
        means_l=torch.randn(n, 3, generator=g) * 0.25,
        quats_l=torch.randn(n, 4, generator=g),
        log_scales=torch.rand(n, 3, generator=g) - 3.2,
        opacity_logits=torch.full((n,), 1.5),
        sh=rand_sh(n, deg, seed=seed + 100, scale=0.3),
        sh_rot_mode=mode,
        p_capture=torch.zeros(3),
        q_capture=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )


def test_tumbled_pose_rotates_sh_exactly():
    # R7: on real demos objects tilt at reset and tumble when grasped, so a
    # non-z delta must ROTATE the SH exactly — never assert, never pass SH
    # through unrotated (G5).
    comp = make_component("so3_deg3", 3)
    axis, ang = [1.0, 0.2, 0.0], 0.3           # rotation with an x component
    q_tumble = aa_quat(axis, ang)
    p = np.array([0.05, -0.1, 0.2])
    world = comp.posed(p, q_tumble)

    Rt = torch.as_tensor(rodrigues(axis, ang), dtype=torch.float32)
    want_means = comp.means_l @ Rt.T + torch.as_tensor(p, dtype=torch.float32)
    torch.testing.assert_close(world.means, want_means, rtol=0.0, atol=1e-6)
    q_t = torch.as_tensor(q_tumble, dtype=torch.float32)
    want_quats = quat_normalize(
        quat_mul(q_t.expand_as(comp.quats_l), comp.quats_l))
    torch.testing.assert_close(world.quats, want_quats, rtol=0.0, atol=1e-6)
    # q_capture is identity, so the delta IS q_tumble: SH must equal the
    # exact SO(3) rotation of the local SH.
    want_sh = rotate_sh_so3(comp.sh, quat_to_R(q_t))
    torch.testing.assert_close(world.sh, want_sh, rtol=0.0, atol=1e-6)
    assert not torch.allclose(world.sh, comp.sh)   # actually rotated


def test_z_only_deg3_alias_maps_to_so3_deg3():
    comp = make_component("z_only_deg3", 3)    # deprecated alias still constructs
    assert comp.sh_rot_mode == "so3_deg3"
    world = comp.posed(np.zeros(3), aa_quat([1.0, 0.2, 0.0], 0.3))  # no assert
    torch.testing.assert_close(
        world.sh,
        rotate_sh_so3(comp.sh, quat_to_R(
            torch.as_tensor(aa_quat([1.0, 0.2, 0.0], 0.3),
                            dtype=torch.float32))),
        rtol=0.0, atol=1e-6)


def test_pure_z_pose_accepted_and_moves_means():
    comp = make_component("so3_deg3", 3)
    theta = math.radians(25.0)
    p = np.array([0.1, -0.2, 0.05])
    world = comp.posed(p, aa_quat([0, 0, 1], theta))
    Rz = torch.as_tensor(rodrigues([0, 0, 1], theta), dtype=torch.float32)
    want = comp.means_l @ Rz.T + torch.as_tensor(p, dtype=torch.float32)
    torch.testing.assert_close(world.means, want, rtol=0.0, atol=1e-6)
    # SH actually rotated (not passed through)
    assert not torch.allclose(world.sh, comp.sh)
    torch.testing.assert_close(world.sh, rotate_sh_z(comp.sh, theta),
                               rtol=0.0, atol=1e-5)


def test_static_background_rejects_nonidentity_pose():
    comp = make_component("static", 3)
    with pytest.raises(AssertionError, match="static"):
        comp.posed(np.array([0.1, 0.0, 0.0]), np.array([1.0, 0, 0, 0]))
    with pytest.raises(AssertionError, match="static"):
        comp.posed(np.zeros(3), aa_quat([0, 0, 1], 0.2))
    world = comp.posed_identity()   # identity pose is fine, SH untouched
    torch.testing.assert_close(world.sh, comp.sh)
    torch.testing.assert_close(world.means, comp.means_l)


def test_so3_deg1_requires_degree1():
    with pytest.raises(ValueError, match="degree 1"):
        make_component("so3_deg1", 3)


def test_world_gaussians_concat_pads_sh():
    a = make_component("so3_deg1", 1).posed(np.zeros(3), [1.0, 0, 0, 0])
    b = make_component("so3_deg3", 3, seed=1).posed(np.zeros(3),
                                                    [1.0, 0, 0, 0])
    joint = WorldGaussians.concat([a, b])
    assert joint.sh.shape == (10, 16, 3) and joint.sh_degree == 3
    torch.testing.assert_close(joint.sh[:5, :4], a.sh)
    assert (joint.sh[:5, 4:] == 0).all()   # padded bands are exactly zero
    torch.testing.assert_close(joint.sh[5:], b.sh)


# ── GaussianAsset ────────────────────────────────────────────────────────────

def make_asset(n=6, deg=2, frame="body"):
    g = torch.Generator().manual_seed(3)
    K = (deg + 1) ** 2
    return GaussianAsset(
        means=torch.randn(n, 3, generator=g),
        quats=torch.randn(n, 4, generator=g),
        log_scales=torch.randn(n, 3, generator=g) * 0.1 - 3.0,
        opacity_logits=torch.randn(n, generator=g),
        sh_dc=torch.randn(n, 3, generator=g),
        sh_rest=torch.randn(n, K - 1, 3, generator=g),
        conventions=dict(EXPECTED_CONVENTIONS),
        meta={"frame": frame, "task": "unit_test",
              "p_capture": [0.0, 0.0, 0.0],
              "q_capture": [1.0, 0.0, 0.0, 0.0]},
    )


def test_asset_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "asset.pt")
    asset = make_asset()
    digest = asset.save(path)
    loaded = GaussianAsset.load(path, expected_frame="body")
    assert loaded.sha1() == digest
    torch.testing.assert_close(loaded.means, asset.means)
    torch.testing.assert_close(loaded.sh_rest, asset.sh_rest)
    assert loaded.sh_degree == 2 and loaded.frame == "body"
    with pytest.raises(RuntimeError, match="frame"):
        GaussianAsset.load(path, expected_frame="world")


def test_asset_conventions_mismatch_refused(tmp_path):
    # direct: an asset built under other conventions never validates/saves
    bad = make_asset()
    bad.conventions["quat_order"] = "xyzw"
    with pytest.raises(RuntimeError, match="convention"):
        bad.validate()
    with pytest.raises(RuntimeError, match="convention"):
        bad.save(str(tmp_path / "bad.pt"))

    # load path: payload surgery on a valid file's conventions block
    path = str(tmp_path / "asset.pt")
    make_asset().save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["conventions"]["sh_layout"] = "some_other_layout"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="convention"):
        GaussianAsset.load(path)


def test_asset_sha1_tamper_refused(tmp_path):
    path = str(tmp_path / "asset.pt")
    make_asset().save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["params"]["means"] = payload["params"]["means"] + 1e-3
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="sha1"):
        GaussianAsset.load(path)
