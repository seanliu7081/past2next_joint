"""SE(2) rewrite of flattened MuJoCo sim states for LIBERO scenes.

Given a robosuite flattened state ``[time(1), qpos(nq), qvel(nv)]``
(``MjSimState.flatten()``), rewrite it so the whole scene is rotated by the
world-frame yaw R_z(theta) about the robot base: every movable object's
free-joint pose (xy about ``p_base``, wxyz quat left-multiplied by q_Rz(theta))
plus the robot's joint 1 (whose axis is the base z axis, so ``q1 += theta`` is
an exact rigid rotation of the whole arm -- no IK). qvel is left untouched:
the rewritten states are render-only inputs to ``regenerate_obs_from_state``.

Fixtures and the robot base live in ``model.body_pos/body_quat`` (not qpos) and
therefore do NOT rotate -- an accepted residual; :func:`check_support_contacts`
catches objects that end up unsupported over a non-rotated fixture and
:func:`check_object_penetration` catches objects rotated INTO one. Validity of
object placement is judged against the PHYSICAL table-top footprint
(:func:`table_top_xy_aabb`), not empirical occupancy: the arm co-rotates
exactly with the world so reachability is rotation-invariant, and the only
real xy failure mode is leaving the table surface.

Module-level imports are numpy/stdlib only (plus the pure-numpy
``oat.equi.se2_transforms``): safe to import in tests and dataloader workers.
Live-sim access (``resolve_addresses``, ``table_top_xy_aabb``,
``check_support_contacts``, ``check_object_penetration``) is duck-typed
against robosuite's ``MjSim`` wrapper -- no robosuite/mujoco import here.

Footgun reminder: MuJoCo qpos quaternions are **wxyz**; theta is in radians.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from oat.equi.se2_transforms import quat_mul_wxyz, quat_z_wxyz, rotate_xy

# Panda joint-1 position limit (rad), symmetric. Matches the <joint range> in
# the robosuite Panda model.
PANDA_JOINT1_LIMIT = 2.8973

# mujoco mjtGeom enum value for box geoms (stable across mujoco versions).
_MJ_GEOM_BOX = 6

# Free joints with these name prefixes belong to the robot stack, not to
# movable scene objects.
_ROBOT_PREFIXES = ("robot", "gripper", "mount")


@dataclass
class SceneAddresses:
    """Where things live inside qpos of one LIBERO scene (one env build)."""

    obj_qpos_slices: "OrderedDict[str, Tuple[int, int]]"  # free-joint name -> (start, end) into qpos, end-start == 7
    joint1_qpos_addr: int   # qpos address of robot0_joint1
    p_base: np.ndarray      # (3,) world position of the robot base / joint-1 axis anchor
    nq: int
    nv: int


def resolve_addresses(control_env) -> SceneAddresses:
    """Resolve qpos addressing for a live LIBERO ``ControlEnv``.

    Movable objects are enumerated as the sim.model FREE joints (qpos span 7)
    that are not robot-prefixed -- robust across LIBERO problem classes, no
    env-object attribute spelunking. ``p_base`` prefers the joint-1 axis
    anchor point (``data.xanchor``) and is asserted to agree in xy with the
    ``robot0_base`` body position; falls back to the body position if the sim
    wrapper does not expose ``xanchor``.
    """
    sim = control_env.sim
    model = sim.model

    obj_qpos_slices: "OrderedDict[str, Tuple[int, int]]" = OrderedDict()
    for name in model.joint_names:
        if name is None or name.startswith(_ROBOT_PREFIXES):
            continue
        addr = model.get_joint_qpos_addr(name)
        if isinstance(addr, tuple) and addr[1] - addr[0] == 7:  # FREE joint
            obj_qpos_slices[name] = (int(addr[0]), int(addr[1]))
    assert obj_qpos_slices, (
        "no non-robot free joints found -- unexpected for a LIBERO scene; "
        f"joint names: {list(model.joint_names)}"
    )

    joint1_qpos_addr = model.get_joint_qpos_addr("robot0_joint1")
    assert isinstance(joint1_qpos_addr, (int, np.integer)), (
        f"robot0_joint1 should be a 1-dim joint, got addr {joint1_qpos_addr}"
    )

    base_xpos = np.array(sim.data.get_body_xpos("robot0_base"), dtype=np.float64)
    anchor = None
    try:
        joint1_id = model.joint_name2id("robot0_joint1")
        anchor = np.array(sim.data.xanchor[joint1_id], dtype=np.float64)
    except AttributeError:
        pass
    if anchor is not None:
        assert np.allclose(anchor[:2], base_xpos[:2], atol=1e-4), (
            "joint-1 axis anchor xy disagrees with robot0_base body xy: "
            f"{anchor[:2]} vs {base_xpos[:2]} -- the joint-1 axis is expected "
            "to pass through the base (Panda); refusing to guess p_base"
        )
        p_base = anchor
    else:
        p_base = base_xpos

    return SceneAddresses(
        obj_qpos_slices=obj_qpos_slices,
        joint1_qpos_addr=int(joint1_qpos_addr),
        p_base=p_base,
        nq=int(model.nq),
        nv=int(model.nv),
    )


def rewrite_state(state: np.ndarray, theta: float, addr: SceneAddresses) -> np.ndarray:
    """Rotate a flattened state ``[time, qpos, qvel]`` by R_z(theta) about
    ``addr.p_base``. Returns a new float64 array; the input is not modified.

    qpos indexing carries the +1 time offset. Object quats are wxyz and
    left-multiplied (world rotation). qvel is untouched (render-only).
    """
    state = np.asarray(state, dtype=np.float64)
    assert state.ndim == 1 and len(state) == 1 + addr.nq + addr.nv, (
        f"flattened state length {state.shape} != 1 + nq({addr.nq}) + nv({addr.nv})"
    )
    out = state.copy()
    qz = quat_z_wxyz(theta)
    for start, _end in addr.obj_qpos_slices.values():
        s = 1 + start  # +1: time slot
        out[s:s + 3] = rotate_xy(out[s:s + 3], theta, center_xy=addr.p_base[:2])
        out[s + 3:s + 7] = quat_mul_wxyz(qz, out[s + 3:s + 7])
    out[1 + addr.joint1_qpos_addr] += theta
    return out


def object_xy_from_state(state: np.ndarray, addr: SceneAddresses) -> np.ndarray:
    """World xy of every movable object, ``(n_obj, 2)``, in
    ``addr.obj_qpos_slices`` order."""
    state = np.asarray(state, dtype=np.float64)
    if not addr.obj_qpos_slices:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack(
        [state[1 + start:1 + start + 2] for start, _end in addr.obj_qpos_slices.values()],
        axis=0,
    )


def table_top_xy_aabb(
    control_env,
    addr: SceneAddresses,
    z_below: float = 0.30,
    z_above: float = 0.02,
    axis_tol_deg: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Union world-xy AABB of the physical table-top box geoms of a live env.

    Call once per task on a freshly reset env (movable objects at their
    resting placement). Candidate geoms are collision-capable BOX geoms whose
    geom or body name contains 'table' (LIBERO arenas: e.g. body
    ``living_room_table_col`` with unnamed collision boxes) and whose world
    AABB top surface lies within ``[obj_min_z - z_below, obj_min_z + z_above]``
    where ``obj_min_z`` is the lowest movable-object free-joint z -- this keeps
    the top panels (plus harmless interior structure already inside the top's
    footprint) and drops shelves/walls far from the surface. If no candidate
    has collision enabled, visual 'table' boxes are used with a warning.

    Per-geom world xy AABB is ``xpos +/- (|xmat| @ size)``, exact when the box
    is axis-aligned up to a 90-degree permutation (LIBERO table boxes have
    local x mapped to world z). A box yawed beyond ``axis_tol_deg`` falls back
    to its conservative INSCRIBED axis-aligned rectangle; a tilted table box
    raises with a dump of every box geom so detection can be iterated.

    Returns ``(min_xy, max_xy, geom_labels)``.
    """
    sim = control_env.sim
    model, data = sim.model, sim.data
    geom_bodyid = np.asarray(model.geom_bodyid)
    cos_tol = float(np.cos(np.deg2rad(axis_tol_deg)))

    obj_z = [float(data.qpos[start + 2]) for start, _end in addr.obj_qpos_slices.values()]
    assert obj_z, "no movable objects -- cannot anchor the table-top z window"
    obj_min_z = min(obj_z)

    def _box_info(gid: int):
        center = np.asarray(data.geom_xpos[gid], dtype=np.float64)
        R = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        size = np.asarray(model.geom_size[gid], dtype=np.float64)
        half_world = np.abs(R) @ size  # circumscribed AABB half-extents
        return center, R, size, half_world

    def _label(gid: int) -> str:
        body = model.body_id2name(int(geom_bodyid[gid])) or "?"
        geom = model.geom_id2name(gid)
        return f"{body}:{geom if geom else f'geom{gid}'}"

    def _dump_boxes() -> str:
        lines = []
        for gid in range(int(model.ngeom)):
            if int(model.geom_type[gid]) != _MJ_GEOM_BOX:
                continue
            center, _R, size, half_world = _box_info(gid)
            lines.append(
                f"  {_label(gid)}: xpos={np.round(center, 4).tolist()} "
                f"size={np.round(size, 4).tolist()} "
                f"top_z={center[2] + half_world[2]:.4f} "
                f"contype={int(model.geom_contype[gid])} "
                f"conaffinity={int(model.geom_conaffinity[gid])}")
        return "\n".join(lines)

    candidates = []  # (gid, collision_enabled)
    for gid in range(int(model.ngeom)):
        if int(model.geom_type[gid]) != _MJ_GEOM_BOX:
            continue
        body_name = (model.body_id2name(int(geom_bodyid[gid])) or "").lower()
        geom_name = (model.geom_id2name(gid) or "").lower()
        if "table" not in body_name and "table" not in geom_name:
            continue
        center, _R, _size, half_world = _box_info(gid)
        top_z = float(center[2] + half_world[2])
        if not (obj_min_z - z_below <= top_z <= obj_min_z + z_above):
            continue
        collision = bool(model.geom_contype[gid]) or bool(model.geom_conaffinity[gid])
        candidates.append((gid, collision))

    if any(col for _gid, col in candidates):
        chosen = [gid for gid, col in candidates if col]
    elif candidates:
        chosen = [gid for gid, _col in candidates]
        print("[se2_state_rewrite] WARN: no COLLISION 'table' box near the "
              "object resting z; falling back to visual-only table boxes")
    else:
        raise RuntimeError(
            "table-top detection failed: no box geom with 'table' in its "
            f"geom/body name has a top surface within [{obj_min_z - z_below:.3f}, "
            f"{obj_min_z + z_above:.3f}] (objects' min resting z {obj_min_z:.3f} "
            f"- {z_below} / + {z_above}). All box geoms:\n" + _dump_boxes())

    min_xy = np.full(2, np.inf)
    max_xy = np.full(2, -np.inf)
    labels: List[str] = []
    for gid in chosen:
        center, R, size, half_world = _box_info(gid)
        absR = np.abs(R)
        if float(absR.max(axis=1).min()) >= cos_tol:
            # axis-aligned up to a 90-degree permutation: AABB is exact
            half_xy = half_world[:2]
        else:
            # yawed about world z: conservative inscribed axis-aligned rect
            jz = int(np.argmax(absR[2]))
            if absR[2, jz] < cos_tol:
                raise RuntimeError(
                    f"table box {_label(gid)} is tilted (|xmat| row z = "
                    f"{np.round(absR[2], 4).tolist()}); refusing to guess its "
                    "footprint. All box geoms:\n" + _dump_boxes())
            jx, jy = [j for j in range(3) if j != jz]
            a, b = float(size[jx]), float(size[jy])
            u = absR[0, jx] * a - absR[0, jy] * b
            v = absR[1, jy] * b - absR[1, jx] * a
            if u <= 0 or v <= 0:
                raise RuntimeError(
                    f"table box {_label(gid)} is yawed too far for a useful "
                    f"inscribed rectangle (half-extents {u:.4f}, {v:.4f}). "
                    "All box geoms:\n" + _dump_boxes())
            print(f"[se2_state_rewrite] WARN: table box {_label(gid)} not "
                  f"axis-aligned; using conservative inscribed rect "
                  f"({u:.3f}, {v:.3f}) m")
            half_xy = np.array([u, v], dtype=np.float64)
        min_xy = np.minimum(min_xy, center[:2] - half_xy)
        max_xy = np.maximum(max_xy, center[:2] + half_xy)
        labels.append(_label(gid))

    return min_xy, max_xy, labels


# ── validity checks ──────────────────────────────────────────────────────────

def check_joint1_limit(
    state_rw: np.ndarray, addr: SceneAddresses, margin: float = 0.05
) -> Tuple[bool, str]:
    """Joint 1 of the rewritten state must stay inside the Panda limit minus
    ``margin`` (rad)."""
    q1 = float(np.asarray(state_rw, dtype=np.float64)[1 + addr.joint1_qpos_addr])
    limit = PANDA_JOINT1_LIMIT - margin
    if abs(q1) <= limit:
        return True, "ok"
    return False, (
        f"joint1 qpos {q1:.4f} rad outside +/-{limit:.4f} "
        f"(limit {PANDA_JOINT1_LIMIT} - margin {margin})"
    )


def check_objects_in_bounds(
    state_rw: np.ndarray,
    addr: SceneAddresses,
    bounds_min_xy: np.ndarray,
    bounds_max_xy: np.ndarray,
    margin: float = 0.05,
) -> Tuple[bool, str]:
    """Every object center xy of the rewritten state must lie inside the
    PHYSICAL table-top AABB (:func:`table_top_xy_aabb`) inset by ``margin``
    meters (the margin SHRINKS the box: objects rotated to within ``margin``
    of the table edge are rejected). The arm co-rotates exactly with the
    world, so reachability is rotation-invariant and needs no check; leaving
    the table surface is the real failure mode."""
    lo = np.asarray(bounds_min_xy, dtype=np.float64) + margin
    hi = np.asarray(bounds_max_xy, dtype=np.float64) - margin
    xy = object_xy_from_state(state_rw, addr)
    for name, p in zip(addr.obj_qpos_slices, xy):
        if not (np.all(p >= lo) and np.all(p <= hi)):
            return False, (
                f"object '{name}' xy={np.round(p, 4).tolist()} outside table "
                f"bounds [{np.round(lo, 4).tolist()}, {np.round(hi, 4).tolist()}] "
                f"(inset {margin})"
            )
    return True, "ok"


def _object_root_ids(model, addr: SceneAddresses) -> "OrderedDict[str, int]":
    """Free-joint name -> body root id of the object's kinematic subtree (the
    free-joint body IS the root)."""
    body_rootid = np.asarray(model.body_rootid)
    jnt_bodyid = np.asarray(model.jnt_bodyid)
    return OrderedDict(
        (name, int(body_rootid[jnt_bodyid[model.joint_name2id(name)]]))
        for name in addr.obj_qpos_slices
    )


def check_support_contacts(
    control_env, addr: SceneAddresses, cos_tol_deg: float = 30.0
) -> Tuple[bool, str]:
    """Each movable object must be physically supported in the rewritten scene.

    Call AFTER ``set_state`` + ``sim.forward()`` (e.g. after
    ``regenerate_obs_from_state``) so ``sim.data.contact`` is current. An
    object counts as supported if it has at least one contact whose normal is
    within ``cos_tol_deg`` of +/-z (resting on something) OR that involves a
    gripper geom (held). Catches objects left hovering over fixtures that do
    not rotate with the scene.
    """
    sim = control_env.sim
    model, data = sim.model, sim.data

    geom_bodyid = np.asarray(model.geom_bodyid)
    body_rootid = np.asarray(model.body_rootid)

    obj_root = _object_root_ids(model, addr)

    def _is_gripper_geom(geom_id: int) -> bool:
        body_name = model.body_id2name(int(geom_bodyid[geom_id])) or ""
        geom_name = model.geom_id2name(int(geom_id)) or ""
        return "gripper" in body_name.lower() or "gripper" in geom_name.lower()

    cos_tol = np.cos(np.deg2rad(cos_tol_deg))
    supported = {name: False for name in obj_root}
    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        r1 = int(body_rootid[geom_bodyid[g1]])
        r2 = int(body_rootid[geom_bodyid[g2]])
        # contact frame: first 3 entries are the normal (geom1 -> geom2)
        normal_ok = abs(float(np.asarray(con.frame).ravel()[2])) >= cos_tol
        for name, root in obj_root.items():
            if supported[name]:
                continue
            if root == r1:
                other = g2
            elif root == r2:
                other = g1
            else:
                continue
            if normal_ok or _is_gripper_geom(other):
                supported[name] = True

    missing = [name for name, ok in supported.items() if not ok]
    if missing:
        return False, (
            f"objects with no supporting contact (normal within {cos_tol_deg} deg "
            f"of +/-z, or gripper touch): {missing}"
        )
    return True, "ok"


def check_object_penetration(
    control_env, addr: SceneAddresses, depth_tol: float = 0.005
) -> Tuple[bool, str]:
    """No contact involving a movable object may penetrate deeper than
    ``depth_tol`` meters (mujoco ``contact.dist < -depth_tol``).

    Call AFTER ``set_state`` + ``sim.forward()`` (same as
    :func:`check_support_contacts`). Catches objects rotated INTO non-rotating
    fixtures, which remain 'supported' (they still have contacts) but are
    unphysically embedded. Co-rotating pairs (object-object, object-arm/
    gripper) keep their theta=0 contact geometry exactly, so any depth they
    show is rotation-independent and the caller's theta=0 exemption absorbs
    it. Returns a reason of the form ``'penetration: <obj> ...'``.
    """
    sim = control_env.sim
    model, data = sim.model, sim.data

    geom_bodyid = np.asarray(model.geom_bodyid)
    body_rootid = np.asarray(model.body_rootid)
    root_to_obj = {root: name for name, root in _object_root_ids(model, addr).items()}

    worst_obj, worst_dist = None, -depth_tol
    for i in range(int(data.ncon)):
        con = data.contact[i]
        dist = float(con.dist)
        if dist >= worst_dist:
            continue
        for g in (int(con.geom1), int(con.geom2)):
            name = root_to_obj.get(int(body_rootid[geom_bodyid[g]]))
            if name is not None:
                worst_obj, worst_dist = name, dist
                break
    if worst_obj is not None:
        return False, (
            f"penetration: {worst_obj} contact depth {-worst_dist * 1000:.1f} mm "
            f"> {depth_tol * 1000:.1f} mm"
        )
    return True, "ok"
