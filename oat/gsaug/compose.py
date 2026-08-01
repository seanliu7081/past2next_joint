"""GSCompositeRenderer: the compositional 3DGS render source (plan §6.2).

Replaces ``env.regenerate_obs_from_state(state_rw)`` as the *pixel source* of
the SE(2) augmentation. MuJoCo stays in the loop for everything except
rasterization (G3): per frame we ``set_state_from_flattened + forward()`` and
read object body poses, robot link poses, and camera extrinsics from the
forwarded sim — but never call a MuJoCo renderer.

Composition model (G1/G2):
    * background — world-frame Gaussians, identity-posed, SH never rotated
      (``sh_rot_mode='static'``);
    * one component per movable free joint — body-frame Gaussians posed by the
      body's current ``data.xpos/xquat``, SH rotated exactly under the full
      capture->current SO(3) delta (``'so3_deg3'``, G5: closed-form z fast
      path, exact projection otherwise — real demos tilt and tumble, R7);
    * one component per robot link — link-frame Gaussians from the per-task
      robot asset split by ``link_id`` (G10), SH degree 1 rotated under full
      SO(3) (``'so3_deg1'``, G5).
    All components are concatenated into ONE Gaussian set and rasterized in a
    SINGLE ``gsplat.rasterization`` call per frame covering all cameras at once
    (G2 — image-space compositing of separately rendered components is
    forbidden; it breaks occlusion).

Conventions are measured, never assumed (G7): the ctor loads
``gs_render_facts.json`` (must PASS) and uses the recorded F1 GL->CV flip for
extrinsics and the F2 orientation for output; ``render`` returns uint8 frames
in DATASET orientation, directly writable next to oracle
``np.flip(obs, axis=0)`` frames.

Provenance (G9): the ctor loads ``manifest.json`` from ``task_assets_dir``,
verifies every asset's content sha1 against the manifest, and the first
``render`` against a given env asserts
``sha1(env.sim.model.get_xml())`` == ``manifest['model_xml_sha1']`` — a
renderer constructed against the wrong task env fails loudly, never produces
subtly wrong pixels.
"""

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch

import gsplat

from oat.env.libero.se2_state_rewrite import resolve_addresses
from oat.gsaug import cameras as cam
from oat.gsaug.components import PosedComponent, WorldGaussians
from oat.gsaug.gaussian_asset import GaussianAsset


def model_xml_sha1_of(env) -> str:
    """The pinned model-hash recipe (G9). Use THIS everywhere the hash is
    compared (capture manifest, renderer, prerender). Delegates to
    ``oat.gsaug.capture.model_xml_sha1`` — sha1 over the CANONICALIZED XML
    (off-buffer attributes stripped so capture-at-512 and render-at-128 envs
    of the same scene hash identically)."""
    from oat.gsaug.capture import model_xml_sha1
    return model_xml_sha1(env)


@dataclass
class _SceneBinding:
    """Lazily-resolved ids for one live env's model. The model OBJECT is held
    and compared by identity (``is``) — an ``id()`` key without a reference
    could be reused by a freed model and silently skip the G9 re-bind."""
    model: object
    obj_body_ids: "OrderedDict[str, int]"    # free-joint name -> body id
    link_body_ids: "OrderedDict[str, int]"   # robot link (body) name -> body id
    cam_ids: "OrderedDict[str, int]"         # zarr image key -> camera id
    Ks: torch.Tensor                         # (C,3,3) float32 on device


class GSCompositeRenderer:
    """Single-pass compositional GS renderer for one task (plan §6.2).

    Args:
        task_assets_dir: ``data/libero/gs_assets/<task>`` — must contain
            ``manifest.json`` and the asset files it references.
        cameras: ordered mapping zarr image key -> MuJoCo camera name, e.g.
            ``{'agentview_rgb': 'agentview',
               'robot0_eye_in_hand_rgb': 'robot0_eye_in_hand'}``.
            The mapping is explicit — no '_image'-suffix convention is assumed.
        resolution: square output size in pixels (G8: read from the base zarr
            by the caller, never hard-coded).
        facts_path: ``gs_render_facts.json`` (must PASS — G7 gate).
        device: CUDA device for Gaussians + rasterization.
    """

    def __init__(self, task_assets_dir: str, cameras: "OrderedDict[str, str]",
                 resolution: int, facts_path: str = cam.DEFAULT_FACTS_PATH,
                 device: str = "cuda:0"):
        assert cameras, "cameras mapping is empty — need at least one camera"
        self.task_assets_dir = str(task_assets_dir)
        self.cameras: "OrderedDict[str, str]" = OrderedDict(cameras)
        self.resolution = int(resolution)
        assert self.resolution > 0, f"resolution={resolution!r}"
        self.device = torch.device(device)

        # G7: conventions are measured facts, asserted at construction.
        self.facts = cam.load_render_facts(facts_path)  # raises unless pass==true
        self._flip = cam.facts_flip(self.facts)                   # F1
        self._flip_ud = cam.facts_orientation_flip_ud(self.facts)  # F2

        # ── manifest (G9) ───────────────────────────────────────────────────
        manifest_path = os.path.join(self.task_assets_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"no manifest.json in '{self.task_assets_dir}': train assets "
                f"first (scripts/gsaug/train_gs_assets.py) or point "
                f"task_assets_dir at data/libero/gs_assets/<task>.")
        with open(manifest_path) as f:
            self.manifest: dict = json.load(f)
        for key in ("task", "model_xml_sha1", "assets"):
            if key not in self.manifest:
                raise RuntimeError(
                    f"'{manifest_path}' lacks required key '{key}' — not a "
                    f"train_gs_assets.py manifest (G9).")
        self.task: str = self.manifest["task"]
        self.model_xml_sha1: str = self.manifest["model_xml_sha1"]
        self.manifest_sha1: Optional[str] = self.manifest.get("manifest_sha1")
        assets = self.manifest["assets"]
        for key in ("background", "objects", "robot"):
            if key not in assets:
                raise RuntimeError(
                    f"'{manifest_path}' assets block lacks '{key}' — Phase 0 "
                    f"composition needs background + objects + robot (G1).")

        # ── components ──────────────────────────────────────────────────────
        bg_asset = self._load_asset(assets["background"], "background",
                                    expected_frame="world")
        self._background = PosedComponent.from_asset(
            bg_asset, "background", "static", device=self.device)
        # Background never moves: pose once, reuse the world set every frame.
        self._bg_world: WorldGaussians = self._background.posed_identity()

        self._objects: "OrderedDict[str, PosedComponent]" = OrderedDict()
        for joint_name, entry in assets["objects"].items():
            a = self._load_asset(entry, f"objects/{joint_name}",
                                 expected_frame="body")
            for k in ("p_capture", "q_capture"):
                if k not in a.meta:
                    raise RuntimeError(
                        f"object asset '{joint_name}' meta lacks '{k}': a "
                        f"body-frame asset without its capture body pose "
                        f"cannot drive the SH delta rotation (G5).")
            self._objects[joint_name] = PosedComponent.from_asset(
                a, joint_name, "so3_deg3", device=self.device)

        robot_asset = self._load_asset(assets["robot"], "robot",
                                       expected_frame="link")
        self._robot_links = self._split_robot(robot_asset)

        self._binding: Optional[_SceneBinding] = None

    # ── asset loading (G9) ──────────────────────────────────────────────────

    def _resolve_asset_path(self, entry_path: Optional[str], label: str) -> str:
        candidates: List[str] = []
        if entry_path:
            candidates.append(entry_path)
            if not os.path.isabs(entry_path):
                candidates.append(os.path.join(self.task_assets_dir, entry_path))
        candidates.append(
            os.path.join(self.task_assets_dir, "assets", label + ".pt"))
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(
            f"asset '{label}' for task '{self.task}' not found; tried "
            f"{candidates} — re-run scripts/gsaug/train_gs_assets.py or fix "
            f"the manifest 'path' entry.")

    def _load_asset(self, entry: dict, label: str,
                    expected_frame: str) -> GaussianAsset:
        path = self._resolve_asset_path(entry.get("path"), label)
        asset = GaussianAsset.load(path, expected_frame=expected_frame)
        digest = asset.sha1()
        want = entry.get("sha1")
        if want and want != digest:
            raise RuntimeError(
                f"asset '{label}' ('{path}') sha1 {digest[:12]}… does not "
                f"match manifest {want[:12]}… — stale or swapped asset file; "
                f"re-run train_gs_assets.py (G9).")
        meta_task = asset.meta.get("task")
        if meta_task is not None and meta_task != self.task:
            raise RuntimeError(
                f"asset '{label}' was trained for task '{meta_task}' but the "
                f"manifest is for '{self.task}' — mixed asset directories (G9).")
        return asset

    def _split_robot(self, asset: GaussianAsset) -> "OrderedDict[str, PosedComponent]":
        """One PosedComponent per robot link, split by ``link_id`` (plan §6.4)."""
        if asset.link_id is None:
            raise RuntimeError(
                "robot asset has no link_id tensor — cannot split into "
                "per-link components; retrain with oat/gsaug/articulated.py.")
        if asset.sh_degree != 1:
            raise RuntimeError(
                f"robot asset has SH degree {asset.sh_degree}, expected 1: "
                f"robot links use the exact l=1 SO(3) rotation (G5).")
        meta = asset.meta
        link_names = meta.get("link_names")
        if not link_names:
            raise RuntimeError(
                "robot asset meta lacks 'link_names' (ordered) — link_id "
                "values are indices into it and cannot be interpreted.")
        poses = self._robot_link_capture_poses(meta, link_names)

        link_ids = asset.link_id.to(torch.long)
        lo, hi = int(link_ids.min()), int(link_ids.max())
        if lo < 0 or hi >= len(link_names):
            raise RuntimeError(
                f"robot asset link_id range [{lo}, {hi}] out of bounds for "
                f"{len(link_names)} link_names — asset/meta mismatch (G9).")

        sh_full = asset.sh_full()
        comps: "OrderedDict[str, PosedComponent]" = OrderedDict()
        for i, lname in enumerate(link_names):
            mask = link_ids == i
            if not bool(mask.any()):
                continue  # a link that contributed no Gaussians renders as nothing
            p_cap, q_cap = poses[lname]
            comps[lname] = PosedComponent(
                lname,
                means_l=asset.means[mask], quats_l=asset.quats[mask],
                log_scales=asset.log_scales[mask],
                opacity_logits=asset.opacity_logits[mask],
                sh=sh_full[mask], sh_rot_mode="so3_deg1",
                p_capture=torch.as_tensor(np.asarray(p_cap, dtype=np.float64),
                                          dtype=torch.float32),
                q_capture=torch.as_tensor(np.asarray(q_cap, dtype=np.float64),
                                          dtype=torch.float32),
            ).to(self.device)
        if not comps:
            raise RuntimeError("robot asset split produced zero non-empty links")
        return comps

    @staticmethod
    def _robot_link_capture_poses(meta: dict, link_names: List[str]) -> Dict[str, tuple]:
        """Per-link capture world poses ``{link: (p(3), q_wxyz(4))}``.

        Contract shape: meta['p_capture'] / meta['q_capture'] are dicts keyed
        by link name for robot assets. The capture ``transforms.json`` shape
        (``link_poses: {name: {'p': .., 'q_wxyz': ..}}``) is accepted as a
        fallback.
        """
        p_cap, q_cap = meta.get("p_capture"), meta.get("q_capture")
        try:
            if isinstance(p_cap, Mapping) and isinstance(q_cap, Mapping):
                return {n: (p_cap[n], q_cap[n]) for n in link_names}
            link_poses = meta.get("link_poses")
            if isinstance(link_poses, Mapping):
                return {n: (link_poses[n]["p"], link_poses[n]["q_wxyz"])
                        for n in link_names}
        except KeyError as e:
            raise RuntimeError(
                f"robot asset meta lacks a capture pose for link {e.args[0]!r} "
                f"— every entry of link_names needs one (G5).") from e
        raise RuntimeError(
            "robot asset meta carries no per-link capture poses: expected "
            "dict-valued 'p_capture'/'q_capture' keyed by link name (or "
            "'link_poses' in the capture transforms.json shape).")

    # ── lazy scene binding (G9) ─────────────────────────────────────────────

    def _bind_scene(self, env) -> _SceneBinding:
        model = env.sim.model
        if self._binding is not None and self._binding.model is model:
            return self._binding

        live = model_xml_sha1_of(env)
        if live != self.model_xml_sha1:
            raise RuntimeError(
                f"live env model-XML sha1 {live[:12]}… != manifest "
                f"{self.model_xml_sha1[:12]}… for task '{self.task}' — this "
                f"renderer's assets were captured from a different env build; "
                f"refusing to render (G9).")

        addr = resolve_addresses(env)
        joint_names = list(addr.obj_qpos_slices)
        missing = [j for j in joint_names if j not in self._objects]
        if missing:
            raise RuntimeError(
                f"task '{self.task}': free joint(s) {missing} have no GS "
                f"asset (assets cover {sorted(self._objects)}) — capture and "
                f"train the missing object(s) before rendering (G1).")
        stale = [j for j in self._objects if j not in joint_names]
        if stale:
            raise RuntimeError(
                f"task '{self.task}': manifest carries object asset(s) "
                f"{stale} with no matching free joint in the live model "
                f"(joints: {joint_names}) — manifest/env mismatch (G9).")

        obj_body_ids: "OrderedDict[str, int]" = OrderedDict(
            (j, int(model.jnt_bodyid[model.joint_name2id(j)]))
            for j in joint_names)

        link_body_ids: "OrderedDict[str, int]" = OrderedDict()
        for lname in self._robot_links:
            try:
                link_body_ids[lname] = int(model.body_name2id(lname))
            except Exception as e:
                raise RuntimeError(
                    f"robot asset link '{lname}' is not a body of the live "
                    f"model — asset/env mismatch (G9).") from e

        cam_ids: "OrderedDict[str, int]" = OrderedDict()
        Ks = []
        for zkey, mj_name in self.cameras.items():
            try:
                cid = int(model.camera_name2id(mj_name))
            except Exception as e:
                raise RuntimeError(
                    f"camera '{mj_name}' (zarr key '{zkey}') not in the live "
                    f"model; cameras: {list(model.camera_names)}") from e
            cam_ids[zkey] = cid
            # G8: intrinsics from the model's fovy at the dataset resolution.
            Ks.append(cam.fovy_to_K(float(model.cam_fovy[cid]),
                                    self.resolution, self.resolution))
        Ks_t = torch.as_tensor(np.stack(Ks), dtype=torch.float32,
                               device=self.device)

        self._binding = _SceneBinding(
            model=model, obj_body_ids=obj_body_ids,
            link_body_ids=link_body_ids, cam_ids=cam_ids, Ks=Ks_t)
        return self._binding

    # ── per-frame plumbing ──────────────────────────────────────────────────

    def _set_and_forward(self, env, state_rw: np.ndarray) -> None:
        """G3: kinematics only — set the rewritten state and forward, never
        touch a MuJoCo renderer."""
        state = np.asarray(state_rw, dtype=np.float64).reshape(-1)
        model = env.sim.model
        expect = 1 + int(model.nq) + int(model.nv)
        if state.shape[0] != expect:
            raise ValueError(
                f"state_rw has length {state.shape[0]}, expected "
                f"1 + nq({model.nq}) + nv({model.nv}) = {expect} — pass the "
                f"flattened MjSimState [time, qpos, qvel], the same input as "
                f"env.regenerate_obs_from_state.")
        env.sim.set_state_from_flattened(state)
        env.sim.forward()

    def _world_parts(self, env, binding: _SceneBinding) -> List[WorldGaussians]:
        """All components posed from the forwarded sim, fixed order:
        background, objects (resolve_addresses order), robot links."""
        d = env.sim.data
        parts: List[WorldGaussians] = [self._bg_world]
        for jname, bid in binding.obj_body_ids.items():
            parts.append(self._objects[jname].posed(d.xpos[bid], d.xquat[bid]))
        for lname, bid in binding.link_body_ids.items():
            parts.append(self._robot_links[lname].posed(d.xpos[bid], d.xquat[bid]))
        return parts

    def _viewmats(self, env, binding: _SceneBinding,
                  cam_keys: Optional[List[str]] = None) -> torch.Tensor:
        """(C,4,4) OpenCV w2c from the FORWARDED sim's cam_xpos/cam_xmat + the
        measured F1 flip — eye-in-hand extrinsics thereby come from the
        rewritten state's FK, exact."""
        d = env.sim.data
        keys = list(binding.cam_ids) if cam_keys is None else cam_keys
        w2cs = np.stack([
            cam.mujoco_cam_to_w2c(d.cam_xpos[binding.cam_ids[k]],
                                  d.cam_xmat[binding.cam_ids[k]], self._flip)
            for k in keys])
        return torch.as_tensor(w2cs, dtype=torch.float32, device=self.device)

    def _rasterize(self, world: WorldGaussians, viewmats: torch.Tensor,
                   Ks: torch.Tensor):
        """ONE gsplat.rasterization call over the full concatenated set for
        all requested cameras at once (G2; C>1 is fine — square resolution is
        shared). Returns (img (C,H,W,3) float 0..1, alpha (C,H,W,1))."""
        img, alpha, _meta = gsplat.rasterization(
            means=world.means, quats=world.quats, scales=world.scales,
            opacities=world.opacities, colors=world.sh,
            viewmats=viewmats, Ks=Ks,
            width=self.resolution, height=self.resolution,
            sh_degree=world.sh_degree, render_mode="RGB")
        return img, alpha

    def _to_dataset_orientation(self, frame: np.ndarray) -> np.ndarray:
        """Apply the measured F2 orientation (gsplat output -> stored zarr)."""
        if self._flip_ud:
            frame = frame[::-1]
        return np.ascontiguousarray(frame)

    # ── public API ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def render(self, env, state_rw: np.ndarray) -> Dict[str, np.ndarray]:
        """Composite-render the rewritten state for all configured cameras.

        Returns ``{zarr_image_key: uint8 (H,W,3)}`` in DATASET orientation —
        directly writable next to oracle ``np.flip(obs, axis=0)`` frames.
        """
        binding = self._bind_scene(env)
        self._set_and_forward(env, state_rw)
        world = WorldGaussians.concat(self._world_parts(env, binding))
        viewmats = self._viewmats(env, binding)
        img, _alpha = self._rasterize(world, viewmats, binding.Ks)
        img8 = img.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        img8 = img8.cpu().numpy()
        return {zkey: self._to_dataset_orientation(img8[i])
                for i, zkey in enumerate(binding.cam_ids)}

    @torch.no_grad()
    def render_component_alpha(self, env, state_rw: np.ndarray,
                               component_name: str, cam_key: str) -> np.ndarray:
        """Alpha (H,W) float32, dataset orientation, of ONE component
        rasterized alone — for probe_gs_geometry's silhouette IoU.

        ``component_name``: 'background', a free-joint name (object), 'robot'
        (all links), or a single robot link name. ``cam_key`` is a zarr image
        key from the ctor's cameras mapping.

        NOTE: solo rasterization is a probe-only path — composite frames must
        come from :meth:`render`'s single concatenated pass (G2).
        """
        if cam_key not in self.cameras:
            raise KeyError(
                f"cam_key '{cam_key}' not in configured cameras "
                f"{list(self.cameras)}")
        binding = self._bind_scene(env)
        self._set_and_forward(env, state_rw)

        d = env.sim.data
        if component_name == "background":
            world = self._bg_world
        elif component_name in self._objects:
            bid = binding.obj_body_ids[component_name]
            world = self._objects[component_name].posed(d.xpos[bid], d.xquat[bid])
        elif component_name == "robot":
            world = WorldGaussians.concat([
                self._robot_links[l].posed(d.xpos[b], d.xquat[b])
                for l, b in binding.link_body_ids.items()])
        elif component_name in self._robot_links:
            bid = binding.link_body_ids[component_name]
            world = self._robot_links[component_name].posed(
                d.xpos[bid], d.xquat[bid])
        else:
            raise KeyError(
                f"unknown component '{component_name}'; available: "
                f"'background', 'robot', objects {list(self._objects)}, "
                f"links {list(self._robot_links)}")

        idx = list(binding.cam_ids).index(cam_key)
        viewmats = self._viewmats(env, binding, cam_keys=[cam_key])
        _img, alpha = self._rasterize(world, viewmats,
                                      binding.Ks[idx:idx + 1])
        a = alpha[0, :, :, 0].float().cpu().numpy()
        return self._to_dataset_orientation(a)
