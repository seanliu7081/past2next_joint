"""Stage 2-3 GS asset training driver (plan §5.2 / §6.4, M3-M4): trains the
background, per-object and robot Gaussian assets for each captured task and
writes the per-task manifest.json (G9 provenance).

Reads capture directories produced by scripts/gsaug/capture_assets.py under
``<assets_root>/<task>/captures/{background, objects/<name>, robot}`` and
writes ``<assets_root>/<task>/assets/{background.pt, objects/<joint>.pt,
robot.pt}`` — object assets are keyed by the movable FREE-JOINT name recorded
in the capture transforms (the name compose.py looks assets up by).

The manifest records asset content sha1s, metrics, capture/train args, the
task model-XML sha1, pinned versions, and the plan's provisional metric floors
as DATA (gross-failure catches, recalibrated at the M6 dry run — plan §11).
Metrics under a floor WARN, never fail.

Usage:
    PATH=<env-bin>:/usr/local/cuda/bin:$PATH python scripts/gsaug/train_gs_assets.py \
        --task LIVING_ROOM_SCENE2_... --component all
"""

import os

# No sim imports here (training is renderer-only, G3 does not apply), so no
# MUJOCO_GL needed — but repo-root chdir IS (script pattern, capture_assets.py).
if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import datetime
import glob
import hashlib
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import gsplat

from oat.gsaug.articulated import fit_robot
from oat.gsaug.capture import CaptureBundle
from oat.gsaug.trainer import fit_static

COMPONENTS = ("background", "objects", "robot")

# Provisional acceptance floors (plan §5.2 / §6.4) — recorded in the manifest
# as DATA, not enforced in code: they are gross-failure catches, recalibrated
# after the one-task M6 dry run (plan §11). Robot EEF floor is 2 px at the
# 128² dataset resolution; capture-resolution medians are scaled before the
# comparison.
PROVISIONAL_FLOORS = {
    "background": {"psnr": 29.0},
    "objects": {"psnr": 30.0, "silhouette_iou": 0.95},
    "robot": {"psnr": 27.0, "silhouette_iou_per_link": 0.85,
              "eef_median_px": 2.0, "eef_px_resolution": 128},
    "note": ("provisional floors (plan §5.2/§6.4): gross-failure catches, "
             "recalibrated at the M6 dry run (plan §11); WARN-only"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", type=str, default="all",
                        help="task name (a dir under --assets_root) or 'all'")
    parser.add_argument("--component", type=str, default="all",
                        choices=list(COMPONENTS) + ["all"])
    parser.add_argument("--assets_root", type=str,
                        default="data/libero/gs_assets")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--iters_background", type=int, default=7000)
    parser.add_argument("--iters_object", type=int, default=5000)
    parser.add_argument("--iters_robot", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def discover_tasks(assets_root: str, task: str) -> List[str]:
    if task != "all":
        d = os.path.join(assets_root, task, "captures")
        assert os.path.isdir(d), (
            f"no captures for task {task!r} under {assets_root!r} "
            f"(expected {d}; run scripts/gsaug/capture_assets.py first)")
        return [task]
    tasks = sorted(
        name for name in os.listdir(assets_root)
        if os.path.isdir(os.path.join(assets_root, name, "captures")))
    assert tasks, f"no task capture dirs under {assets_root!r}"
    return tasks


def warn(msg: str) -> None:
    print(f"[train_gs_assets] WARNING: {msg}")


def check_floors(task: str, component_key: str, label: str,
                 metrics: Dict, capture_size: int) -> None:
    """WARN (never fail) on metrics under the provisional floors."""
    floors = PROVISIONAL_FLOORS[component_key]
    tag = f"{task}/{label}"
    psnr = metrics.get("psnr_component", metrics.get("psnr_robot"))
    if psnr is not None and psnr < floors["psnr"]:
        warn(f"{tag}: PSNR {psnr:.2f} < provisional floor {floors['psnr']}")
    if component_key == "objects":
        iou = metrics.get("silhouette_iou")
        if iou is not None and iou < floors["silhouette_iou"]:
            warn(f"{tag}: silhouette IoU {iou:.3f} < provisional floor "
                 f"{floors['silhouette_iou']}")
    if component_key == "robot":
        for link, iou in (metrics.get("silhouette_iou_per_link") or {}).items():
            if iou is not None and iou < floors["silhouette_iou_per_link"]:
                warn(f"{tag}: link '{link}' silhouette IoU {iou:.3f} < "
                     f"provisional floor {floors['silhouette_iou_per_link']}")
        eef = metrics.get("eef_median_px")
        if eef is not None:
            scaled = eef * floors["eef_px_resolution"] / float(capture_size)
            if scaled > floors["eef_median_px"]:
                warn(f"{tag}: EEF projection error {eef:.2f} px @{capture_size}² "
                     f"(≈{scaled:.2f} px @{floors['eef_px_resolution']}²) > "
                     f"provisional floor {floors['eef_median_px']} px")


def object_capture_dirs(task_dir: str) -> List[str]:
    dirs = sorted(os.path.dirname(p) for p in glob.glob(
        os.path.join(task_dir, "captures", "objects", "*", "transforms.json")))
    return dirs


def manifest_sha1_of(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha1"}
    return hashlib.sha1(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def train_task(task: str, components: List[str],
               args: argparse.Namespace) -> List[Tuple[str, Dict]]:
    """Train the requested components for one task and update its manifest.

    The manifest read-modify-write ends in an atomic tmp + ``os.replace`` (the
    oat/equi/normalization.py save_spec pattern), so a crash never leaves a
    torn manifest.json. NOTE: concurrent per-component runs of the SAME task
    remain last-writer-wins on distinct entries (each run rewrites the whole
    file from its own read).
    """
    task_dir = os.path.join(args.assets_root, task)
    cap_dir = os.path.join(task_dir, "captures")
    print(f"[train_gs_assets] === {task} ({', '.join(components)}) ===")

    # (component_key, label, asset_rel_path, asset, metrics, bundle)
    trained: List[tuple] = []

    if "background" in components:
        d = os.path.join(cap_dir, "background")
        if os.path.isdir(d):
            bundle = CaptureBundle.load(d)
            rel = "assets/background.pt"
            asset, metrics = fit_static(
                bundle, "background", os.path.join(task_dir, rel),
                iters=args.iters_background, device=args.device,
                seed=args.seed)
            trained.append(("background", "background", rel, asset, metrics,
                            bundle))
        elif components != ["background"]:
            warn(f"{task}: no background capture at {d} — skipped")
        else:
            raise FileNotFoundError(f"{task}: no background capture at {d}")

    if "objects" in components:
        dirs = object_capture_dirs(task_dir)
        if not dirs and components == ["objects"]:
            raise FileNotFoundError(
                f"{task}: no object captures under {cap_dir}/objects/")
        if not dirs:
            warn(f"{task}: no object captures under {cap_dir}/objects/ — skipped")
        for d in dirs:
            bundle = CaptureBundle.load(d)
            joint = bundle.transforms.get("joint_name")
            assert joint, (
                f"{d}: transforms.json lacks 'joint_name' — object assets are "
                f"keyed by free-joint name (compose contract)")
            rel = os.path.join("assets", "objects", f"{joint}.pt")
            asset, metrics = fit_static(
                bundle, "object", os.path.join(task_dir, rel),
                iters=args.iters_object, device=args.device, seed=args.seed)
            trained.append(("objects", joint, rel, asset, metrics, bundle))

    if "robot" in components:
        d = os.path.join(cap_dir, "robot")
        if os.path.isdir(d):
            bundle = CaptureBundle.load(d)
            rel = "assets/robot.pt"
            asset, metrics = fit_robot(
                bundle, os.path.join(task_dir, rel),
                iters=args.iters_robot, device=args.device, seed=args.seed)
            trained.append(("robot", "robot", rel, asset, metrics, bundle))
        elif components != ["robot"]:
            warn(f"{task}: no robot capture at {d} — skipped")
        else:
            raise FileNotFoundError(f"{task}: no robot capture at {d}")

    assert trained, f"{task}: nothing to train for components {components}"

    # ── manifest (G9) ───────────────────────────────────────────────────────
    # the manifest task is the CAPTURED task name (compose asserts asset meta
    # task == manifest task); the directory name should agree.
    task_names = {b.task for *_x, b in trained}
    assert len(task_names) == 1, (
        f"{task}: capture bundles record different task names "
        f"{sorted(task_names)} — mixed capture directories")
    manifest_task = task_names.pop()
    if manifest_task != task:
        warn(f"task dir {task!r} != captured task name {manifest_task!r}; "
             f"manifest records the captured name")
    hashes = {b.model_xml_sha1 for *_x, b in trained}
    assert len(hashes) == 1, (
        f"{task}: capture bundles carry different model_xml_sha1 values "
        f"{sorted(hashes)} — captures come from different env builds (G9); "
        f"re-run capture_assets.py for the whole task")
    model_sha1 = hashes.pop()

    manifest_path = os.path.join(task_dir, "manifest.json")
    manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("model_xml_sha1") not in (None, model_sha1):
            full_retrain = ("background" in {t[0] for t in trained}
                            and "robot" in {t[0] for t in trained}
                            and "objects" in {t[0] for t in trained})
            if not full_retrain:
                raise RuntimeError(
                    f"{task}: existing manifest model_xml_sha1 "
                    f"{manifest['model_xml_sha1'][:12]}… != capture "
                    f"{model_sha1[:12]}… — a partial retrain would mix assets "
                    f"from different env builds (G9). Retrain all components "
                    f"(--component all) or fix the captures.")
            warn(f"{task}: model_xml_sha1 changed; manifest fully replaced")
            manifest = {}

    manifest["task"] = manifest_task
    manifest["model_xml_sha1"] = model_sha1
    manifest["versions"] = {"gsplat": str(gsplat.__version__),
                            "torch": str(torch.__version__)}
    manifest["thresholds"] = PROVISIONAL_FLOORS
    manifest["date"] = datetime.datetime.now().isoformat()
    assets = manifest.setdefault("assets", {})
    assets.setdefault("objects", {})
    capture_args = manifest.setdefault("capture_args", {})
    train_args = manifest.setdefault("train_args", {})
    train_args["cli"] = {k: getattr(args, k) for k in
                         ("component", "device", "iters_background",
                          "iters_object", "iters_robot", "seed")}

    for comp_key, label, rel, asset, metrics, bundle in trained:
        entry = {"path": rel, "sha1": asset.sha1(), "metrics": metrics}
        if comp_key == "objects":
            assets["objects"][label] = entry
            capture_args.setdefault("objects", {})[label] = \
                bundle.transforms.get("capture_args")
            train_args.setdefault("objects", {})[label] = \
                asset.meta.get("train_args")
        else:
            assets[comp_key] = entry
            capture_args[comp_key] = bundle.transforms.get("capture_args")
            train_args[comp_key] = asset.meta.get("train_args")

    manifest["manifest_sha1"] = manifest_sha1_of(manifest)
    # atomic replace (save_spec pattern): never leave a torn manifest.json
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp_path, manifest_path)
    print(f"[train_gs_assets] {task}: manifest -> {manifest_path} "
          f"(sha1 {manifest['manifest_sha1'][:12]}…)")

    # ── metrics table + floor warnings ──────────────────────────────────────
    print(f"\n  {'component':<40} {'n_gauss':>8} {'PSNR':>7} {'IoU':>7} "
          f"{'EEF px':>7}")
    rows = []
    for comp_key, label, _rel, _asset, metrics, bundle in trained:
        psnr = metrics.get("psnr_component", metrics.get("psnr_robot"))
        iou = metrics.get("silhouette_iou",
                          metrics.get("silhouette_iou_mean"))
        eef = metrics.get("eef_median_px")
        name = label if comp_key != "objects" else f"objects/{label}"
        print(f"  {name:<40} {metrics['n_gaussians']:>8} "
              f"{psnr:>7.2f} "
              f"{(f'{iou:.3f}' if iou is not None else '-'):>7} "
              f"{(f'{eef:.2f}' if eef is not None else '-'):>7}")
        rows.append((comp_key, name, metrics, bundle.image_size))
    print()
    for comp_key, name, metrics, size in rows:
        check_floors(task, comp_key, name, metrics, size)
    return [(name, metrics) for _c, name, metrics, _s in rows]


def main() -> None:
    args = parse_args()
    components = (list(COMPONENTS) if args.component == "all"
                  else [args.component])
    tasks = discover_tasks(args.assets_root, args.task)
    for task in tasks:
        train_task(task, components, args)
    print(f"[train_gs_assets] done: {len(tasks)} task(s), "
          f"components {components}")


if __name__ == "__main__":
    main()
