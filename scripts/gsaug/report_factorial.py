"""Factorial report for the GS-render Phase-0 study (plan §9, M8).

Aggregates the four inputs of the A1..A6 factorial into ONE markdown table +
ONE csv:

  1. Eval logs (``eval_log.json`` written by ``scripts/eval_policy_sim.py``):
     per arm, one json PER TRAINING SEED. Schema (verified against
     ``scripts/eval_policy_sim.py`` + ``oat/env_runner/libero_runner.py``):
       - ``mean_success_rate_mean``          pooled SR over all task inits
       - ``{task_name}/mean_success_rate_mean``  per-task SR
       - ``checkpoint``, ``num_exp``; ``*_std``/``*_stderr`` iff num_exp>1
       - ``{task}/video_{seed}_{i}``         (ignored)
     Keys WITHOUT the ``_mean`` suffix (a raw runner_log dump) are accepted as
     a fallback.
  2. Oracle prerender report ``libero10_N500_se2aug.zarr.report.json`` and the
     GS twin ``..._gs.zarr.report.json`` (schema: write_report() in
     ``scripts/prerender_se2_aug.py``) — valid rates + args + (G9) any
     ``*manifest_sha1*`` / ``render_source`` fields for the provenance appendix.
  3. Photometric probe ``data/libero/probe_gs_photometric.json`` (§8.2,
     report-only): per-task partitioned PSNR/SSIM, LPIPS, contact-band PSNR.
     Parsed tolerantly: the per-task dict is flattened to dot-joined numeric
     leaves and every metric whose name contains psnr/ssim/lpips is correlated
     (Pearson + Spearman) against the per-task A3−A4 SR delta.

Contrasts printed (plan §9): A3−A1 (aug value @ oracle renders), A4−A2
(aug value @ GS renders), A3−A4 (rendering cost under augmentation), A1−A2
(pure render-domain cost) — pooled and per task, with across-seed std.

Every input is optional: missing/malformed files degrade to a note in the
report ("Notes" section), never a crash. Pure stdlib + numpy; NO sim imports.

CSV is long-format with columns
    kind, name, task, metric, value, value2, n, note
where (value, value2) mean:
    arm_pooled / arm_task      -> (mean SR, across-seed std or '')
    arm_seed                   -> (that seed's SR, '')
    contrast_pooled / _task    -> (delta of means, standard error or '')
    correlation                -> (pearson_r, spearman_rho)
SR values are fractions in [0, 1]; markdown tables show percent.

Usage:
    python scripts/gsaug/report_factorial.py \\
        --arm A1='output/*train_flow_noaug*/eval*/eval_log.json' \\
        --arm A3='output/*train_flowpolicy_aug*/eval*/eval_log.json' \\
        --arm A2='output/*train_flowpolicy_gs_noaug*/eval*/eval_log.json' \\
        --arm A4='output/*train_flowpolicy_gs_aug*/eval*/eval_log.json'
    # or: --spec arms.json   with {"A1": ["path_or_glob", ...], ...}
"""

import os

if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import csv
import datetime
import glob
import json
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Plan §9: the four pinned contrasts, in print order.
CONTRASTS: List[Tuple[str, str]] = [
    ("A3", "A1"),  # augmentation value @ oracle renderer (upper bound)
    ("A4", "A2"),  # augmentation value @ GS renderer
    ("A3", "A4"),  # rendering cost under augmentation (THE number)
    ("A1", "A2"),  # pure render-domain cost, no augmentation
]
# The correlation target (plan §8.2): per-task A3−A4 delta vs photometric metrics.
CORR_CONTRAST = ("A3", "A4")
# Photometric metric name filter (case-insensitive substring match on the
# flattened key). Covers robot/movables/background PSNR+SSIM, contact-band
# PSNR, full-frame + bbox-crop LPIPS regardless of the probe's exact nesting.
METRIC_SUBSTRINGS = ("psnr", "ssim", "lpips")
MIN_TASKS_FOR_CORR = 3


# ──────────────────────────────────────────────────────────────────────────────
# small numerics (no scipy)
# ──────────────────────────────────────────────────────────────────────────────

def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Mean and across-seed sample std (ddof=1); std is None when n < 2."""
    if not vals:
        return None, None
    v = np.asarray(vals, dtype=np.float64)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if len(v) > 1 else None
    return mean, std


def _ranks(v: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with tie handling — for Spearman."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None  # degenerate: constant column -> correlation undefined
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    return _pearson(_ranks(x), _ranks(y))


# ──────────────────────────────────────────────────────────────────────────────
# input loading (all tolerant: failures become notes, not crashes)
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: str, label: str, notes: List[str]) -> Optional[dict]:
    if not path:
        notes.append(f"{label}: no path given — skipped.")
        return None
    if not os.path.exists(path):
        notes.append(f"{label}: MISSING file {path} — dependent sections skipped.")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        notes.append(f"{label}: could not parse {path} ({e}) — skipped.")
        return None


def parse_eval_log(d: dict, path: str, notes: List[str]) -> Optional[dict]:
    """Extract pooled + per-task SR from one eval_log.json.

    Returns {'pooled': float, 'per_task': {task: float}, 'checkpoint': str,
    'num_exp': int, 'pooled_within_std': float|None} or None if unusable.
    """
    pooled = None
    per_task: Dict[str, float] = {}
    per_task_raw: Dict[str, float] = {}  # fallback keys w/o '_mean'
    for k, v in d.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if k == "mean_success_rate_mean":
            pooled = float(v)
        elif k == "mean_success_rate" and pooled is None:
            pooled = float(v)
        elif k.endswith("/mean_success_rate_mean"):
            per_task[k[: -len("/mean_success_rate_mean")]] = float(v)
        elif k.endswith("/mean_success_rate"):
            per_task_raw[k[: -len("/mean_success_rate")]] = float(v)
    for t, v in per_task_raw.items():
        per_task.setdefault(t, v)
    if pooled is None and per_task:
        # Raw runner_log dump without the pooled key: fall back to the task
        # mean (equal-weight per task, NOT per init — flagged in notes).
        pooled = float(np.mean(list(per_task.values())))
        notes.append(
            f"eval {path}: no pooled 'mean_success_rate[_mean]' key; used the "
            f"unweighted mean over {len(per_task)} task SRs instead."
        )
    if pooled is None:
        notes.append(f"eval {path}: no success-rate keys found — file ignored.")
        return None
    return {
        "pooled": pooled,
        "per_task": per_task,
        "checkpoint": d.get("checkpoint", "?"),
        "num_exp": d.get("num_exp", 1),
        "pooled_within_std": d.get("mean_success_rate_std"),
        "path": path,
    }


def collect_arms(args: argparse.Namespace,
                 notes: List[str]) -> "OrderedDict[str, List[dict]]":
    """Resolve --arm NAME=GLOB (repeatable) and/or --spec json into
    {arm_name: [parsed eval dicts]} (sorted by arm name)."""
    specs: "OrderedDict[str, List[str]]" = OrderedDict()

    for item in args.arm or []:
        assert "=" in item, (
            f"--arm expects NAME=GLOB, got {item!r} "
            "(e.g. --arm A1='output/*noaug*/eval*/eval_log.json')"
        )
        name, pattern = item.split("=", 1)
        specs.setdefault(name.strip(), []).append(pattern)

    if args.spec:
        spec = _load_json(args.spec, f"--spec {args.spec}", notes)
        if spec is not None:
            assert isinstance(spec, dict), (
                f"--spec {args.spec} must be a json object mapping "
                "arm name -> path-or-glob (or list thereof)"
            )
            for name, val in spec.items():
                patterns = [val] if isinstance(val, str) else list(val)
                specs.setdefault(str(name), []).extend(patterns)

    arms: "OrderedDict[str, List[dict]]" = OrderedDict()
    for name in sorted(specs.keys()):
        paths: List[str] = []
        for pattern in specs[name]:
            hits = sorted(glob.glob(pattern, recursive=True))
            if not hits and os.path.exists(pattern):
                hits = [pattern]
            if not hits:
                notes.append(f"arm {name}: pattern {pattern!r} matched no files.")
            paths.extend(hits)
        runs = []
        for p in paths:
            d = _load_json(p, f"arm {name} eval", notes)
            if d is None:
                continue
            parsed = parse_eval_log(d, p, notes)
            if parsed is not None:
                runs.append(parsed)
        if runs:
            arms[name] = runs
        else:
            notes.append(f"arm {name}: no usable eval jsons — arm dropped.")
    return arms


# ──────────────────────────────────────────────────────────────────────────────
# stats over arms
# ──────────────────────────────────────────────────────────────────────────────

def arm_stats(runs: List[dict]) -> dict:
    """Across-seed stats: pooled (mean, std, n) + per-task (mean, std, n)."""
    pooled_vals = [r["pooled"] for r in runs]
    tasks = sorted({t for r in runs for t in r["per_task"]})
    per_task = {}
    for t in tasks:
        vals = [r["per_task"][t] for r in runs if t in r["per_task"]]
        m, s = _mean_std(vals)
        per_task[t] = {"mean": m, "std": s, "n": len(vals)}
    m, s = _mean_std(pooled_vals)
    return {
        "pooled": {"mean": m, "std": s, "n": len(pooled_vals)},
        "per_task": per_task,
        "seed_values": pooled_vals,
        "runs": runs,
    }


def contrast_stats(sa: dict, sb: dict) -> dict:
    """delta = mean(A) − mean(B); se = sqrt(sA²/nA + sB²/nB) (None if either
    side has a single seed). Per-task over tasks present in BOTH arms."""

    def _delta(ea: dict, eb: dict) -> dict:
        delta = ea["mean"] - eb["mean"]
        se = None
        if ea["std"] is not None and eb["std"] is not None:
            se = math.sqrt(ea["std"] ** 2 / ea["n"] + eb["std"] ** 2 / eb["n"])
        return {"delta": delta, "se": se, "n_a": ea["n"], "n_b": eb["n"]}

    per_task = {
        t: _delta(sa["per_task"][t], sb["per_task"][t])
        for t in sorted(set(sa["per_task"]) & set(sb["per_task"]))
    }
    return {"pooled": _delta(sa["pooled"], sb["pooled"]), "per_task": per_task}


# ──────────────────────────────────────────────────────────────────────────────
# photometric probe (§8.2) — tolerant per-task metric extraction
# ──────────────────────────────────────────────────────────────────────────────

def _flatten_numeric(d: dict, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(_flatten_numeric(v, key))
    return out


def photometric_per_task(pj: dict, known_tasks: List[str],
                         notes: List[str]) -> Dict[str, Dict[str, float]]:
    """Return {task: {flat_metric_key: value}} from the probe json.

    Looks for a 'per_task'/'tasks' dict first; otherwise treats top-level dict
    values whose keys are known task names (or look like LIBERO task names) as
    the per-task container."""
    container = None
    for key in ("per_task", "tasks"):
        if isinstance(pj.get(key), dict) and pj[key] and all(
            isinstance(v, dict) for v in pj[key].values()
        ):
            container = pj[key]
            break
    if container is None:
        cand = {k: v for k, v in pj.items() if isinstance(v, dict)}
        if known_tasks:
            container = {k: v for k, v in cand.items() if k in set(known_tasks)}
        if not container:
            container = {k: v for k, v in cand.items() if "SCENE" in k}
    if not container:
        notes.append(
            "photometric probe: could not locate a per-task metrics dict "
            "(looked for 'per_task'/'tasks'/task-named top-level keys) — "
            "correlation section skipped."
        )
        return {}
    out = {}
    for t, d in container.items():
        flat = _flatten_numeric(d)
        flat = {
            k: v for k, v in flat.items()
            if any(s in k.lower() for s in METRIC_SUBSTRINGS)
        }
        if flat:
            out[t] = flat
    if not out:
        notes.append(
            "photometric probe: per-task dict found but no psnr/ssim/lpips "
            "numeric leaves — correlation section skipped."
        )
    return out


def correlations(delta_per_task: Dict[str, dict],
                 photo: Dict[str, Dict[str, float]],
                 notes: List[str]) -> List[dict]:
    """Pearson + Spearman of per-task SR delta vs each photometric metric."""
    common = sorted(set(delta_per_task) & set(photo))
    if len(common) < MIN_TASKS_FOR_CORR:
        notes.append(
            f"correlation: only {len(common)} task(s) present in BOTH the "
            f"{CORR_CONTRAST[0]}-{CORR_CONTRAST[1]} delta and the photometric "
            f"probe (need >= {MIN_TASKS_FOR_CORR}) — correlation skipped."
        )
        return []
    metric_keys = sorted({k for t in common for k in photo[t]})
    rows = []
    for mk in metric_keys:
        tasks_mk = [t for t in common if mk in photo[t]]
        if len(tasks_mk) < MIN_TASKS_FOR_CORR:
            continue
        x = np.asarray([photo[t][mk] for t in tasks_mk], dtype=np.float64)
        y = np.asarray([delta_per_task[t]["delta"] for t in tasks_mk],
                       dtype=np.float64)
        rows.append({
            "metric": mk,
            "n_tasks": len(tasks_mk),
            "pearson_r": _pearson(x, y),
            "spearman_rho": _spearman(x, y),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# provenance (G9)
# ──────────────────────────────────────────────────────────────────────────────

def _find_keys_recursive(d, substrings: Tuple[str, ...], prefix: str = ""
                         ) -> List[Tuple[str, object]]:
    """All (dot.path, value) pairs whose key name contains any substring —
    used to surface *manifest_sha1* / render_source wherever M6 put them.
    A matched key whose value is a FLAT dict/list of scalars (e.g. per-task
    ``gs_manifest_sha1: {task: sha}``) is expanded one level."""
    hits: List[Tuple[str, object]] = []
    if isinstance(d, dict):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if any(s in str(k).lower() for s in substrings):
                if not isinstance(v, (dict, list)):
                    hits.append((path, v))
                elif isinstance(v, dict) and all(
                    not isinstance(x, (dict, list)) for x in v.values()
                ):
                    hits.extend((f"{path}.{kk}", vv) for kk, vv in v.items())
                elif isinstance(v, list) and all(
                    not isinstance(x, (dict, list)) for x in v
                ):
                    hits.extend((f"{path}[{i}]", vv) for i, vv in enumerate(v))
            hits.extend(_find_keys_recursive(v, substrings, path))
    return hits


def report_provenance(report: Optional[dict], label: str) -> List[str]:
    """Markdown lines summarizing one prerender report (args, valid rates,
    manifest sha1s)."""
    if report is None:
        return [f"- **{label}**: not available."]
    lines = [f"- **{label}** (created {report.get('created', '?')}):"]
    args = report.get("args", {})
    if args:
        arg_str = ", ".join(f"{k}={v}" for k, v in sorted(args.items()))
        lines.append(f"  - args: `{arg_str}`")
    angles = report.get("angles_deg")
    rates = report.get("per_angle_valid_rate")
    if angles is not None and rates is not None:
        pairs = ", ".join(
            f"{a:g}°:{('%.3f' % r) if r is not None else 'n/a'}"
            for a, r in zip(angles, rates)
        )
        lines.append(f"  - per-angle valid rate: {pairs}")
        done = [r for r in rates if r is not None]
        if done:
            lines.append(f"  - overall valid rate: {float(np.mean(done)):.3f}")
    per_task = report.get("per_task", {})
    if per_task and angles is not None:
        k = len(angles)
        vt = []
        for t, td in sorted(per_task.items()):
            vpa, n_ep = td.get("valid_per_angle"), td.get("n_episodes")
            if vpa and n_ep:
                vt.append(f"{t}: {sum(vpa) / (n_ep * k):.3f}")
        if vt:
            lines.append("  - per-task valid rate: " + "; ".join(vt))
    sha_hits = _find_keys_recursive(
        report, ("manifest_sha1", "render_source", "model_xml_sha1")
    )
    if sha_hits:
        for path, v in sha_hits[:40]:
            lines.append(f"  - {path}: `{v}`")
    elif "gs" in label.lower():
        lines.append(
            "  - NOTE: no manifest_sha1 / render_source fields found in this "
            "report (G9 provenance incomplete)."
        )
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# formatting
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_sr(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{100 * mean:.1f}"
    return f"{100 * mean:.1f} ± {100 * std:.1f}"


def _fmt_delta(delta: Optional[float], se: Optional[float]) -> str:
    if delta is None:
        return "—"
    s = f"{100 * delta:+.1f}"
    if se is not None:
        s += f" ± {100 * se:.1f}"
    return s


def _task_legend(tasks: List[str]) -> "OrderedDict[str, str]":
    return OrderedDict((t, f"T{i + 1:02d}") for i, t in enumerate(sorted(tasks)))


def _md_table(header: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--arm", action="append", metavar="NAME=GLOB", default=[],
        help="repeatable; eval_log.json glob for one arm, e.g. "
             "A1='output/*noaug*/eval*/eval_log.json'",
    )
    p.add_argument(
        "--spec", default=None,
        help="json file mapping arm name -> eval json path/glob (or list); "
             "merged with --arm",
    )
    p.add_argument("--oracle_report",
                   default="data/libero/libero10_N500_se2aug.zarr.report.json",
                   help="oracle prerender report json")
    p.add_argument("--gs_report",
                   default="data/libero/libero10_N500_se2aug_gs.zarr.report.json",
                   help="GS prerender report json")
    p.add_argument("--photometric",
                   default="data/libero/probe_gs_photometric.json",
                   help="photometric probe json (§8.2)")
    p.add_argument("--out_md", default="data/libero/gs_factorial_report.md",
                   help="output markdown path")
    p.add_argument("--out_csv", default="data/libero/gs_factorial_report.csv",
                   help="output csv path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    notes: List[str] = []

    arms_raw = collect_arms(args, notes)
    if not arms_raw:
        notes.append(
            "No eval logs provided/found (--arm / --spec): arm and contrast "
            "tables are empty. Provide e.g. "
            "--arm A1='output/*noaug*/eval*/eval_log.json'."
        )
    arms = OrderedDict((name, arm_stats(runs)) for name, runs in arms_raw.items())

    oracle_report = _load_json(args.oracle_report, "oracle report", notes)
    gs_report = _load_json(args.gs_report, "gs report", notes)
    photometric = _load_json(args.photometric, "photometric probe", notes)

    # union of task names (eval logs first, prerender reports as fallback)
    tasks = sorted(
        {t for s in arms.values() for t in s["per_task"]}
        | set((oracle_report or {}).get("per_task", {}).keys())
        | set((gs_report or {}).get("per_task", {}).keys())
    )
    legend = _task_legend(tasks)

    # contrasts (plan §9 pinned list)
    contrasts: "OrderedDict[str, dict]" = OrderedDict()
    for a, b in CONTRASTS:
        if a in arms and b in arms:
            contrasts[f"{a}-{b}"] = contrast_stats(arms[a], arms[b])
        else:
            missing = [x for x in (a, b) if x not in arms]
            notes.append(
                f"contrast {a}-{b}: arm(s) {', '.join(missing)} missing — skipped."
            )

    # correlation: per-task A3−A4 delta vs photometric metrics
    corr_rows: List[dict] = []
    corr_name = f"{CORR_CONTRAST[0]}-{CORR_CONTRAST[1]}"
    if photometric is not None and corr_name in contrasts:
        photo = photometric_per_task(photometric, tasks, notes)
        if photo:
            corr_rows = correlations(
                contrasts[corr_name]["per_task"], photo, notes
            )
    elif photometric is not None:
        notes.append(
            f"correlation: photometric probe present but contrast {corr_name} "
            "unavailable — correlation skipped."
        )

    # ── markdown ────────────────────────────────────────────────────────────
    md: List[str] = []
    md.append("# GS-render factorial report (plan §9 / M8)")
    md.append("")
    md.append(f"Generated {datetime.datetime.now().isoformat()} by "
              f"`scripts/gsaug/report_factorial.py`. SR values in percent; "
              f"± is std across training seeds (contrasts: standard error of "
              f"the mean difference).")
    md.append("")

    if notes:
        md.append("## Notes / missing inputs")
        md.append("")
        md += [f"- {n}" for n in notes]
        md.append("")

    if tasks:
        md.append("## Task legend")
        md.append("")
        md += _md_table(["id", "task"],
                        [[sid, f"`{t}`"] for t, sid in legend.items()])
        md.append("")

    if arms:
        md.append("## Arms — pooled success rate (%)")
        md.append("")
        rows = []
        for name, s in arms.items():
            e = s["pooled"]
            seeds = ", ".join(f"{100 * v:.1f}" for v in s["seed_values"])
            rows.append([name, str(e["n"]), _fmt_sr(e["mean"], e["std"]), seeds])
        md += _md_table(["arm", "seeds", "SR (mean ± std)", "per-seed SR"], rows)
        md.append("")

        md.append("## Arms — per-task success rate (%)")
        md.append("")
        header = ["task"] + list(arms.keys())
        rows = []
        for t in tasks:
            row = [legend[t]]
            for s in arms.values():
                e = s["per_task"].get(t)
                row.append(_fmt_sr(e["mean"], e["std"]) if e else "—")
            rows.append(row)
        md += _md_table(header, rows)
        md.append("")

    if contrasts:
        md.append("## Contrasts — pooled (SR points)")
        md.append("")
        rows = []
        for cname, c in contrasts.items():
            e = c["pooled"]
            rows.append([cname, _fmt_delta(e["delta"], e["se"]),
                         f"{e['n_a']}/{e['n_b']}"])
        md += _md_table(["contrast", "Δ SR (± se)", "seeds (a/b)"], rows)
        md.append("")

        md.append("## Contrasts — per task (SR points)")
        md.append("")
        header = ["task"] + list(contrasts.keys())
        rows = []
        for t in tasks:
            row = [legend[t]]
            for c in contrasts.values():
                e = c["per_task"].get(t)
                row.append(_fmt_delta(e["delta"], e["se"]) if e else "—")
            rows.append(row)
        md += _md_table(header, rows)
        md.append("")

    if corr_rows:
        md.append(f"## Correlation: per-task {corr_name} ΔSR vs photometric "
                  f"metrics (§8.2)")
        md.append("")
        rows = [
            [r["metric"],
             f"{r['pearson_r']:+.3f}" if r["pearson_r"] is not None else "n/a",
             f"{r['spearman_rho']:+.3f}" if r["spearman_rho"] is not None else "n/a",
             str(r["n_tasks"])]
            for r in corr_rows
        ]
        md += _md_table(["metric", "pearson r", "spearman ρ", "n tasks"], rows)
        md.append("")

    # provenance appendix (G9)
    md.append("## Provenance appendix")
    md.append("")
    md.append(f"- report args: `{vars(args)}`")
    md += report_provenance(oracle_report, f"oracle report {args.oracle_report}")
    md += report_provenance(gs_report, f"gs report {args.gs_report}")
    if photometric is not None:
        lines = [f"- **photometric probe {args.photometric}**"]
        for key in ("probe", "date", "created", "args", "n_frames_per_task"):
            if key in photometric and not isinstance(photometric[key], dict):
                lines[0] += f" ({key}={photometric[key]})"
        sha_hits = _find_keys_recursive(
            photometric, ("manifest_sha1", "render_source"))
        for path, v in sha_hits[:20]:
            lines.append(f"  - {path}: `{v}`")
        md += lines
    else:
        md.append(f"- **photometric probe {args.photometric}**: not available.")
    for name, s in arms.items():
        md.append(f"- **arm {name}** ({s['pooled']['n']} seed(s)):")
        for r in s["runs"]:
            md.append(
                f"  - `{r['path']}` SR={100 * r['pooled']:.1f} "
                f"num_exp={r['num_exp']} ckpt=`{r['checkpoint']}`"
            )
    md.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(md) + "\n")

    # ── csv (long format; column semantics in the module docstring) ─────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "name", "task", "metric", "value", "value2", "n",
                    "note"])

        def _num(x):
            return "" if x is None else f"{x:.6f}"

        for name, s in arms.items():
            e = s["pooled"]
            w.writerow(["arm_pooled", name, "", "success_rate",
                        _num(e["mean"]), _num(e["std"]), e["n"], ""])
            for r in s["runs"]:
                w.writerow(["arm_seed", name, "", "success_rate",
                            _num(r["pooled"]), "", r["num_exp"], r["path"]])
            for t, e in s["per_task"].items():
                w.writerow(["arm_task", name, t, "success_rate",
                            _num(e["mean"]), _num(e["std"]), e["n"], ""])
        for cname, c in contrasts.items():
            e = c["pooled"]
            w.writerow(["contrast_pooled", cname, "", "success_rate_delta",
                        _num(e["delta"]), _num(e["se"]),
                        min(e["n_a"], e["n_b"]),
                        f"n_a={e['n_a']} n_b={e['n_b']}"])
            for t, e in c["per_task"].items():
                w.writerow(["contrast_task", cname, t, "success_rate_delta",
                            _num(e["delta"]), _num(e["se"]),
                            min(e["n_a"], e["n_b"]),
                            f"n_a={e['n_a']} n_b={e['n_b']}"])
        for r in corr_rows:
            w.writerow(["correlation", corr_name, "", r["metric"],
                        _num(r["pearson_r"]), _num(r["spearman_rho"]),
                        r["n_tasks"], "value=pearson_r value2=spearman_rho"])
        for n in notes:
            w.writerow(["note", "", "", "", "", "", "", n])

    # ── stdout summary ──────────────────────────────────────────────────────
    print(f"[report_factorial] markdown -> {args.out_md}")
    print(f"[report_factorial] csv      -> {args.out_csv}")
    if arms:
        print("[report_factorial] pooled SR (%):")
        for name, s in arms.items():
            e = s["pooled"]
            print(f"  {name}: {_fmt_sr(e['mean'], e['std'])}  (n={e['n']})")
    if contrasts:
        print("[report_factorial] contrasts (SR points):")
        for cname, c in contrasts.items():
            e = c["pooled"]
            print(f"  {cname}: {_fmt_delta(e['delta'], e['se'])}")
    if notes:
        print(f"[report_factorial] {len(notes)} note(s) — see the report's "
              f"'Notes / missing inputs' section:")
        for n in notes:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
