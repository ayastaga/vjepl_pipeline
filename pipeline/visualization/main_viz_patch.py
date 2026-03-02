"""
pipeline/main.py  — VISUALIZATION-AWARE VERSION
================================================

Modifications to your existing main.py to integrate the modular visualization
subsystem. The changes are marked with  # VIZ-ADDED  comments so you can
grep for them and apply them to your existing file.

Key changes:
1. argparse: --visualize flag (all / none / comma-separated stage list)
2. Config helper: resolve which stages to visualize
3. After each pipeline stage: optional viz call with metrics dict
4. At pipeline end: summary of all saved plot paths
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# ── VIZ-ADDED: argument parsing ───────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V-JEPA Robotics Dataset Pipeline"
    )
    parser.add_argument(
        "--visualize",
        default="all",
        metavar="STAGES",
        help=(
            "Comma-separated list of stages to visualize, or 'all' / 'none'. "
            "Example: --visualize quality,eqs,rejection  "
            "Example: --visualize all  "
            "Example: --visualize none"
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config YAML file.",
    )
    return parser


# ── VIZ-ADDED: resolve visualization stage list ───────────────────────────────

_ALL_VIZ_STAGES = [
    "drift", "alignment", "window", "quality",
    "eqs", "dynamics", "episode", "diversity", "rejection",
]


def resolve_viz_stages(visualize_arg: str, config: Any) -> set[str]:
    """
    Returns the set of stage names that should produce visualizations
    for this run.

    Resolution order:
    1. --visualize none   → empty set (no visuals)
    2. --visualize all    → all stages enabled in config.visualization.stages
    3. --visualize a,b,c  → exactly those stages (still respects config.enabled)
    """
    if not getattr(getattr(config, "visualization", None), "enabled", True):
        return set()

    vis_arg = visualize_arg.strip().lower()

    if vis_arg == "none":
        return set()

    config_stages = getattr(getattr(config, "visualization", None), "stages", None)

    if vis_arg == "all":
        if config_stages is None:
            return set(_ALL_VIZ_STAGES)
        return {s for s in _ALL_VIZ_STAGES
                if getattr(config_stages, s, True)}

    # Explicit comma-separated list
    requested = {s.strip() for s in vis_arg.split(",")}
    unknown   = requested - set(_ALL_VIZ_STAGES)
    if unknown:
        print(f"[WARNING] Unknown visualization stages: {unknown}. Ignoring.")
        requested -= unknown
    return requested


# ── VIZ-ADDED: visualization call helper ─────────────────────────────────────

def maybe_visualize(
    stage: str,
    metrics: dict,
    viz_stages: set[str],
    config: Any,
) -> list[Path]:
    """
    Call this after each pipeline stage to optionally produce visualizations.

    Args:
        stage:      Stage name string, e.g. "quality"
        metrics:    The structured metrics dict returned by the stage
        viz_stages: Set of enabled stage names (from resolve_viz_stages)
        config:     Full config object

    Returns:
        List of Path objects for saved plots, or [] if skipped.

    Usage in orchestrator:
        quality_metrics = quality_flagger.run(windows)
        saved_plots = maybe_visualize("quality", quality_metrics, viz_stages, config)
    """
    if stage not in viz_stages:
        return []

    if not metrics:
        print(f"  [viz/{stage}] Skipping — metrics dict is empty.")
        return []

    try:
        from pipeline.visualization import get_visualizer
        output_root = Path(
            getattr(getattr(config, "visualization", None), "output_dir",
                    "data_processed/plots")
        )
        viz   = get_visualizer(stage, output_root, config)
        saved = viz.plot(metrics)
        if saved:
            print(f"  [viz/{stage}] Saved {len(saved)} plot(s) → {output_root / stage}/")
        return saved
    except Exception as exc:
        # Visualization must never crash the pipeline
        print(f"  [viz/{stage}] WARNING: visualization failed ({exc}). Continuing.")
        return []


# ── EXAMPLE: patched orchestrator (apply these changes to your existing main.py) ──

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Skeleton showing where to insert visualization calls in your existing
    pipeline orchestrator. Replace the stub calls with your actual stage calls.

    IMPORTANT: Each pipeline stage should return BOTH its primary output AND
    a metrics dict. The metrics dict is a lightweight structured summary —
    not the raw tensors. Example:

        def run_quality_flagger(windows, config):
            # ... existing logic ...
            metrics = {
                "flag_counts":      {flag: count, ...},
                "bitmask_values":   np.array([...]),
                "n_windows_total":  len(windows),
                "flag_types":       {"MISSING_V": "hard", ...},
                ...
            }
            return flagged_windows, metrics
    """
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)

    # Convert dict to namespace-style object for attribute access
    config = _dict_to_namespace(raw_cfg)

    # VIZ-ADDED: Determine which stages produce visualizations
    viz_stages = resolve_viz_stages(args.visualize, config)
    if viz_stages:
        print(f"[pipeline] Visualization enabled for stages: {sorted(viz_stages)}")
    else:
        print("[pipeline] Visualization disabled for this run.")

    all_saved_plots: list[Path] = []
    t0 = time.perf_counter()

    # ── Stage 0: Log generation ───────────────────────────────────────────────
    print("\n[Stage 0] Generating synthetic logs...")
    from pipeline.log_generator import LogGenerator
    generator = LogGenerator(config)
    episodes  = generator.generate()
    # (no visualization for stage 0 — synthetic data only)

    # ── Stage 1: Validation ──────────────────────────────────────────────────
    print("\n[Stage 1] Validating episodes...")
    from pipeline.validator import EpisodeValidator
    validator      = EpisodeValidator(config)
    valid_episodes = validator.validate(episodes)

    # ── Stages 2-3: Drift correction + alignment ─────────────────────────────
    print("\n[Stage 2-3] Drift correction & alignment...")
    from pipeline.aligner import DriftCorrector, TimestampAligner
    drift_corrector = DriftCorrector(config)
    aligner         = TimestampAligner(config)

    aligned_episodes = []
    drift_metrics_all   = []
    alignment_metrics_all = []

    for ep in valid_episodes:
        corrected, drift_metrics     = drift_corrector.correct(ep)
        aligned, alignment_metrics   = aligner.align(corrected)
        aligned_episodes.append(aligned)
        drift_metrics_all.append(drift_metrics)
        alignment_metrics_all.append(alignment_metrics)

    # Aggregate per-episode metrics into dataset-level metrics for visualization
    drift_agg     = _aggregate_metrics(drift_metrics_all)
    alignment_agg = _aggregate_metrics(alignment_metrics_all)

    all_saved_plots += maybe_visualize("drift",     drift_agg,     viz_stages, config)
    all_saved_plots += maybe_visualize("alignment", alignment_agg, viz_stages, config)

    # ── Stage 4: Window sampling ──────────────────────────────────────────────
    print("\n[Stage 4] Sampling windows...")
    from pipeline.sampler import WindowSampler
    sampler         = WindowSampler(config)
    windows, window_metrics = sampler.sample(aligned_episodes)
    all_saved_plots += maybe_visualize("window", window_metrics, viz_stages, config)

    # ── Stage 5: Quality flagging ─────────────────────────────────────────────
    print("\n[Stage 5] Detecting quality flags...")
    from pipeline.quality import QualityFlagger
    flagger = QualityFlagger(config)
    flagged_windows, quality_metrics = flagger.flag(windows)
    all_saved_plots += maybe_visualize("quality", quality_metrics, viz_stages, config)

    # ── Stage 6: EQS scoring ──────────────────────────────────────────────────
    print("\n[Stage 6] Computing Example Quality Scores...")
    from pipeline.scorer import ExampleScorer
    scorer = ExampleScorer(config)
    scored_windows, eqs_metrics = scorer.score(flagged_windows)
    all_saved_plots += maybe_visualize("eqs", eqs_metrics, viz_stages, config)

    # ── Stage 7: Export ───────────────────────────────────────────────────────
    print("\n[Stage 7] Exporting dataset...")
    from pipeline.exporter import DatasetExporter
    exporter = DatasetExporter(config)
    manifest, export_metrics = exporter.export(scored_windows)

    # Post-export analytics (use manifest to build remaining metrics)
    dynamics_metrics  = _build_dynamics_metrics(scored_windows, config)
    episode_metrics   = _build_episode_metrics(scored_windows, config)
    rejection_metrics = _build_rejection_metrics(export_metrics, config)

    all_saved_plots += maybe_visualize("dynamics",  dynamics_metrics,  viz_stages, config)
    all_saved_plots += maybe_visualize("episode",   episode_metrics,   viz_stages, config)
    all_saved_plots += maybe_visualize("rejection", rejection_metrics, viz_stages, config)

    # Diversity requires pre-computed embeddings (skip unless available)
    if "diversity" in viz_stages:
        diversity_metrics = _build_diversity_metrics(export_metrics, config)
        if diversity_metrics:
            all_saved_plots += maybe_visualize("diversity", diversity_metrics,
                                               viz_stages, config)
        else:
            print("  [viz/diversity] Skipped — no embeddings available.")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print(f"\n{'─'*60}")
    print(f"[pipeline] Complete in {elapsed:.1f}s")
    print(f"[pipeline] Examples: {len([w for w in scored_windows if w.get('accepted', False)])} accepted")
    if all_saved_plots:
        output_root = getattr(getattr(config, "visualization", None),
                              "output_dir", "data_processed/plots")
        print(f"[pipeline] {len(all_saved_plots)} plots saved to {output_root}/")
    print(f"{'─'*60}")


# ── VIZ-ADDED: metrics aggregation helpers ────────────────────────────────────

def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    """
    Merge a list of per-episode metric dicts into a single dataset-level dict.
    Numpy arrays are concatenated; scalars are collected into lists.
    This is a general-purpose aggregator — stage-specific aggregation should
    live in the stage module itself for production use.
    """
    if not metrics_list:
        return {}
    import numpy as np
    result: dict = {}
    for key in metrics_list[0]:
        values = [m[key] for m in metrics_list if key in m]
        if not values:
            continue
        sample = values[0]
        if isinstance(sample, np.ndarray):
            result[key] = np.concatenate([v.flatten() for v in values])
        elif isinstance(sample, (int, float)):
            result[key] = float(sum(values) / len(values))
        elif isinstance(sample, list):
            result[key] = [item for v in values for item in v]
        else:
            result[key] = values[-1]  # keep last for string/dict types
    return result


def _build_dynamics_metrics(scored_windows: list, config: Any) -> dict:
    """
    Build dynamics metrics dict from scored windows.
    Each window is expected to have 'ctx_actions' key (np.ndarray shape (T, 6)).
    """
    import numpy as np
    action_mags, action_dirs = [], []
    joint_vels = []

    for w in scored_windows:
        actions = w.get("ctx_actions")
        if actions is None:
            continue
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] < 2:
            continue
        mags = np.linalg.norm(actions, axis=1)
        action_mags.append(mags.mean())
        if actions.shape[1] >= 2:
            action_dirs.append(float(np.arctan2(actions[:, 1].mean(), actions[:, 0].mean())))
        diff = np.diff(actions, axis=0)
        joint_vels.append(np.sqrt((diff ** 2).mean(axis=0)))

    if not action_mags:
        return {}

    return {
        "action_magnitudes":    np.array(action_mags),
        "action_directions":    np.array(action_dirs) if action_dirs else np.array([]),
        "joint_velocities":     np.array(joint_vels) if joint_vels else np.zeros((1, 6)),
        "joint_names":          ["J1", "J2", "J3", "J4", "J5", "J6"],
        "failure_modes":        [],
    }


def _build_episode_metrics(scored_windows: list, config: Any) -> dict:
    """
    Build episode metrics from per-window data.
    Groups windows by episode_id and computes per-episode aggregates.
    """
    import numpy as np
    from collections import defaultdict

    ep_windows = defaultdict(list)
    for w in scored_windows:
        ep_windows[w.get("episode_id", "unknown")].append(w)

    durations, motions, stall_ratios = [], [], []
    quality_timeline = {}

    for ep_id, ws in ep_windows.items():
        anchor_times = [w.get("anchor_time", 0) for w in ws]
        if len(anchor_times) > 1:
            durations.append(max(anchor_times) - min(anchor_times) + 3.0)  # +3 for window
        motions.append(sum(w.get("stall_score", 0.0) for w in ws))
        stall_count = sum(1 for w in ws if "STALL" in str(w.get("quality_flags", "")))
        stall_ratios.append(stall_count / max(len(ws), 1))
        eqs_timeline = [w.get("example_quality_score", 1.0) for w in ws]
        quality_timeline[str(ep_id)] = np.array(eqs_timeline)

    return {
        "episode_durations":  np.array(durations) if durations else np.array([]),
        "motion_energy":      np.array(motions),
        "stall_ratios":       np.array(stall_ratios),
        "quality_timeline":   quality_timeline,
        "failure_modes":      [],
    }


def _build_rejection_metrics(export_metrics: dict, config: Any) -> dict:
    """
    Build rejection diagnostics from exporter output.
    export_metrics is expected to contain 'accepted' and 'rejected' window lists.
    """
    import numpy as np
    acc = export_metrics.get("accepted_windows", [])
    rej = export_metrics.get("rejected_windows", [])
    threshold = getattr(getattr(config, "quality", None), "min_quality_score", 0.6)

    rejection_reasons: dict = {}
    hard_flag_breakdown: dict = {}
    for w in rej:
        flags = str(w.get("quality_flags", "")).split(",")
        for f in flags:
            f = f.strip()
            if f:
                rejection_reasons[f] = rejection_reasons.get(f, 0) + 1
                if f in {"MISSING_V", "DROP_P"}:
                    hard_flag_breakdown[f] = hard_flag_breakdown.get(f, 0) + 1

    eqs_acc = np.array([w.get("example_quality_score", 1.0) for w in acc])
    eqs_rej = np.array([w.get("example_quality_score", 0.0) for w in rej])

    return {
        "rejection_reasons":   rejection_reasons,
        "eqs_accepted":        eqs_acc,
        "eqs_rejected":        eqs_rej,
        "hard_flag_breakdown": hard_flag_breakdown,
        "acceptance_threshold": threshold,
        "failure_modes":       [],
    }


def _build_diversity_metrics(export_metrics: dict, config: Any) -> dict:
    """
    Build diversity metrics if embeddings are available.
    In production, embeddings would come from a pre-trained encoder pass.
    Returns empty dict if no embeddings are available (safe skip).
    """
    embeddings = export_metrics.get("embeddings", None)
    if embeddings is None:
        return {}
    import numpy as np
    return {
        "embeddings":           np.asarray(embeddings),
        "episode_labels":       np.asarray(export_metrics.get("episode_labels", [])),
        "env_drift_timeline":   np.asarray(export_metrics.get("env_drift_timeline", [])),
        "failure_modes":        [],
    }


# ── Utility ───────────────────────────────────────────────────────────────────

def _dict_to_namespace(d: dict) -> Any:
    """Recursively convert a dict to a SimpleNamespace for attribute access."""
    from types import SimpleNamespace
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    return d


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_arg_parser()
    args   = parser.parse_args()
    run_pipeline(args)