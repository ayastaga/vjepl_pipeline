"""
pipeline/visualization/episode_viz.py

Episode-Level Metrics Visualization
=====================================
Produces plots to: data_processed/plots/episode/

Metrics dict contract:
    episode_durations   : np.ndarray  — duration per episode (s)
    motion_energy       : np.ndarray  — cumulative joint delta per episode
    quality_timeline    : dict[str, np.ndarray]  — episode_id → EQS over time
    stall_ratios        : np.ndarray  — fraction of stalled frames per episode
    episode_ids         : list[str]
    failure_modes       : list[str]

CLI:
    python -m pipeline.visualization.episode_viz --from-manifest data_processed/manifest.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.visualization.base import (
    BaseStageVisualizer,
    COLOR_ACCEPTED, COLOR_INFO, COLOR_REJECTED, COLOR_WARN,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_TITLE,
    annotate_failure, make_figure, plot_histogram,
)

_RNG_SEED = 42


class StageVisualizer(BaseStageVisualizer):
    STAGE_NAME = "episode"

    def plot_primary_diagnostics(self, metrics: dict) -> list[Path]:
        paths = []
        durations = np.asarray(metrics.get("episode_durations", []))
        if not durations.size:
            return paths

        fig, axes = make_figure(1, 2, title="Episode Metrics — Duration & Motion Energy")
        ax_dur, ax_energy = axes

        plot_histogram(ax_dur, durations, xlabel="Duration (s)",
                       title="Episode Duration Distribution", bins=30, color=COLOR_INFO)
        ax_dur.axvline(np.median(durations), color=COLOR_WARN, linestyle="--",
                       linewidth=1.2, label=f"median={np.median(durations):.1f}s")
        ax_dur.legend(fontsize=8)
        if durations.mean() < 5.0:
            annotate_failure(ax_dur, "Many short/trivial episodes")

        motion = np.asarray(metrics.get("motion_energy", []))
        if motion.size:
            plot_histogram(ax_energy, motion, xlabel="Cumulative joint delta (rad)",
                           title="Motion Energy Per Episode", bins=30, color=COLOR_WARN)
            if motion.mean() < 0.3:
                annotate_failure(ax_energy, "Low motion energy — static episodes")
        else:
            ax_energy.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "01_duration_energy.png"))
        return paths

    def plot_distributions(self, metrics: dict) -> list[Path]:
        paths = []
        stall_ratios    = np.asarray(metrics.get("stall_ratios", []))
        quality_timeline = metrics.get("quality_timeline", {})

        fig, axes = make_figure(1, 2, title="Episode Metrics — Quality Timeline & Stall Ratio")
        ax_timeline, ax_stall = axes

        # Quality timeline: show a few episodes
        if quality_timeline:
            sample_eps = list(quality_timeline.keys())[:6]
            cmap = plt.get_cmap("tab10")
            for i, ep in enumerate(sample_eps):
                eqs_t = np.asarray(quality_timeline[ep])
                t     = np.linspace(0, 1, len(eqs_t))
                ax_timeline.plot(t, eqs_t, linewidth=0.9, alpha=0.7,
                                 color=cmap(i), label=ep)
            ax_timeline.axhline(0.6, color=COLOR_WARN, linestyle="--",
                                linewidth=1.0, label="acceptance threshold")
            ax_timeline.set_xlabel("Normalized episode time", fontsize=FONT_SIZE_LABEL)
            ax_timeline.set_ylabel("EQS", fontsize=FONT_SIZE_LABEL)
            ax_timeline.set_title("Quality Score Timeline Per Episode", fontsize=FONT_SIZE_TITLE)
            ax_timeline.legend(fontsize=7, ncol=2)
            ax_timeline.tick_params(labelsize=FONT_SIZE_TICK)
        else:
            ax_timeline.set_visible(False)

        if stall_ratios.size:
            plot_histogram(ax_stall, stall_ratios, xlabel="Stall ratio",
                           title="Stall Ratio Distribution\n(fraction of frames flagged STALL)",
                           bins=30, color=COLOR_REJECTED,
                           threshold=0.3, threshold_label="30% stall threshold")
            if stall_ratios.mean() > 0.2:
                annotate_failure(ax_stall, "High average stall ratio")
        else:
            ax_stall.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "02_quality_stall.png"))
        return paths

    def plot_failure_modes(self, metrics: dict) -> list[Path]:
        paths = []
        durations     = np.asarray(metrics.get("episode_durations", []))
        motion        = np.asarray(metrics.get("motion_energy", []))
        stall_ratios  = np.asarray(metrics.get("stall_ratios", []))
        detected      = metrics.get("failure_modes", [])

        fig, axes = make_figure(1, 3, title="Episode — Failure Mode Diagnostics")
        ax_trivial, ax_static, ax_imbal = axes

        # Too many trivial episodes
        if durations.size:
            trivial_thresh = 5.0
            trivial_frac   = float((durations < trivial_thresh).mean())
            colors = [COLOR_ACCEPTED if d >= trivial_thresh else COLOR_REJECTED
                      for d in durations]
            ax_trivial.scatter(range(len(durations)), np.sort(durations),
                               s=8, c=colors, alpha=0.7, linewidths=0)
            ax_trivial.axhline(trivial_thresh, color=COLOR_WARN, linestyle="--",
                               linewidth=1.2, label=f"trivial threshold={trivial_thresh}s")
            ax_trivial.set_ylabel("Duration (s)", fontsize=FONT_SIZE_LABEL)
            ax_trivial.set_title("Trivial Episode Detection", fontsize=FONT_SIZE_TITLE)
            ax_trivial.legend(fontsize=8)
            ax_trivial.tick_params(labelsize=FONT_SIZE_TICK)
            if trivial_frac > 0.2 or "trivial_episodes" in detected:
                annotate_failure(ax_trivial, f"{trivial_frac*100:.1f}% trivial episodes")
        else:
            ax_trivial.set_visible(False)

        # Long static periods
        if stall_ratios.size:
            ax_static.plot(np.sort(stall_ratios)[::-1], color=COLOR_REJECTED, linewidth=1.0)
            ax_static.axhline(0.3, color=COLOR_WARN, linestyle="--",
                              linewidth=1.2, label="30% stall threshold")
            ax_static.set_ylabel("Stall ratio", fontsize=FONT_SIZE_LABEL)
            ax_static.set_title("Long Static Period Detection", fontsize=FONT_SIZE_TITLE)
            ax_static.legend(fontsize=8)
            ax_static.tick_params(labelsize=FONT_SIZE_TICK)
        else:
            ax_static.set_visible(False)

        # Data imbalance: motion energy distribution
        if motion.size:
            percentiles = np.percentile(motion, [25, 50, 75])
            ax_imbal.hist(motion, bins=30, color=COLOR_INFO, edgecolor="white", linewidth=0.3)
            for p, label in zip(percentiles, ["Q1", "Q2", "Q3"]):
                ax_imbal.axvline(p, color=COLOR_WARN, linestyle="--", linewidth=1.0,
                                 label=f"{label}={p:.3f}")
            ax_imbal.set_xlabel("Motion energy (cumulative Δjoint)", fontsize=FONT_SIZE_LABEL)
            ax_imbal.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
            ax_imbal.set_title("Data Imbalance\n(motion energy distribution)", fontsize=FONT_SIZE_TITLE)
            ax_imbal.legend(fontsize=7)
            ax_imbal.tick_params(labelsize=FONT_SIZE_TICK)
            iqr = percentiles[2] - percentiles[0]
            if iqr < 0.05 or "data_imbalance" in detected:
                annotate_failure(ax_imbal, "Narrow IQR → imbalanced motion diversity")
        else:
            ax_imbal.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "03_failure_modes.png"))
        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone episode visualizer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-manifest", metavar="PATH")
    source.add_argument("--from-json",     metavar="PATH")
    parser.add_argument("--output-dir", default="data_processed/plots/episode")
    args = parser.parse_args()

    rng = np.random.default_rng(_RNG_SEED)
    E = 30
    metrics = {
        "episode_durations": rng.normal(12, 4, E).clip(2, 30),
        "motion_energy":     rng.exponential(0.5, E),
        "stall_ratios":      rng.beta(1, 5, E),
        "quality_timeline":  {f"ep_{i:04d}": rng.beta(6, 2, 40) for i in range(6)},
        "failure_modes":     [],
    }

    class _DummyConfig: pass
    viz = StageVisualizer(Path(args.output_dir), _DummyConfig())
    for p in viz.plot(metrics):
        print(f"  Saved: {p}")