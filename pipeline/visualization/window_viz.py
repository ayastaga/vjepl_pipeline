"""
pipeline/visualization/window_viz.py

Stage 4 — Window Sampling Visualization
=========================================
Consumes structured metrics from WindowSampler.
Produces plots to: data_processed/plots/window/

Metrics dict contract:
    anchor_times         : np.ndarray   — anchor timestamps for all windows
    episode_durations    : dict[str, float]  — episode_id → duration (s)
    window_starts        : np.ndarray   — window start times
    window_ends          : np.ndarray   — window end times
    overlap_matrix       : np.ndarray   — per-window pair IoU (sampled subset)
    coverage_per_episode : dict[str, float]  — episode_id → coverage fraction
    anchor_density       : np.ndarray   — per-frame anchor frequency histogram
    blind_gap_frames     : int          — number of blind-gap frames
    failure_modes        : list[str]

CLI usage:
    python -m pipeline.visualization.window_viz --from-manifest data_processed/manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pipeline.visualization.base import (
    BaseStageVisualizer,
    COLOR_ACCEPTED, COLOR_INFO, COLOR_REJECTED, COLOR_WARN,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_TITLE,
    annotate_failure, make_figure, plot_histogram,
)

_RNG_SEED = 42


class StageVisualizer(BaseStageVisualizer):
    STAGE_NAME = "window"

    def plot_primary_diagnostics(self, metrics: dict) -> list[Path]:
        """
        Diagram: Context / blind gap / target window overlay for 5 sample windows.
        """
        paths = []
        anchor_times = np.asarray(metrics.get("anchor_times", []))
        blind_gap    = int(metrics.get("blind_gap_frames", 2))

        if not anchor_times.size:
            return paths

        # Show first 8 windows as timeline bands
        n_show = min(8, len(anchor_times))
        fig, ax = plt.subplots(figsize=(10, n_show * 0.6 + 1.5))
        fig.suptitle("Window Sampling — Context / Blind Gap / Target Layout",
                     fontsize=FONT_SIZE_TITLE + 1, fontweight="bold")

        ctx_dur  = 2.0   # 2 s context
        tgt_dur  = 1.0   # 1 s target
        gap_s    = blind_gap / 30.0  # convert frames to seconds at 30 Hz

        for i, anchor in enumerate(anchor_times[:n_show]):
            y    = i
            # Context bar
            ax.barh(y, ctx_dur, left=anchor - ctx_dur - gap_s / 2,
                    height=0.5, color=COLOR_INFO, alpha=0.7, label="Context" if i == 0 else "")
            # Blind gap
            ax.barh(y, gap_s, left=anchor - gap_s / 2,
                    height=0.5, color=COLOR_WARN, alpha=0.9, label="Blind gap" if i == 0 else "")
            # Target bar
            ax.barh(y, tgt_dur, left=anchor + gap_s / 2,
                    height=0.5, color=COLOR_ACCEPTED, alpha=0.7, label="Target" if i == 0 else "")
            # Anchor marker
            ax.axvline(anchor, ymin=(i - 0.3) / n_show, ymax=(i + 0.3) / n_show,
                       color="black", linewidth=0.8, alpha=0.6)

        ax.set_yticks(range(n_show))
        ax.set_yticklabels([f"win_{j}" for j in range(n_show)], fontsize=FONT_SIZE_TICK)
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL)
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)

        fig.tight_layout()
        paths.append(self._save(fig, "01_window_layout.png"))
        return paths

    def plot_distributions(self, metrics: dict) -> list[Path]:
        """
        Panel A: Anchor time distribution across episode.
        Panel B: Window overlap heatmap (sampled subset).
        Panel C: Coverage per episode bar.
        """
        paths = []
        anchor_times   = np.asarray(metrics.get("anchor_times", []))
        anchor_density = np.asarray(metrics.get("anchor_density", []))
        overlap_matrix = np.asarray(metrics.get("overlap_matrix", []))
        coverage       = metrics.get("coverage_per_episode", {})

        fig, axes = make_figure(1, 3, title="Window Sampling — Distributions")
        ax_anchor, ax_overlap, ax_cov = axes

        # Anchor distribution
        if anchor_times.size:
            ax_anchor.hist(anchor_times % anchor_times.max(), bins=40,
                           color=COLOR_INFO, edgecolor="white", linewidth=0.3)
            ax_anchor.set_xlabel("Position in episode (normalized)", fontsize=FONT_SIZE_LABEL)
            ax_anchor.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
            ax_anchor.set_title("Anchor Time Distribution", fontsize=FONT_SIZE_TITLE)
            ax_anchor.tick_params(labelsize=FONT_SIZE_TICK)
        else:
            ax_anchor.set_visible(False)

        # Overlap heatmap
        if overlap_matrix.size and overlap_matrix.ndim == 2:
            import seaborn as sns
            sns.heatmap(overlap_matrix[:20, :20], ax=ax_overlap, cmap="Blues",
                        vmin=0, vmax=1, linewidths=0.2, linecolor="white",
                        cbar_kws={"shrink": 0.8})
            ax_overlap.set_title("Window Overlap IoU\n(top-20 subset)", fontsize=FONT_SIZE_TITLE)
            ax_overlap.tick_params(labelsize=6)
            if overlap_matrix.mean() > 0.5:
                annotate_failure(ax_overlap, f"Excessive redundancy (mean IoU={overlap_matrix.mean():.2f})")
        else:
            ax_overlap.set_visible(False)

        # Coverage per episode
        if coverage:
            ep_ids = list(coverage.keys())[:20]
            covs   = [coverage[e] for e in ep_ids]
            colors = [COLOR_ACCEPTED if c >= 0.8 else COLOR_WARN if c >= 0.5 else COLOR_REJECTED
                      for c in covs]
            ax_cov.bar(ep_ids, covs, color=colors, edgecolor="white", linewidth=0.4)
            ax_cov.axhline(0.8, color=COLOR_WARN, linestyle="--", linewidth=1.0,
                           label="80% coverage target")
            ax_cov.set_ylim(0, 1.05)
            ax_cov.set_ylabel("Coverage fraction", fontsize=FONT_SIZE_LABEL)
            ax_cov.set_title("Coverage Redundancy Per Episode", fontsize=FONT_SIZE_TITLE)
            ax_cov.tick_params(axis="x", labelsize=6, rotation=45)
            ax_cov.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
            ax_cov.legend(fontsize=8)
        else:
            ax_cov.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "02_sampling_distributions.png"))
        return paths

    def plot_failure_modes(self, metrics: dict) -> list[Path]:
        """
        Action leakage detection, biased anchor sampling, excessive redundancy.
        """
        paths = []
        detected     = metrics.get("failure_modes", [])
        anchor_times = np.asarray(metrics.get("anchor_times", []))
        overlap_matrix = np.asarray(metrics.get("overlap_matrix", []))

        fig, axes = make_figure(1, 3, title="Window Sampling — Failure Mode Diagnostics")
        ax_leak, ax_bias, ax_redund = axes

        # 1. Action leakage proxy: anchor distribution near end of episode
        if anchor_times.size:
            normalized = (anchor_times - anchor_times.min()) / (anchor_times.ptp() + 1e-9)
            ax_leak.hist(normalized, bins=40, color=COLOR_INFO, edgecolor="white", linewidth=0.3)
            ax_leak.axvspan(0.9, 1.0, alpha=0.2, color=COLOR_REJECTED,
                            label="Final 10% (leakage risk)")
            ax_leak.set_xlabel("Normalized position in episode", fontsize=FONT_SIZE_LABEL)
            ax_leak.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
            ax_leak.set_title("Action Leakage Risk\n(anchors near episode end)",
                              fontsize=FONT_SIZE_TITLE)
            ax_leak.legend(fontsize=8)
            ax_leak.tick_params(labelsize=FONT_SIZE_TICK)
            tail_frac = float((normalized > 0.9).mean())
            if tail_frac > 0.05 or "action_leakage" in detected:
                annotate_failure(ax_leak, f"End-of-episode anchors: {tail_frac*100:.1f}%")
        else:
            ax_leak.set_visible(False)

        # 2. Biased anchor sampling: check uniform coverage
        if anchor_times.size:
            bins = np.linspace(anchor_times.min(), anchor_times.max(), 11)
            counts, _ = np.histogram(anchor_times, bins=bins)
            expected  = anchor_times.size / 10
            cv        = counts.std() / (counts.mean() + 1e-9)  # coefficient of variation
            ax_bias.bar(range(10), counts, color=COLOR_WARN, edgecolor="white", linewidth=0.4)
            ax_bias.axhline(expected, color=COLOR_ACCEPTED, linestyle="--",
                            linewidth=1.2, label=f"Expected uniform ({expected:.0f})")
            ax_bias.set_xlabel("Episode time decile", fontsize=FONT_SIZE_LABEL)
            ax_bias.set_ylabel("Window count", fontsize=FONT_SIZE_LABEL)
            ax_bias.set_title(f"Anchor Sampling Bias\n(CV={cv:.2f}, lower=more uniform)",
                              fontsize=FONT_SIZE_TITLE)
            ax_bias.legend(fontsize=8)
            ax_bias.tick_params(labelsize=FONT_SIZE_TICK)
            if cv > 0.3 or "biased_anchor_sampling" in detected:
                annotate_failure(ax_bias, f"Biased anchor sampling (CV={cv:.2f})")
        else:
            ax_bias.set_visible(False)

        # 3. Excessive redundancy
        if overlap_matrix.size and overlap_matrix.ndim == 2:
            high_overlap = (overlap_matrix > 0.8).mean()
            ax_redund.hist(overlap_matrix.flatten(), bins=30,
                           color=COLOR_REJECTED, edgecolor="white", linewidth=0.3)
            ax_redund.axvline(0.8, color=COLOR_WARN, linestyle="--", linewidth=1.2,
                              label="0.8 redundancy threshold")
            ax_redund.set_xlabel("Window IoU", fontsize=FONT_SIZE_LABEL)
            ax_redund.set_ylabel("Pair count", fontsize=FONT_SIZE_LABEL)
            ax_redund.set_title("Window Redundancy Distribution", fontsize=FONT_SIZE_TITLE)
            ax_redund.legend(fontsize=8)
            ax_redund.tick_params(labelsize=FONT_SIZE_TICK)
            if high_overlap > 0.15 or "excessive_redundancy" in detected:
                annotate_failure(ax_redund, f"{high_overlap*100:.1f}% pairs highly redundant")
        else:
            ax_redund.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "03_failure_modes.png"))
        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone window sampling visualizer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-manifest", metavar="PATH")
    source.add_argument("--from-json",     metavar="PATH")
    parser.add_argument("--output-dir", default="data_processed/plots/window")
    args = parser.parse_args()

    rng = np.random.default_rng(_RNG_SEED)
    n   = 200
    metrics = {
        "anchor_times":          np.sort(rng.uniform(2.5, 15.0, n)),
        "blind_gap_frames":      2,
        "anchor_density":        rng.integers(1, 20, 30),
        "overlap_matrix":        rng.uniform(0, 1, (30, 30)),
        "coverage_per_episode":  {f"ep_{i:04d}": float(rng.uniform(0.4, 1.0))
                                   for i in range(10)},
        "failure_modes":         [],
    }

    class _DummyConfig:
        pass

    viz = StageVisualizer(Path(args.output_dir), _DummyConfig())
    for p in viz.plot(metrics):
        print(f"  Saved: {p}")