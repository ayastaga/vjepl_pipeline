"""
pipeline/visualization/dynamics_viz.py

Stage — Action & State Dynamics Visualization
==============================================
Consumes structured metrics from downstream analysis (post-export).
Produces plots to: data_processed/plots/dynamics/

Metrics dict contract:
    action_magnitudes    : np.ndarray  — per-window action L2 norm, shape (N,)
    action_directions    : np.ndarray  — per-window dominant action direction angle (rad), shape (N,)
    joint_velocities     : np.ndarray  — per-window per-joint velocity rms, shape (N, 6)
    joint_accelerations  : np.ndarray  — per-window per-joint accel rms, shape (N, 6)
    action_entropy       : np.ndarray  — per-episode action entropy, shape (E,)
    action_covariance    : np.ndarray  — 6×6 covariance matrix of actions over dataset
    joint_names          : list[str]   — names of the 6 DOF joints
    failure_modes        : list[str]

CLI usage:
    python -m pipeline.visualization.dynamics_viz --from-manifest data_processed/manifest.csv
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
    COLOR_INFO, COLOR_REJECTED, COLOR_WARN,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_TITLE,
    annotate_failure, make_figure, plot_histogram, plot_heatmap,
)

_RNG_SEED  = 42
_DOF_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]


class StageVisualizer(BaseStageVisualizer):
    STAGE_NAME = "dynamics"

    def plot_primary_diagnostics(self, metrics: dict) -> list[Path]:
        """
        Panel A: Action magnitude histogram.
        Panel B: Action direction polar plot.
        """
        paths = []
        magnitudes = np.asarray(metrics.get("action_magnitudes", []))
        directions = np.asarray(metrics.get("action_directions", []))

        if not magnitudes.size:
            return paths

        fig = plt.figure(figsize=(9, 3.5))
        fig.suptitle("Dynamics — Action Magnitude & Direction", fontsize=FONT_SIZE_TITLE + 1,
                     fontweight="bold")

        ax_hist = fig.add_subplot(1, 2, 1)
        plot_histogram(ax_hist, magnitudes, xlabel="Action L2 norm",
                       title="Action Magnitude Distribution", bins=50, color=COLOR_INFO)
        if magnitudes.std() < 0.01 or magnitudes.max() < 0.05:
            annotate_failure(ax_hist, "Under-excitation: very low action magnitudes")

        ax_polar = fig.add_subplot(1, 2, 2, polar=True)
        if directions.size:
            finite = directions[np.isfinite(directions)]
            bins   = np.linspace(-np.pi, np.pi, 37)
            counts, edges = np.histogram(finite, bins=bins)
            thetas = (edges[:-1] + edges[1:]) / 2
            ax_polar.bar(thetas, counts, width=np.diff(edges),
                         color=COLOR_INFO, alpha=0.7, edgecolor="white", linewidth=0.3)
            ax_polar.set_title("Action Direction Distribution\n(polar)", fontsize=FONT_SIZE_TITLE,
                               pad=15)
            ax_polar.tick_params(labelsize=FONT_SIZE_TICK)
        else:
            ax_polar.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "01_action_magnitude_direction.png"))
        return paths

    def plot_distributions(self, metrics: dict) -> list[Path]:
        """
        Panel A: Per-joint velocity distribution.
        Panel B: Per-joint acceleration distribution.
        Panel C: Action entropy over dataset.
        Panel D: Action covariance heatmap.
        """
        paths = []
        joint_vel   = np.asarray(metrics.get("joint_velocities", []))
        joint_accel = np.asarray(metrics.get("joint_accelerations", []))
        entropy     = np.asarray(metrics.get("action_entropy", []))
        covariance  = np.asarray(metrics.get("action_covariance", []))
        joint_names = metrics.get("joint_names", _DOF_NAMES)

        # Velocity + acceleration distributions
        if joint_vel.size and joint_vel.ndim == 2:
            fig, axes = make_figure(1, 2, title="Dynamics — Joint Velocity & Acceleration")
            ax_vel, ax_acc = axes

            import pandas as pd
            vel_df = pd.DataFrame(joint_vel[:, :len(joint_names)], columns=joint_names[:joint_vel.shape[1]])
            vel_df.boxplot(ax=ax_vel, patch_artist=True)
            ax_vel.set_title("Joint Velocity RMS (per window)", fontsize=FONT_SIZE_TITLE)
            ax_vel.set_xlabel("Joint", fontsize=FONT_SIZE_LABEL)
            ax_vel.set_ylabel("RMS velocity (rad/s)", fontsize=FONT_SIZE_LABEL)
            ax_vel.tick_params(labelsize=FONT_SIZE_TICK)

            if joint_accel.size and joint_accel.ndim == 2:
                acc_df = pd.DataFrame(joint_accel[:, :len(joint_names)], columns=joint_names[:joint_accel.shape[1]])
                acc_df.boxplot(ax=ax_acc, patch_artist=True)
                ax_acc.set_title("Joint Acceleration RMS (per window)", fontsize=FONT_SIZE_TITLE)
                ax_acc.set_xlabel("Joint", fontsize=FONT_SIZE_LABEL)
                ax_acc.set_ylabel("RMS acceleration (rad/s²)", fontsize=FONT_SIZE_LABEL)
                ax_acc.tick_params(labelsize=FONT_SIZE_TICK)
                if joint_accel.max() < 0.1:
                    annotate_failure(ax_acc, "Under-excitation: low joint acceleration")
            else:
                ax_acc.set_visible(False)

            fig.tight_layout()
            paths.append(self._save(fig, "02_joint_distributions.png"))

        # Entropy + covariance
        if entropy.size or covariance.size:
            n_panels = int(entropy.size > 0) + int(covariance.size > 0)
            fig, axes = make_figure(1, n_panels, title="Dynamics — Entropy & Covariance")
            flat = np.array(axes).flatten() if n_panels > 1 else [axes]
            idx  = 0

            if entropy.size:
                flat[idx].plot(entropy, color=COLOR_WARN, linewidth=1.0, alpha=0.85)
                flat[idx].axhline(entropy.mean(), color=COLOR_INFO, linestyle="--",
                                  linewidth=1.2, label=f"mean={entropy.mean():.3f}")
                flat[idx].set_xlabel("Episode index", fontsize=FONT_SIZE_LABEL)
                flat[idx].set_ylabel("Action entropy (nats)", fontsize=FONT_SIZE_LABEL)
                flat[idx].set_title("Action Entropy Over Dataset", fontsize=FONT_SIZE_TITLE)
                flat[idx].legend(fontsize=8)
                flat[idx].tick_params(labelsize=FONT_SIZE_TICK)
                if entropy.mean() < 0.5:
                    annotate_failure(flat[idx], "Dataset collapse: low entropy")
                idx += 1

            if covariance.size and covariance.ndim == 2:
                plot_heatmap(flat[idx], covariance,
                             xlabels=joint_names, ylabels=joint_names,
                             title="Action Covariance Heatmap", fmt=".3f",
                             cmap="RdBu_r")
                idx += 1

            fig.tight_layout()
            paths.append(self._save(fig, "03_entropy_covariance.png"))

        return paths

    def plot_failure_modes(self, metrics: dict) -> list[Path]:
        """
        Dataset collapse, under-excitation, low diversity diagnostics.
        """
        paths = []
        magnitudes  = np.asarray(metrics.get("action_magnitudes", []))
        entropy     = np.asarray(metrics.get("action_entropy", []))
        joint_vel   = np.asarray(metrics.get("joint_velocities", []))
        detected    = metrics.get("failure_modes", [])

        fig, axes = make_figure(1, 3, title="Dynamics — Failure Mode Diagnostics")
        ax_collapse, ax_excite, ax_div = axes

        # 1. Dataset collapse: near-zero entropy
        if entropy.size:
            ax_collapse.hist(entropy, bins=30, color=COLOR_WARN, edgecolor="white", linewidth=0.3)
            ax_collapse.axvline(1.0, color=COLOR_REJECTED, linestyle="--", linewidth=1.2,
                                label="min healthy entropy = 1.0")
            ax_collapse.set_xlabel("Action entropy (nats)", fontsize=FONT_SIZE_LABEL)
            ax_collapse.set_ylabel("Episode count", fontsize=FONT_SIZE_LABEL)
            ax_collapse.set_title("Dataset Collapse Detection", fontsize=FONT_SIZE_TITLE)
            ax_collapse.legend(fontsize=8)
            ax_collapse.tick_params(labelsize=FONT_SIZE_TICK)
            if entropy.mean() < 1.0 or "dataset_collapse" in detected:
                annotate_failure(ax_collapse, f"Collapse risk (mean entropy={entropy.mean():.3f})")
        else:
            ax_collapse.set_visible(False)

        # 2. Under-excitation: action magnitude percentile plot
        if magnitudes.size:
            percentiles = np.percentile(magnitudes, np.linspace(0, 100, 101))
            ax_excite.plot(np.linspace(0, 100, 101), percentiles,
                           color=COLOR_INFO, linewidth=1.5)
            ax_excite.axhline(0.05, color=COLOR_REJECTED, linestyle="--",
                              linewidth=1.2, label="0.05 under-excitation threshold")
            ax_excite.set_xlabel("Percentile", fontsize=FONT_SIZE_LABEL)
            ax_excite.set_ylabel("Action magnitude", fontsize=FONT_SIZE_LABEL)
            ax_excite.set_title("Action Magnitude Percentile Plot\n(under-excitation check)",
                                fontsize=FONT_SIZE_TITLE)
            ax_excite.legend(fontsize=8)
            ax_excite.tick_params(labelsize=FONT_SIZE_TICK)
            p75 = np.percentile(magnitudes, 75)
            if p75 < 0.05 or "under_excitation" in detected:
                annotate_failure(ax_excite, f"Under-excitation (P75={p75:.4f})")
        else:
            ax_excite.set_visible(False)

        # 3. Low diversity: per-joint coefficient of variation
        if joint_vel.size and joint_vel.ndim == 2:
            cv = joint_vel.std(axis=0) / (joint_vel.mean(axis=0) + 1e-9)
            joint_names = metrics.get("joint_names", _DOF_NAMES)[:joint_vel.shape[1]]
            colors = [COLOR_REJECTED if v < 0.2 else COLOR_INFO for v in cv]
            ax_div.bar(joint_names, cv, color=colors, edgecolor="white", linewidth=0.4)
            ax_div.axhline(0.2, color=COLOR_WARN, linestyle="--", linewidth=1.2,
                           label="CV=0.2 diversity minimum")
            ax_div.set_xlabel("Joint", fontsize=FONT_SIZE_LABEL)
            ax_div.set_ylabel("Coefficient of Variation", fontsize=FONT_SIZE_LABEL)
            ax_div.set_title("Joint Velocity Diversity\n(low CV → robot stuck in habit)",
                             fontsize=FONT_SIZE_TITLE)
            ax_div.legend(fontsize=8)
            ax_div.tick_params(labelsize=FONT_SIZE_TICK)
            low_div = [n for n, v in zip(joint_names, cv) if v < 0.2]
            if low_div or "low_diversity" in detected:
                annotate_failure(ax_div, f"Low diversity joints: {', '.join(low_div)}")
        else:
            ax_div.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "04_failure_modes.png"))
        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone dynamics visualizer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-manifest", metavar="PATH")
    source.add_argument("--from-json",     metavar="PATH")
    parser.add_argument("--output-dir", default="data_processed/plots/dynamics")
    args = parser.parse_args()

    rng = np.random.default_rng(_RNG_SEED)
    N, E = 500, 20
    metrics = {
        "action_magnitudes":   np.abs(rng.normal(0.15, 0.08, N)),
        "action_directions":   rng.uniform(-np.pi, np.pi, N),
        "joint_velocities":    np.abs(rng.normal(0.3, 0.15, (N, 6))),
        "joint_accelerations": np.abs(rng.normal(0.1, 0.05, (N, 6))),
        "action_entropy":      rng.uniform(0.5, 2.5, E),
        "action_covariance":   np.cov(rng.normal(0, 1, (6, N))),
        "joint_names":         _DOF_NAMES,
        "failure_modes":       [],
    }

    class _DummyConfig:
        pass

    viz = StageVisualizer(Path(args.output_dir), _DummyConfig())
    for p in viz.plot(metrics):
        print(f"  Saved: {p}")