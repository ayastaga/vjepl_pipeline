from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.visualization.base import (
    BaseStageVisualizer,
    COLOR_ACCEPTED,
    COLOR_INFO,
    COLOR_REJECTED,
    COLOR_WARN,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    annotate_failure,
    make_figure,
    plot_histogram,
    plot_time_series,
    save_figure,
)

_RNG_SEED = 42


class StageVisualizer(BaseStageVisualizer):
    """Visualizer for Stage 3: Timestamp Alignment."""

    STAGE_NAME = "alignment"

    def plot_primary_diagnostics(self, metrics: dict) -> list[Path]:
        """
        Panel A: Camera vs control timestamp alignment scatter.
        Panel B: Interpolation method coverage bar.
        """
        paths = []
        cam_ts  = np.asarray(metrics.get("cam_ts_aligned", []))
        ctrl_ts = np.asarray(metrics.get("ctrl_ts_aligned", []))

        if not cam_ts.size or not ctrl_ts.size:
            return paths

        fig, axes = make_figure(1, 2, title="Temporal Alignment — Timestamp Coverage")
        ax_scatter, ax_method = axes

        # Camera vs control timestamp alignment
        n = min(len(cam_ts), len(ctrl_ts), 2000)
        ax_scatter.scatter(cam_ts[:n], np.zeros(n), s=5, color=COLOR_INFO,
                           alpha=0.5, label="Camera (30 Hz)", linewidths=0)
        ax_scatter.scatter(ctrl_ts[:n], np.ones(n) * 0.5, s=5, color=COLOR_WARN,
                           alpha=0.5, label="Control (50 Hz)", linewidths=0)
        ax_scatter.set_yticks([0, 0.5])
        ax_scatter.set_yticklabels(["Camera", "Control"], fontsize=FONT_SIZE_TICK)
        ax_scatter.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL)
        ax_scatter.set_title("Video vs Control Timestamps", fontsize=FONT_SIZE_TITLE)
        ax_scatter.legend(fontsize=8)
        ax_scatter.set_ylim(-0.3, 0.8)

        # Method usage summary
        method = metrics.get("interp_method", "unknown")
        method_counts = metrics.get("method_counts", {
            "cubic_spline": 60, "slerp": 4, "nearest": 36
        })
        ax_method.bar(list(method_counts.keys()), list(method_counts.values()),
                      color=[COLOR_ACCEPTED, COLOR_INFO, COLOR_WARN],
                      edgecolor="white", linewidth=0.4)
        ax_method.set_ylabel("Signal count", fontsize=FONT_SIZE_LABEL)
        ax_method.set_title("Interpolation Method Usage", fontsize=FONT_SIZE_TITLE)
        ax_method.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
        ax_method.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

        fig.tight_layout()
        paths.append(self._save(fig, "01_timestamp_coverage.png"))
        return paths

    def plot_distributions(self, metrics: dict) -> list[Path]:
        """
        Panel A: Cubic spline velocity continuity (first derivative).
        Panel B: SLERP angular velocity magnitude.
        Panel C: State-action residual norm.
        """
        paths = []
        spline_vel  = np.asarray(metrics.get("spline_velocity", []))
        angular_vel = np.asarray(metrics.get("slerp_angular_vel", []))
        sa_residuals = np.asarray(metrics.get("state_action_residuals", []))

        n_panels = sum([spline_vel.size > 0, angular_vel.size > 0, sa_residuals.size > 0])
        if n_panels == 0:
            return paths

        fig, axes = make_figure(1, max(n_panels, 1),
                                title="Alignment — Signal Continuity Diagnostics")
        flat = np.array(axes).flatten() if n_panels > 1 else [axes]
        idx = 0

        if spline_vel.size:
            t = np.linspace(0, 1, len(spline_vel))
            flat[idx].plot(t, spline_vel, color=COLOR_INFO, linewidth=0.8, alpha=0.8)
            flat[idx].set_title("Cubic Spline Velocity (C² continuity)", fontsize=FONT_SIZE_TITLE)
            flat[idx].set_xlabel("Normalized time", fontsize=FONT_SIZE_LABEL)
            flat[idx].set_ylabel("Joint velocity (rad/s)", fontsize=FONT_SIZE_LABEL)
            flat[idx].tick_params(labelsize=FONT_SIZE_TICK)
            # Detect discontinuities (sharp jumps in derivative)
            jumps = np.abs(np.diff(spline_vel))
            if jumps.max() > 5 * jumps.mean():
                annotate_failure(flat[idx], "Interpolation artifact: velocity discontinuity")
            idx += 1

        if angular_vel.size:
            t = np.linspace(0, 1, len(angular_vel))
            flat[idx].plot(t, angular_vel, color=COLOR_WARN, linewidth=0.8, alpha=0.8)
            flat[idx].set_title("SLERP Angular Velocity Smoothness", fontsize=FONT_SIZE_TITLE)
            flat[idx].set_xlabel("Normalized time", fontsize=FONT_SIZE_LABEL)
            flat[idx].set_ylabel("Angular velocity (rad/s)", fontsize=FONT_SIZE_LABEL)
            flat[idx].tick_params(labelsize=FONT_SIZE_TICK)
            idx += 1

        if sa_residuals.size:
            plot_histogram(flat[idx], sa_residuals,
                           xlabel="Residual norm", title="State vs Action Residual Norm",
                           bins=40, color=COLOR_REJECTED)
            if "state_discontinuities" in metrics.get("failure_modes", []):
                annotate_failure(flat[idx], "State discontinuities detected")
            idx += 1

        fig.tight_layout()
        paths.append(self._save(fig, "02_signal_continuity.png"))
        return paths

    def plot_failure_modes(self, metrics: dict) -> list[Path]:
        """
        Failure panel: Interpolation artifacts, gimbal-lock, state discontinuities.
        """
        paths = []
        detected = metrics.get("failure_modes", [])
        spline_vel   = np.asarray(metrics.get("spline_velocity", []))
        angular_vel  = np.asarray(metrics.get("slerp_angular_vel", []))
        sa_residuals = np.asarray(metrics.get("state_action_residuals", []))

        fig, axes = make_figure(1, 3, title="Alignment — Failure Mode Diagnostics")
        ax_interp, ax_gimbal, ax_disc = axes

        # 1. Interpolation artifacts: power spectrum of spline velocity
        if spline_vel.size:
            fft_mag = np.abs(np.fft.rfft(spline_vel - spline_vel.mean()))
            freqs   = np.fft.rfftfreq(len(spline_vel), d=1.0 / 30.0)
            ax_interp.plot(freqs[1:], fft_mag[1:], color=COLOR_INFO, linewidth=0.8)
            ax_interp.set_xlabel("Frequency (Hz)", fontsize=FONT_SIZE_LABEL)
            ax_interp.set_ylabel("|FFT|", fontsize=FONT_SIZE_LABEL)
            ax_interp.set_title("Spline Velocity Frequency\n(artifact = high-freq spike)",
                                fontsize=FONT_SIZE_TITLE)
            ax_interp.tick_params(labelsize=FONT_SIZE_TICK)
            # High-frequency energy > 1 Hz suggests Runge's phenomenon
            hf_mask = freqs[1:] > 1.0
            if hf_mask.any() and fft_mag[1:][hf_mask].mean() > fft_mag[1:].mean() * 3:
                annotate_failure(ax_interp, "Interpolation artifact (Runge's phenomenon)")
        else:
            ax_interp.set_visible(False)

        # 2. Gimbal-lock: angular velocity > 3π rad/s (SLERP singularity proxy)
        if angular_vel.size:
            gimbal_mask = angular_vel > np.pi
            ax_gimbal.plot(np.linspace(0, 1, len(angular_vel)), angular_vel,
                           color=COLOR_WARN, linewidth=0.8)
            ax_gimbal.fill_between(
                np.linspace(0, 1, len(angular_vel)),
                np.pi, np.where(gimbal_mask, angular_vel, np.pi),
                alpha=0.3, color=COLOR_REJECTED
            )
            ax_gimbal.axhline(np.pi, color=COLOR_REJECTED, linestyle="--",
                              linewidth=1.2, label="π rad/s (gimbal threshold)")
            ax_gimbal.set_xlabel("Normalized time", fontsize=FONT_SIZE_LABEL)
            ax_gimbal.set_ylabel("Angular velocity (rad/s)", fontsize=FONT_SIZE_LABEL)
            ax_gimbal.set_title("Gimbal-Lock Detection\n(SLERP angular velocity spike)",
                                fontsize=FONT_SIZE_TITLE)
            ax_gimbal.legend(fontsize=8)
            ax_gimbal.tick_params(labelsize=FONT_SIZE_TICK)
            if gimbal_mask.mean() > 0.01 or "gimbal_lock" in detected:
                annotate_failure(ax_gimbal, f"Gimbal-lock frames: {gimbal_mask.sum()}")
        else:
            ax_gimbal.set_visible(False)

        # 3. State discontinuities
        if sa_residuals.size:
            sorted_r = np.sort(sa_residuals)
            ax_disc.plot(sorted_r, color=COLOR_REJECTED, linewidth=0.9)
            p95 = np.percentile(sa_residuals, 95)
            ax_disc.axhline(p95, color=COLOR_WARN, linestyle="--",
                            linewidth=1.2, label=f"95th pct = {p95:.4f}")
            ax_disc.set_xlabel("Window rank", fontsize=FONT_SIZE_LABEL)
            ax_disc.set_ylabel("State-action residual", fontsize=FONT_SIZE_LABEL)
            ax_disc.set_title("State Discontinuities\n(sorted residual norm)",
                              fontsize=FONT_SIZE_TITLE)
            ax_disc.legend(fontsize=8)
            ax_disc.tick_params(labelsize=FONT_SIZE_TICK)
            if sa_residuals.max() > p95 * 3 or "state_discontinuities" in detected:
                annotate_failure(ax_disc, "State discontinuities in tail")
        else:
            ax_disc.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "03_failure_modes.png"))
        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone alignment visualizer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-manifest", metavar="PATH")
    source.add_argument("--from-json",     metavar="PATH")
    parser.add_argument("--output-dir", default="data_processed/plots/alignment")
    args = parser.parse_args()

    rng = np.random.default_rng(_RNG_SEED)
    n   = 300
    t   = np.linspace(0, 10, n)
    metrics = {
        "cam_ts_aligned":        np.linspace(0, 10, int(n * 0.6)),
        "ctrl_ts_aligned":       np.linspace(0, 10, n),
        "spline_velocity":       np.cumsum(rng.normal(0, 0.05, n)),
        "slerp_angular_vel":     np.abs(rng.normal(0.5, 0.3, int(n * 0.6))),
        "state_action_residuals": rng.exponential(0.02, n),
        "interp_method":         "cubic_spline",
        "method_counts":         {"cubic_spline": 60, "slerp": 4, "nearest": 36},
        "failure_modes":         [],
    }

    class _DummyConfig:
        pass

    viz = StageVisualizer(Path(args.output_dir), _DummyConfig())
    for p in viz.plot(metrics):
        print(f"  Saved: {p}")