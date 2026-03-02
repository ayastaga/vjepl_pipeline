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
    annotate_failure,
    make_figure,
    plot_histogram,
    plot_scatter,
    plot_time_series,
    save_figure,
)

_RNG_SEED = 42


class StageVisualizer(BaseStageVisualizer):
    """Visualizer for Stage 2: Clock Drift Correction."""

    STAGE_NAME = "drift"

    # ── Primary diagnostics ──────────────────────────────────────────────────

    def plot_primary_diagnostics(self, metrics: dict) -> list[Path]:
        """
        Panel 1: Raw vs corrected timestamp overlay.
        Panel 2: Linear regression line (α, β) overlaid on scatter.
        """
        paths = []

        raw_cam    = np.asarray(metrics.get("raw_cam_ts", []))
        raw_ctrl   = np.asarray(metrics.get("raw_ctrl_ts", []))
        corr_ctrl  = np.asarray(metrics.get("corrected_ctrl_ts", []))
        alpha      = metrics.get("alpha", None)
        beta       = metrics.get("beta", None)

        # ── Plot A: raw vs corrected timestamps ──────────────────────────────
        if raw_ctrl.size and corr_ctrl.size:
            fig, axes = make_figure(1, 2, title="Drift Correction — Timestamp Alignment")
            ax0, ax1 = axes

            t = np.arange(len(raw_ctrl))
            ax0.plot(t, raw_ctrl, color=COLOR_REJECTED, linewidth=0.9,
                     label="raw ctrl", alpha=0.8)
            ax0.plot(t, corr_ctrl, color=COLOR_ACCEPTED, linewidth=0.9,
                     label="corrected ctrl", alpha=0.9)
            if raw_cam.size:
                cam_t = np.linspace(0, len(raw_ctrl) - 1, len(raw_cam))
                ax0.scatter(cam_t, raw_cam, s=4, color=COLOR_WARN,
                            label="raw cam", alpha=0.5, zorder=3)
            ax0.set_xlabel("Sample index", fontsize=9)
            ax0.set_ylabel("Timestamp (s)", fontsize=9)
            ax0.set_title("Raw vs Corrected Timestamps", fontsize=10)
            ax0.legend(fontsize=8)
            ax0.tick_params(labelsize=8)

            # ── Plot B: regression scatter ──────────────────────────────────
            if raw_cam.size and raw_ctrl.size:
                n = min(len(raw_cam), len(raw_ctrl))
                ax1.scatter(raw_cam[:n], raw_ctrl[:n], s=6, color=COLOR_INFO,
                            alpha=0.5, linewidths=0, label="samples")
                if alpha is not None and beta is not None:
                    x_line = np.linspace(raw_cam[:n].min(), raw_cam[:n].max(), 300)
                    y_line = alpha * x_line + beta
                    ax1.plot(x_line, y_line, color=COLOR_WARN, linewidth=1.5,
                             linestyle="--",
                             label=f"T_ctrl = {alpha:.5f}·T_cam + {beta:.4f}")
                ax1.set_xlabel("Camera timestamp (s)", fontsize=9)
                ax1.set_ylabel("Control timestamp (s)", fontsize=9)
                ax1.set_title("Regression: Clock Model", fontsize=10)
                ax1.legend(fontsize=8)
                ax1.tick_params(labelsize=8)
            else:
                ax1.set_visible(False)

            fig.tight_layout()
            paths.append(self._save(fig, "01_timestamp_alignment.png"))

        return paths

    # ── Distributions ────────────────────────────────────────────────────────

    def plot_distributions(self, metrics: dict) -> list[Path]:
        """
        Panel: Residual histogram + drift-over-episode-time series.
        """
        paths = []
        residuals       = np.asarray(metrics.get("residuals", []))
        drift_over_time = np.asarray(metrics.get("drift_over_time", []))

        n_panels = int(residuals.size > 0) + int(drift_over_time.size > 0)
        if n_panels == 0:
            return paths

        fig, axes = make_figure(1, max(n_panels, 1),
                                title="Drift Correction — Signal Distributions")
        if n_panels == 1:
            axes = [axes]

        idx = 0
        if residuals.size:
            residuals_ms = residuals * 1000.0  # convert s → ms
            plot_histogram(
                axes[idx], residuals_ms,
                xlabel="Residual (ms)",
                title="Regression Residuals",
                bins=50,
                color=COLOR_INFO,
                threshold=0.0,
                threshold_label="zero-residual",
            )
            # annotate std
            axes[idx].annotate(
                f"σ = {residuals_ms.std():.3f} ms\nμ = {residuals_ms.mean():.3f} ms",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8),
            )
            idx += 1

        if drift_over_time.size:
            t = np.linspace(0, 1, len(drift_over_time))
            plot_time_series(
                axes[idx], t, drift_over_time * 1000.0,
                ylabel="Cumulative drift (ms)",
                title="Drift Over Episode Time",
                color=COLOR_WARN,
            )
            idx += 1

        fig.tight_layout()
        paths.append(self._save(fig, "02_drift_distributions.png"))
        return paths

    # ── Failure modes ────────────────────────────────────────────────────────

    def plot_failure_modes(self, metrics: dict) -> list[Path]:
        """
        Failure mode panel for drift stage:
        - Clock skew instability  → residual std > 5 ms
        - Firmware update drift   → stepwise jump in drift_over_time
        - Sensor desync           → R² of regression < 0.95
        """
        paths = []
        residuals       = np.asarray(metrics.get("residuals", []))
        drift_over_time = np.asarray(metrics.get("drift_over_time", []))
        alpha           = metrics.get("alpha", None)
        beta            = metrics.get("beta", None)
        raw_cam         = np.asarray(metrics.get("raw_cam_ts", []))
        raw_ctrl        = np.asarray(metrics.get("raw_ctrl_ts", []))
        detected        = metrics.get("failure_modes", [])

        fig, axes = make_figure(1, 3, title="Drift Correction — Failure Mode Diagnostics")
        ax_skew, ax_fw, ax_desync = axes

        # ── 1. Clock skew instability: rolling std of residuals ──────────────
        if residuals.size > 10:
            window = max(5, len(residuals) // 20)
            rolling_std = np.array([
                residuals[max(0, i - window): i + 1].std() * 1000.0
                for i in range(len(residuals))
            ])
            ax_skew.plot(rolling_std, color=COLOR_REJECTED, linewidth=0.9)
            ax_skew.axhline(5.0, color=COLOR_WARN, linestyle="--", linewidth=1.2,
                            label="5 ms instability threshold")
            ax_skew.set_xlabel("Sample", fontsize=9)
            ax_skew.set_ylabel("Rolling σ (ms)", fontsize=9)
            ax_skew.set_title("Clock Skew Instability", fontsize=10)
            ax_skew.legend(fontsize=8)
            ax_skew.tick_params(labelsize=8)
            if "clock_skew_instability" in detected or (rolling_std > 5.0).mean() > 0.1:
                annotate_failure(ax_skew, "Clock skew instability detected")
        else:
            ax_skew.set_visible(False)

        # ── 2. Firmware update drift: detect stepwise jumps ──────────────────
        if drift_over_time.size > 5:
            diff = np.diff(drift_over_time * 1000.0)
            t    = np.arange(len(drift_over_time))
            ax_fw.plot(t, drift_over_time * 1000.0, color=COLOR_INFO, linewidth=0.9,
                       label="drift")
            jump_mask = np.abs(diff) > np.abs(diff).mean() + 3 * np.abs(diff).std()
            jump_indices = np.where(jump_mask)[0]
            for ji in jump_indices:
                ax_fw.axvline(ji, color=COLOR_REJECTED, linewidth=1.0,
                              linestyle=":", alpha=0.7)
            ax_fw.set_xlabel("Frame index", fontsize=9)
            ax_fw.set_ylabel("Drift (ms)", fontsize=9)
            ax_fw.set_title("Firmware-Update Drift Detection", fontsize=10)
            ax_fw.tick_params(labelsize=8)
            if jump_indices.size or "firmware_update_drift" in detected:
                annotate_failure(ax_fw, f"{jump_indices.size} stepwise jump(s) detected")
        else:
            ax_fw.set_visible(False)

        # ── 3. Sensor desync: R² of regression ──────────────────────────────
        if alpha is not None and beta is not None and raw_cam.size and raw_ctrl.size:
            n      = min(len(raw_cam), len(raw_ctrl))
            x, y   = raw_cam[:n], raw_ctrl[:n]
            y_pred = alpha * x + beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2     = 1.0 - ss_res / (ss_tot + 1e-12)

            colors = [
                COLOR_ACCEPTED if r2 >= 0.95 else COLOR_REJECTED,
                COLOR_WARN if 0.90 <= r2 < 0.95 else "lightgrey",
            ]
            ax_desync.bar(["R²", "1 - R²"], [r2, 1.0 - r2], color=colors)
            ax_desync.axhline(0.95, color=COLOR_WARN, linestyle="--", linewidth=1.2,
                              label="0.95 threshold")
            ax_desync.set_ylim(0, 1.05)
            ax_desync.set_ylabel("Value", fontsize=9)
            ax_desync.set_title("Sensor Desync (Regression R²)", fontsize=10)
            ax_desync.legend(fontsize=8)
            ax_desync.tick_params(labelsize=8)
            ax_desync.annotate(f"R² = {r2:.4f}", xy=(0.5, 0.85),
                               xycoords="axes fraction", ha="center", fontsize=9,
                               fontweight="bold")
            if r2 < 0.95 or "sensor_desync" in detected:
                annotate_failure(ax_desync, f"Sensor desync (R²={r2:.3f} < 0.95)")
        else:
            ax_desync.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "03_failure_modes.png"))
        return paths


# ── CLI entry point ────────────────────────────────────────────────────────────

def _build_metrics_from_json(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # Convert list-typed fields back to numpy arrays
    array_keys = ["raw_cam_ts", "raw_ctrl_ts", "corrected_ctrl_ts",
                  "residuals", "drift_over_time"]
    for k in array_keys:
        if k in raw:
            raw[k] = np.asarray(raw[k])
    return raw


def _build_metrics_from_manifest(manifest_path: str) -> dict:
    """
    Build a minimal drift metrics dict from manifest.csv.
    In production this would load per-episode drift JSON sidecars.
    Here we synthesize demo data for debugging purposes.
    """
    rng = np.random.default_rng(_RNG_SEED)
    n = 500
    t_cam  = np.linspace(0, 10, n)
    alpha  = 1.0003
    beta   = 0.012
    noise  = rng.normal(0, 0.002, n)
    t_ctrl = alpha * t_cam + beta + noise
    corrected = t_cam * alpha + beta
    return {
        "raw_cam_ts":        t_cam,
        "raw_ctrl_ts":       t_ctrl,
        "corrected_ctrl_ts": corrected,
        "alpha":             alpha,
        "beta":              beta,
        "residuals":         noise,
        "drift_over_time":   np.cumsum(rng.normal(0, 0.0001, n)),
        "failure_modes":     [],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone drift correction visualizer (debug mode)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-json",     metavar="PATH",
                        help="Load metrics from JSON sidecar file")
    source.add_argument("--from-manifest", metavar="PATH",
                        help="Synthesize debug metrics from manifest.csv")
    parser.add_argument("--output-dir", default="data_processed/plots/drift",
                        help="Output directory for plots")
    args = parser.parse_args()

    if args.from_json:
        metrics = _build_metrics_from_json(args.from_json)
    else:
        metrics = _build_metrics_from_manifest(args.from_manifest)

    class _DummyConfig:
        pass

    viz = StageVisualizer(output_dir=Path(args.output_dir), config=_DummyConfig())
    saved = viz.plot(metrics)
    for p in saved:
        print(f"  Saved: {p}")