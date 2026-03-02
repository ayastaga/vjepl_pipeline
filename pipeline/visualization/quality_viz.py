"""
pipeline/visualization/quality_viz.py

Stage 5 — Quality Flag Visualization
======================================
Consumes structured metrics emitted by QualityFlagger.
Produces plots to: data_processed/plots/quality/

Metrics dict contract:
    flag_counts       : dict[str, int]   — how many windows triggered each flag
    bitmask_values    : np.ndarray[int]  — per-window quality bitmask integers
    flag_signals      : dict[str, np.ndarray]  — raw signal per flag (e.g. blur values)
    flag_thresholds   : dict[str, float]       — configured threshold per flag
    flag_cooccurrence : np.ndarray (9×9)       — co-occurrence count matrix
    flag_types        : dict[str, str]         — "hard" | "soft" per flag name
    n_windows_total   : int
    failure_modes     : list[str]

FLAG NAMES (from quality.py):
    MISSING_V, JITTER_A, BLUR, STALL, EXPO_S, SYNC_ERR, DROP_P, COMP_A, JITTER_V

CLI usage:
    python -m pipeline.visualization.quality_viz --from-manifest data_processed/manifest.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pipeline.visualization.base import (
    BaseStageVisualizer,
    COLOR_ACCEPTED,
    COLOR_INFO,
    COLOR_REJECTED,
    COLOR_WARN,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    PALETTE_DIV,
    annotate_failure,
    make_figure,
    plot_heatmap,
    plot_histogram,
    save_figure,
)

_ALL_FLAGS = ["MISSING_V", "JITTER_A", "BLUR", "STALL", "EXPO_S",
              "SYNC_ERR", "DROP_P", "COMP_A", "JITTER_V"]
_HARD_FLAGS = {"MISSING_V", "DROP_P"}
_RNG_SEED = 42


class StageVisualizer(BaseStageVisualizer):
    """Visualizer for Stage 5: Quality Flag Detection."""

    STAGE_NAME = "quality"

    # ── Primary diagnostics ──────────────────────────────────────────────────

    def plot_primary_diagnostics(self, metrics: dict) -> list[Path]:
        """
        Panel A: Per-flag trigger frequency bar chart, coloured by hard/soft.
        Panel B: Bitmask integer frequency distribution.
        """
        paths = []
        flag_counts   = metrics.get("flag_counts", {})
        bitmasks      = np.asarray(metrics.get("bitmask_values", []))
        n_total       = int(metrics.get("n_windows_total", max(sum(flag_counts.values()), 1)))
        flag_types    = metrics.get("flag_types", {})

        # ── Panel A: flag frequency bar ──────────────────────────────────────
        if flag_counts:
            fig, axes = make_figure(1, 2, title="Quality Flags — Primary Diagnostics")
            ax_bar, ax_bitmask = axes

            flags  = _ALL_FLAGS
            counts = [flag_counts.get(f, 0) for f in flags]
            rates  = [c / max(n_total, 1) * 100 for c in counts]
            colors = [
                COLOR_REJECTED if flag_types.get(f, "soft") == "hard" else COLOR_INFO
                for f in flags
            ]

            bars = ax_bar.bar(flags, rates, color=colors, edgecolor="white", linewidth=0.4)
            ax_bar.set_xlabel("Quality Flag", fontsize=FONT_SIZE_LABEL)
            ax_bar.set_ylabel("Trigger Rate (%)", fontsize=FONT_SIZE_LABEL)
            ax_bar.set_title("Per-Flag Trigger Rates", fontsize=FONT_SIZE_TITLE)
            ax_bar.tick_params(axis="x", labelsize=FONT_SIZE_TICK, rotation=35)
            ax_bar.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{rate:.1f}%",
                    ha="center", va="bottom", fontsize=7,
                )

            # Legend: hard vs soft
            hard_patch = mpatches.Patch(color=COLOR_REJECTED, label="Hard flag (EQS→0)")
            soft_patch = mpatches.Patch(color=COLOR_INFO,     label="Soft flag (exp penalty)")
            ax_bar.legend(handles=[hard_patch, soft_patch], fontsize=8, loc="upper right")

            # ── Panel B: bitmask distribution ────────────────────────────────
            if bitmasks.size:
                unique_masks, mask_counts = np.unique(bitmasks, return_counts=True)
                # Show top 20 most common bitmasks
                top_idx   = np.argsort(mask_counts)[::-1][:20]
                top_masks = unique_masks[top_idx]
                top_counts = mask_counts[top_idx]
                top_labels = [f"0b{m:09b}" for m in top_masks]

                ax_bitmask.barh(range(len(top_masks)), top_counts,
                                color=COLOR_WARN, edgecolor="white", linewidth=0.4)
                ax_bitmask.set_yticks(range(len(top_masks)))
                ax_bitmask.set_yticklabels(top_labels, fontsize=6)
                ax_bitmask.set_xlabel("Window Count", fontsize=FONT_SIZE_LABEL)
                ax_bitmask.set_title("Top-20 Bitmask Patterns", fontsize=FONT_SIZE_TITLE)
                ax_bitmask.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
                ax_bitmask.invert_yaxis()
            else:
                ax_bitmask.set_visible(False)

            fig.tight_layout()
            paths.append(self._save(fig, "01_flag_frequency.png"))

        return paths

    # ── Distributions ────────────────────────────────────────────────────────

    def plot_distributions(self, metrics: dict) -> list[Path]:
        """
        For each flag that has a raw signal array, plot its distribution
        with threshold overlay. Laid out as a 3×3 grid (one cell per flag).
        """
        paths = []
        flag_signals    = metrics.get("flag_signals", {})
        flag_thresholds = metrics.get("flag_thresholds", {})

        available_flags = [f for f in _ALL_FLAGS if f in flag_signals]
        if not available_flags:
            return paths

        n_flags = len(available_flags)
        ncols   = 3
        nrows   = (n_flags + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 4.0, nrows * 3.2),
        )
        fig.suptitle("Quality Flags — Raw Signal Distributions", fontsize=FONT_SIZE_TITLE + 1,
                     fontweight="bold")
        flat_axes = np.array(axes).flatten()

        for i, flag in enumerate(available_flags):
            ax     = flat_axes[i]
            signal = np.asarray(flag_signals[flag])
            thresh = flag_thresholds.get(flag, None)
            color  = COLOR_REJECTED if flag in _HARD_FLAGS else COLOR_INFO

            finite = signal[np.isfinite(signal)]
            if finite.size == 0:
                ax.set_visible(False)
                continue

            ax.hist(finite, bins=40, color=color, edgecolor="white", linewidth=0.3,
                    density=True, alpha=0.85)
            ax.set_title(flag, fontsize=FONT_SIZE_TICK + 1, fontweight="bold")
            ax.set_xlabel("Signal value", fontsize=7)
            ax.set_ylabel("Density", fontsize=7)
            ax.tick_params(labelsize=7)

            if thresh is not None:
                ax.axvline(thresh, color=COLOR_WARN, linewidth=1.2, linestyle="--",
                           label=f"thresh={thresh:.3g}")
                ax.legend(fontsize=6)

            # Shade triggered region
            if thresh is not None:
                # Hard flags: missing above threshold means frames absent; varies by flag
                # General rule: shade values that would trigger the flag
                x_fill = np.linspace(finite.min(), finite.max(), 200)
                # For MISSING_V, DROP_P, JITTER_A, JITTER_V, EXPO_S, SYNC_ERR, COMP_A: triggered > thresh
                # For BLUR, STALL: triggered < thresh
                shade_above = flag not in {"BLUR", "STALL"}
                if shade_above:
                    ax.axvspan(thresh, finite.max(), alpha=0.12, color=COLOR_REJECTED)
                else:
                    ax.axvspan(finite.min(), thresh, alpha=0.12, color=COLOR_REJECTED)

        # Hide unused axes
        for j in range(len(available_flags), len(flat_axes)):
            flat_axes[j].set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "02_signal_distributions.png"))
        return paths

    # ── Failure modes ────────────────────────────────────────────────────────

    def plot_failure_modes(self, metrics: dict) -> list[Path]:
        """
        1. Flag co-occurrence heatmap — reveals correlated quality problems.
        2. Hard flag dominance bar — shows proportion of EQS=0 windows.
        3. Over-triggering analysis — flags whose rates exceed 30%.
        """
        paths = []
        cooccurrence = np.asarray(metrics.get("flag_cooccurrence", []))
        flag_counts  = metrics.get("flag_counts", {})
        n_total      = int(metrics.get("n_windows_total", 1))
        detected     = metrics.get("failure_modes", [])

        fig, axes = make_figure(1, 3, title="Quality Flags — Failure Mode Diagnostics",
                                width_per_col=4.2, height_per_row=4.0)
        ax_heatmap, ax_hard, ax_overtrigger = axes

        # ── 1. Co-occurrence heatmap ─────────────────────────────────────────
        if cooccurrence.size and cooccurrence.shape == (9, 9):
            # Normalize to conditional rate: P(flag_j | flag_i triggered)
            row_sums = cooccurrence.sum(axis=1, keepdims=True)
            norm = cooccurrence / (row_sums + 1e-9)
            import seaborn as sns
            sns.heatmap(
                norm,
                ax=ax_heatmap,
                xticklabels=_ALL_FLAGS,
                yticklabels=_ALL_FLAGS,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                linewidths=0.3,
                linecolor="white",
                annot_kws={"size": 7},
            )
            ax_heatmap.set_title("Flag Co-occurrence\n(row-normalized)", fontsize=FONT_SIZE_TITLE)
            ax_heatmap.tick_params(axis="x", labelsize=7, rotation=45)
            ax_heatmap.tick_params(axis="y", labelsize=7, rotation=0)

            # Detect highly correlated pairs
            upper = np.triu(norm, k=1)
            if upper.max() > 0.6:
                r, c = np.unravel_index(upper.argmax(), upper.shape)
                annotate_failure(
                    ax_heatmap,
                    f"Correlated: {_ALL_FLAGS[r]} ↔ {_ALL_FLAGS[c]} ({upper.max():.2f})"
                )
        else:
            ax_heatmap.set_visible(False)

        # ── 2. Hard flag dominance ────────────────────────────────────────────
        if flag_counts:
            hard_triggered = sum(flag_counts.get(f, 0) for f in _HARD_FLAGS)
            soft_only      = max(0, sum(flag_counts.get(f, 0) for f in _ALL_FLAGS
                                        if f not in _HARD_FLAGS) - hard_triggered)
            clean          = max(0, n_total - hard_triggered - soft_only)

            wedges = [hard_triggered, soft_only, clean]
            labels = ["Hard flag (EQS=0)", "Soft-flag only", "Clean"]
            colors = [COLOR_REJECTED, COLOR_WARN, COLOR_ACCEPTED]
            non_zero = [(w, l, c) for w, l, c in zip(wedges, labels, colors) if w > 0]
            if non_zero:
                ws, ls, cs = zip(*non_zero)
                ax_hard.pie(ws, labels=ls, colors=cs, autopct="%1.1f%%",
                            textprops={"fontsize": 8}, startangle=90)
                ax_hard.set_title("Window Disposition\n(Hard / Soft / Clean)",
                                  fontsize=FONT_SIZE_TITLE)

                hard_rate = hard_triggered / max(n_total, 1)
                if hard_rate > 0.3 or "hard_flag_dominance" in detected:
                    annotate_failure(ax_hard, f"Hard flags dominate ({hard_rate*100:.1f}%)")
        else:
            ax_hard.set_visible(False)

        # ── 3. Over-triggering analysis ───────────────────────────────────────
        if flag_counts and n_total > 0:
            flags  = _ALL_FLAGS
            rates  = [flag_counts.get(f, 0) / n_total * 100 for f in flags]
            colors = [
                COLOR_REJECTED if r > 30 else (COLOR_WARN if r > 15 else COLOR_ACCEPTED)
                for r in rates
            ]
            bars = ax_overtrigger.bar(flags, rates, color=colors,
                                      edgecolor="white", linewidth=0.4)
            ax_overtrigger.axhline(30.0, color=COLOR_REJECTED, linewidth=1.2,
                                   linestyle="--", label="30% over-trigger threshold")
            ax_overtrigger.axhline(15.0, color=COLOR_WARN, linewidth=1.0,
                                   linestyle=":", label="15% warning threshold")
            ax_overtrigger.set_ylabel("Trigger Rate (%)", fontsize=FONT_SIZE_LABEL)
            ax_overtrigger.set_title("Over-Triggering Analysis", fontsize=FONT_SIZE_TITLE)
            ax_overtrigger.tick_params(axis="x", labelsize=7, rotation=35)
            ax_overtrigger.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
            ax_overtrigger.legend(fontsize=7)
            ax_overtrigger.set_ylim(0, max(max(rates) * 1.2, 35))

            overtriggered = [f for f, r in zip(flags, rates) if r > 30]
            if overtriggered or "over_triggering" in detected:
                annotate_failure(ax_overtrigger, f"Over-triggered: {', '.join(overtriggered)}")
        else:
            ax_overtrigger.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "03_failure_modes.png"))
        return paths


# ── CLI entry point ────────────────────────────────────────────────────────────

def _metrics_from_manifest(manifest_path: str) -> dict:
    """Build debug metrics from manifest.csv quality_flags / bitmask columns."""
    import pandas as pd
    rng = np.random.default_rng(_RNG_SEED)
    df  = pd.read_csv(manifest_path)

    flag_counts = {}
    bitmasks    = []

    if "quality_flags" in df.columns:
        for cell in df["quality_flags"].dropna():
            flags = [f.strip() for f in str(cell).split(",") if f.strip()]
            for f in flags:
                flag_counts[f] = flag_counts.get(f, 0) + 1

    if "quality_bitmask" in df.columns:
        bitmasks = df["quality_bitmask"].dropna().astype(int).values

    # Synthesize signal distributions for demo
    flag_signals = {
        "BLUR":      rng.normal(150, 40, 500),
        "STALL":     rng.exponential(0.05, 500),
        "JITTER_A":  rng.gamma(2, 0.01, 500),
        "EXPO_S":    rng.normal(0.08, 0.02, 500),
        "JITTER_V":  rng.normal(0.5, 0.15, 500),
    }
    flag_thresholds = {
        "BLUR":      100.0,
        "STALL":     0.02,
        "JITTER_A":  0.03,
        "EXPO_S":    0.1,
        "JITTER_V":  0.8,
    }
    cooccurrence = rng.integers(0, 50, (9, 9)).astype(float)
    np.fill_diagonal(cooccurrence, 0)

    return {
        "flag_counts":       flag_counts,
        "bitmask_values":    bitmasks,
        "flag_signals":      flag_signals,
        "flag_thresholds":   flag_thresholds,
        "flag_cooccurrence": cooccurrence,
        "flag_types":        {f: ("hard" if f in _HARD_FLAGS else "soft") for f in _ALL_FLAGS},
        "n_windows_total":   len(df),
        "failure_modes":     [],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone quality flag visualizer (debug mode)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-manifest", metavar="PATH",
                        help="Load flag data from manifest.csv")
    source.add_argument("--from-json",     metavar="PATH",
                        help="Load metrics from JSON sidecar")
    parser.add_argument("--output-dir", default="data_processed/plots/quality")
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json) as f:
            metrics = json.load(f)
        for k in ["bitmask_values", "flag_cooccurrence"]:
            if k in metrics:
                metrics[k] = np.asarray(metrics[k])
        for k in (metrics.get("flag_signals") or {}):
            metrics["flag_signals"][k] = np.asarray(metrics["flag_signals"][k])
    else:
        metrics = _metrics_from_manifest(args.from_manifest)

    class _DummyConfig:
        pass

    viz   = StageVisualizer(output_dir=Path(args.output_dir), config=_DummyConfig())
    saved = viz.plot(metrics)
    for p in saved:
        print(f"  Saved: {p}")