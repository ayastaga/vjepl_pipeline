"""
pipeline/visualization/rejection_viz.py

Accepted vs Rejected Example Diagnostics
==========================================
Produces plots to: data_processed/plots/rejection/

Metrics dict contract:
    rejection_reasons    : dict[str, int]  — reason → count
    accepted_features    : dict[str, np.ndarray]  — feature_name → values for accepted
    rejected_features    : dict[str, np.ndarray]  — feature_name → values for rejected
    eqs_accepted         : np.ndarray
    eqs_rejected         : np.ndarray
    hard_flag_breakdown  : dict[str, int]  — hard_flag_name → times it caused rejection
    failure_modes        : list[str]

CLI:
    python -m pipeline.visualization.rejection_viz --from-manifest data_processed/manifest.csv
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
    annotate_failure, make_figure, plot_histogram, plot_box,
)

_RNG_SEED = 42


class StageVisualizer(BaseStageVisualizer):
    STAGE_NAME = "rejection"

    def plot_primary_diagnostics(self, metrics: dict) -> list[Path]:
        """
        Panel A: Rejection reason pie chart.
        Panel B: EQS distribution for accepted vs rejected.
        """
        paths = []
        rejection_reasons = metrics.get("rejection_reasons", {})
        eqs_acc = np.asarray(metrics.get("eqs_accepted", []))
        eqs_rej = np.asarray(metrics.get("eqs_rejected", []))

        fig, axes = make_figure(1, 2, title="Rejection Diagnostics — Overview")
        ax_pie, ax_eqs = axes

        # Pie chart
        if rejection_reasons:
            labels = list(rejection_reasons.keys())
            counts = list(rejection_reasons.values())
            total  = sum(counts)
            colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax_pie.pie(
                counts, labels=labels, colors=colors, autopct="%1.1f%%",
                textprops={"fontsize": 8}, startangle=90,
            )
            ax_pie.set_title(f"Rejection Reasons\n(total rejected: {total})",
                             fontsize=FONT_SIZE_TITLE)

            # Hard flag dominance
            hard_reasons = [r for r in labels if r in {"MISSING_V", "DROP_P"}]
            hard_counts  = sum(rejection_reasons.get(r, 0) for r in hard_reasons)
            if hard_counts / max(total, 1) > 0.6:
                annotate_failure(ax_pie, f"Hard flag dominance ({hard_counts/total*100:.1f}%)")
        else:
            ax_pie.set_visible(False)

        # EQS comparison
        if eqs_acc.size or eqs_rej.size:
            bins = np.linspace(0, 1, 51)
            if eqs_acc.size:
                ax_eqs.hist(eqs_acc, bins=bins, color=COLOR_ACCEPTED, alpha=0.7,
                            label=f"Accepted (n={len(eqs_acc)})", edgecolor="white",
                            linewidth=0.3)
            if eqs_rej.size:
                ax_eqs.hist(eqs_rej, bins=bins, color=COLOR_REJECTED, alpha=0.7,
                            label=f"Rejected (n={len(eqs_rej)})", edgecolor="white",
                            linewidth=0.3)
            ax_eqs.set_xlabel("EQS", fontsize=FONT_SIZE_LABEL)
            ax_eqs.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
            ax_eqs.set_title("EQS: Accepted vs Rejected", fontsize=FONT_SIZE_TITLE)
            ax_eqs.legend(fontsize=8)
            ax_eqs.tick_params(labelsize=FONT_SIZE_TICK)
        else:
            ax_eqs.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "01_rejection_overview.png"))
        return paths

    def plot_distributions(self, metrics: dict) -> list[Path]:
        """
        Accepted vs rejected feature boxplots for each feature dimension.
        """
        paths = []
        acc_features = metrics.get("accepted_features", {})
        rej_features = metrics.get("rejected_features", {})

        shared_keys = [k for k in acc_features if k in rej_features]
        if not shared_keys:
            return paths

        n   = len(shared_keys)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.0))
        fig.suptitle("Accepted vs Rejected — Feature Distributions",
                     fontsize=FONT_SIZE_TITLE + 1, fontweight="bold")
        flat = np.array(axes).flatten()

        for i, key in enumerate(shared_keys):
            ax  = flat[i]
            acc = np.asarray(acc_features[key])
            rej = np.asarray(rej_features[key])
            ax.boxplot([acc[np.isfinite(acc)], rej[np.isfinite(rej)]],
                       labels=["Accepted", "Rejected"],
                       patch_artist=True,
                       boxprops={"facecolor": COLOR_ACCEPTED, "alpha": 0.6},
                       medianprops={"color": COLOR_WARN, "linewidth": 1.5},
                       flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
            # Colour rejected box differently
            try:
                boxes = ax.findobj(plt.matplotlib.patches.PathPatch)
                if len(boxes) >= 2:
                    boxes[1].set_facecolor(COLOR_REJECTED)
                    boxes[1].set_alpha(0.6)
            except Exception:
                pass
            ax.set_title(key, fontsize=FONT_SIZE_TICK + 1, fontweight="bold")
            ax.tick_params(labelsize=7)

        for j in range(len(shared_keys), len(flat)):
            flat[j].set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "02_feature_boxplots.png"))
        return paths

    def plot_failure_modes(self, metrics: dict) -> list[Path]:
        """
        Over-rejection, hard flag dominance, threshold miscalibration.
        """
        paths = []
        rejection_reasons   = metrics.get("rejection_reasons", {})
        hard_flag_breakdown = metrics.get("hard_flag_breakdown", {})
        eqs_acc  = np.asarray(metrics.get("eqs_accepted", []))
        eqs_rej  = np.asarray(metrics.get("eqs_rejected", []))
        detected = metrics.get("failure_modes", [])
        threshold = float(metrics.get("acceptance_threshold", 0.6))

        fig, axes = make_figure(1, 3, title="Rejection — Failure Mode Diagnostics")
        ax_over, ax_hard, ax_thresh = axes

        # 1. Over-rejection: rejection rate
        total_acc = len(eqs_acc)
        total_rej = len(eqs_rej)
        total     = max(total_acc + total_rej, 1)
        rej_rate  = total_rej / total

        ax_over.bar(["Accepted", "Rejected"], [total_acc, total_rej],
                    color=[COLOR_ACCEPTED, COLOR_REJECTED],
                    edgecolor="white", linewidth=0.4)
        ax_over.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
        ax_over.set_title(f"Over-Rejection Check\n(rejection rate={rej_rate*100:.1f}%)",
                          fontsize=FONT_SIZE_TITLE)
        ax_over.tick_params(labelsize=FONT_SIZE_TICK)
        if rej_rate > 0.5 or "over_rejection" in detected:
            annotate_failure(ax_over, f"Over-rejection: {rej_rate*100:.1f}% rejected")

        # 2. Hard flag breakdown bar
        if hard_flag_breakdown:
            flags  = list(hard_flag_breakdown.keys())
            counts = list(hard_flag_breakdown.values())
            ax_hard.bar(flags, counts, color=COLOR_REJECTED, edgecolor="white", linewidth=0.4)
            ax_hard.set_ylabel("Rejection count", fontsize=FONT_SIZE_LABEL)
            ax_hard.set_title("Hard Flag Rejection Breakdown", fontsize=FONT_SIZE_TITLE)
            ax_hard.tick_params(axis="x", labelsize=FONT_SIZE_TICK, rotation=20)
            ax_hard.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
            dom_flag = max(hard_flag_breakdown, key=hard_flag_breakdown.get)
            dom_frac = hard_flag_breakdown[dom_flag] / max(sum(counts), 1)
            if dom_frac > 0.7 or "hard_flag_dominance" in detected:
                annotate_failure(ax_hard, f"{dom_flag} dominates ({dom_frac*100:.1f}%)")
        else:
            ax_hard.set_visible(False)

        # 3. Threshold miscalibration: what happens at ±0.1 threshold shift
        if eqs_acc.size or eqs_rej.size:
            all_eqs = np.concatenate([eqs_acc, eqs_rej]) if eqs_acc.size and eqs_rej.size else (
                eqs_acc if eqs_acc.size else eqs_rej
            )
            deltas = np.linspace(-0.2, 0.2, 41)
            rej_rates = [(all_eqs < threshold + d).mean() * 100 for d in deltas]

            ax_thresh.plot(deltas, rej_rates, color=COLOR_REJECTED, linewidth=1.5)
            ax_thresh.axvline(0, color="grey", linestyle=":", linewidth=1.0)
            ax_thresh.axhline(50.0, color=COLOR_WARN, linestyle="--",
                              linewidth=1.0, label="50% rejection rate")
            ax_thresh.set_xlabel("Threshold shift (Δ)", fontsize=FONT_SIZE_LABEL)
            ax_thresh.set_ylabel("Rejection rate (%)", fontsize=FONT_SIZE_LABEL)
            ax_thresh.set_title("Threshold Miscalibration\n(sensitivity analysis)",
                                fontsize=FONT_SIZE_TITLE)
            ax_thresh.legend(fontsize=8)
            ax_thresh.tick_params(labelsize=FONT_SIZE_TICK)
            slope = abs(rej_rates[-1] - rej_rates[0]) / (deltas[-1] - deltas[0])
            if slope > 200 or "threshold_miscalibration" in detected:
                annotate_failure(ax_thresh, f"High threshold sensitivity (slope={slope:.0f}%/unit)")
        else:
            ax_thresh.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "03_failure_modes.png"))
        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone rejection visualizer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-manifest", metavar="PATH")
    source.add_argument("--from-json",     metavar="PATH")
    parser.add_argument("--output-dir", default="data_processed/plots/rejection")
    args = parser.parse_args()

    rng = np.random.default_rng(_RNG_SEED)
    N = 400
    eqs = rng.beta(4, 2, N)
    threshold = 0.6

    metrics = {
        "rejection_reasons":  {"MISSING_V": 30, "DROP_P": 15, "low_eqs": 55},
        "accepted_features":  {"blur": rng.normal(180, 30, int(N * 0.7)),
                               "stall": rng.exponential(0.08, int(N * 0.7))},
        "rejected_features":  {"blur": rng.normal(80, 20, int(N * 0.3)),
                               "stall": rng.exponential(0.2, int(N * 0.3))},
        "eqs_accepted":       eqs[eqs >= threshold],
        "eqs_rejected":       eqs[eqs < threshold],
        "hard_flag_breakdown": {"MISSING_V": 30, "DROP_P": 15},
        "acceptance_threshold": threshold,
        "failure_modes":      [],
    }

    class _DummyConfig: pass
    viz = StageVisualizer(Path(args.output_dir), _DummyConfig())
    for p in viz.plot(metrics):
        print(f"  Saved: {p}")