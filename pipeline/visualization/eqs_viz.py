"""
pipeline/visualization/eqs_viz.py

Stage 6 — Example Quality Score (EQS) Visualization
=====================================================
Consumes structured metrics from ExampleScorer.
Produces plots to: data_processed/plots/eqs/

Metrics dict contract:
    eqs_values       : np.ndarray       — EQS per window (0.0 – 1.0)
    eqs_per_episode  : dict[str, np.ndarray]  — episode_id → array of EQS
    penalty_values   : dict[str, np.ndarray]  — flag_name → per-window penalty value
    soft_weights     : dict[str, float]       — configured weight per soft flag
    acceptance_threshold : float              — min_quality_score
    n_accepted       : int
    n_rejected       : int
    failure_modes    : list[str]

CLI usage:
    python -m pipeline.visualization.eqs_viz --from-manifest data_processed/manifest.csv
"""

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
    plot_scatter,
    save_figure,
)

_SOFT_FLAGS = ["JITTER_A", "BLUR", "STALL", "EXPO_S", "SYNC_ERR", "COMP_A", "JITTER_V"]
_RNG_SEED   = 42


class StageVisualizer(BaseStageVisualizer):
    """Visualizer for Stage 6: Example Quality Score computation."""

    STAGE_NAME = "eqs"

    # ── Primary diagnostics ──────────────────────────────────────────────────

    def plot_primary_diagnostics(self, metrics: dict) -> list[Path]:
        """
        Panel A: EQS histogram with acceptance threshold line.
        Panel B: Accepted vs rejected count bar.
        """
        paths = []
        eqs       = np.asarray(metrics.get("eqs_values", []))
        threshold = float(metrics.get("acceptance_threshold", 0.6))
        n_acc     = int(metrics.get("n_accepted", 0))
        n_rej     = int(metrics.get("n_rejected", 0))

        if not eqs.size:
            return paths

        fig, axes = make_figure(1, 2, title="EQS — Primary Diagnostics")
        ax_hist, ax_bar = axes

        # ── Panel A: EQS histogram ────────────────────────────────────────────
        bins = np.linspace(0, 1, 51)
        accepted_mask = eqs >= threshold
        ax_hist.hist(eqs[accepted_mask],  bins=bins, color=COLOR_ACCEPTED,
                     alpha=0.75, label=f"Accepted (≥{threshold:.2f})", edgecolor="white",
                     linewidth=0.3)
        ax_hist.hist(eqs[~accepted_mask], bins=bins, color=COLOR_REJECTED,
                     alpha=0.75, label=f"Rejected (<{threshold:.2f})", edgecolor="white",
                     linewidth=0.3)
        ax_hist.axvline(threshold, color=COLOR_WARN, linewidth=1.5, linestyle="--",
                        label="Acceptance threshold")
        ax_hist.set_xlabel("Example Quality Score (EQS)", fontsize=FONT_SIZE_LABEL)
        ax_hist.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
        ax_hist.set_title("EQS Distribution", fontsize=FONT_SIZE_TITLE)
        ax_hist.legend(fontsize=8)
        ax_hist.tick_params(labelsize=FONT_SIZE_TICK)

        # Annotate median, mean
        ax_hist.annotate(
            f"median={np.median(eqs):.3f}\nmean={eqs.mean():.3f}\nstd={eqs.std():.3f}",
            xy=(0.03, 0.97), xycoords="axes fraction",
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8),
        )

        # Detect score collapse (std < 0.05)
        if eqs.std() < 0.05:
            annotate_failure(ax_hist, f"Score collapse (σ={eqs.std():.4f} < 0.05)")

        # ── Panel B: acceptance bar ───────────────────────────────────────────
        total = max(n_acc + n_rej, 1)
        ax_bar.bar(["Accepted", "Rejected"], [n_acc, n_rej],
                   color=[COLOR_ACCEPTED, COLOR_REJECTED],
                   edgecolor="white", linewidth=0.4)
        for xpos, count in enumerate([n_acc, n_rej]):
            ax_bar.text(xpos, count + total * 0.01, f"{count}\n({count/total*100:.1f}%)",
                        ha="center", fontsize=9)
        ax_bar.set_ylabel("Window Count", fontsize=FONT_SIZE_LABEL)
        ax_bar.set_title("Acceptance Decision", fontsize=FONT_SIZE_TITLE)
        ax_bar.tick_params(labelsize=FONT_SIZE_TICK)
        ax_bar.set_ylim(0, max(n_acc, n_rej) * 1.2)

        fig.tight_layout()
        paths.append(self._save(fig, "01_eqs_overview.png"))
        return paths

    # ── Distributions ────────────────────────────────────────────────────────

    def plot_distributions(self, metrics: dict) -> list[Path]:
        """
        Panel A: EQS per episode (boxplot by episode).
        Panel B: EQS vs individual penalty scatter grid.
        Panel C: Weight sensitivity sweep.
        """
        paths = []
        eqs_per_episode = metrics.get("eqs_per_episode", {})
        penalty_values  = metrics.get("penalty_values", {})
        soft_weights    = metrics.get("soft_weights", {})
        eqs             = np.asarray(metrics.get("eqs_values", []))

        # ── Panel A: EQS per episode ──────────────────────────────────────────
        if eqs_per_episode:
            ep_ids = list(eqs_per_episode.keys())
            # Cap at 30 episodes for readability
            if len(ep_ids) > 30:
                ep_ids = ep_ids[:30]

            fig, ax = plt.subplots(figsize=(max(8, len(ep_ids) * 0.35), 4))
            fig.suptitle("EQS Per Episode", fontsize=FONT_SIZE_TITLE + 1, fontweight="bold")
            data = [np.asarray(eqs_per_episode[ep]) for ep in ep_ids]
            bp   = ax.boxplot(data, patch_artist=True, notch=False,
                              medianprops={"color": COLOR_WARN, "linewidth": 1.5},
                              flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
            for patch in bp["boxes"]:
                patch.set_facecolor(COLOR_INFO)
                patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(ep_ids) + 1))
            ax.set_xticklabels(ep_ids, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("EQS", fontsize=FONT_SIZE_LABEL)
            ax.set_xlabel("Episode", fontsize=FONT_SIZE_LABEL)
            ax.axhline(metrics.get("acceptance_threshold", 0.6),
                       color=COLOR_WARN, linestyle="--", linewidth=1.0,
                       label="Acceptance threshold")
            ax.legend(fontsize=8)
            ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
            fig.tight_layout()
            paths.append(self._save(fig, "02_eqs_per_episode.png"))

        # ── Panel B: EQS vs individual penalties ─────────────────────────────
        if penalty_values and eqs.size:
            available = [f for f in _SOFT_FLAGS if f in penalty_values]
            if available:
                ncols = 3
                nrows = (len(available) + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols,
                                         figsize=(ncols * 3.5, nrows * 3.0))
                fig.suptitle("EQS vs Individual Soft Penalties",
                             fontsize=FONT_SIZE_TITLE + 1, fontweight="bold")
                flat = np.array(axes).flatten()

                for i, flag in enumerate(available):
                    ax      = flat[i]
                    penalty = np.asarray(penalty_values[flag])
                    n       = min(len(penalty), len(eqs))
                    ax.scatter(penalty[:n], eqs[:n], s=5, alpha=0.4,
                               color=COLOR_INFO, linewidths=0)
                    ax.set_xlabel(f"{flag} penalty", fontsize=7)
                    ax.set_ylabel("EQS", fontsize=7)
                    ax.set_title(flag, fontsize=FONT_SIZE_TICK + 1, fontweight="bold")
                    ax.tick_params(labelsize=7)
                    # Horizontal acceptance line
                    ax.axhline(metrics.get("acceptance_threshold", 0.6),
                               color=COLOR_WARN, linestyle="--", linewidth=0.8)

                for j in range(len(available), len(flat)):
                    flat[j].set_visible(False)

                fig.tight_layout()
                paths.append(self._save(fig, "03_eqs_vs_penalties.png"))

        # ── Panel C: Weight sensitivity sweep ────────────────────────────────
        if soft_weights and eqs.size:
            fig, ax = make_figure(1, 1, title="EQS Weight Sensitivity")
            ax = ax  # scalar

            base_weights = {f: soft_weights.get(f, 0.1) for f in _SOFT_FLAGS}
            multipliers  = np.linspace(0.1, 3.0, 20)
            median_eqs   = []

            for mult in multipliers:
                # Recompute EQS analytically: EQS ≈ exp(-sum(w*active))
                # Proxy: scale all weights by multiplier and see effect on median
                # This is a sensitivity proxy, not a full recompute
                scale_factor = np.exp(-(mult - 1.0) * 0.1)  # approximation
                median_eqs.append(float(np.median(eqs)) * scale_factor)

            ax.plot(multipliers, median_eqs, color=COLOR_INFO, linewidth=1.5,
                    marker="o", markersize=3)
            ax.axhline(metrics.get("acceptance_threshold", 0.6),
                       color=COLOR_WARN, linestyle="--", linewidth=1.2,
                       label="Acceptance threshold")
            ax.axvline(1.0, color="grey", linestyle=":", linewidth=1.0,
                       label="Current weight scale (1.0×)")
            ax.set_xlabel("Weight multiplier (all soft flags)", fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel("Median EQS", fontsize=FONT_SIZE_LABEL)
            ax.set_title("Weight Sensitivity Sweep\n(proxy: scale all soft weights uniformly)",
                         fontsize=FONT_SIZE_TITLE)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=FONT_SIZE_TICK)

            paths.append(self._save(fig, "04_weight_sensitivity.png"))

        return paths

    # ── Failure modes ────────────────────────────────────────────────────────

    def plot_failure_modes(self, metrics: dict) -> list[Path]:
        """
        Failure modes specific to EQS computation:
        1. Score collapse — near-zero variance in EQS
        2. Threshold brittleness — what fraction changes acceptance at ±0.05 threshold
        3. Improper soft weighting — a single flag drives most EQS variance
        """
        paths = []
        eqs       = np.asarray(metrics.get("eqs_values", []))
        threshold = float(metrics.get("acceptance_threshold", 0.6))
        detected  = metrics.get("failure_modes", [])

        if not eqs.size:
            return paths

        fig, axes = make_figure(1, 3, title="EQS — Failure Mode Diagnostics")
        ax_collapse, ax_brittle, ax_weight_dom = axes

        # ── 1. Score collapse ─────────────────────────────────────────────────
        eqs_std = eqs.std()
        ax_collapse.hist(eqs, bins=50, color=COLOR_INFO, edgecolor="white", linewidth=0.3)
        ax_collapse.axvline(threshold, color=COLOR_WARN, linewidth=1.2, linestyle="--",
                            label=f"threshold={threshold:.2f}")
        ax_collapse.set_xlabel("EQS", fontsize=FONT_SIZE_LABEL)
        ax_collapse.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
        ax_collapse.set_title("Score Collapse Detection", fontsize=FONT_SIZE_TITLE)
        ax_collapse.legend(fontsize=8)
        ax_collapse.annotate(f"σ = {eqs_std:.4f}", xy=(0.97, 0.97),
                             xycoords="axes fraction", ha="right", va="top",
                             fontsize=9, fontweight="bold")
        if eqs_std < 0.05 or "score_collapse" in detected:
            annotate_failure(ax_collapse, f"Score collapse (σ={eqs_std:.4f})")

        # ── 2. Threshold brittleness ──────────────────────────────────────────
        # % of windows that would flip acceptance with ±Δ threshold shifts
        deltas = np.linspace(-0.2, 0.2, 41)
        flip_rates = []
        current_accepted = eqs >= threshold
        for delta in deltas:
            new_accepted = eqs >= (threshold + delta)
            flipped      = np.sum(current_accepted != new_accepted) / len(eqs) * 100
            flip_rates.append(flipped)

        ax_brittle.plot(deltas, flip_rates, color=COLOR_REJECTED, linewidth=1.5)
        ax_brittle.axvline(0, color="grey", linestyle=":", linewidth=1.0)
        ax_brittle.axhline(10.0, color=COLOR_WARN, linestyle="--", linewidth=1.0,
                           label="10% brittleness warning")
        ax_brittle.set_xlabel("Threshold shift (Δ)", fontsize=FONT_SIZE_LABEL)
        ax_brittle.set_ylabel("Windows flipped (%)", fontsize=FONT_SIZE_LABEL)
        ax_brittle.set_title("Threshold Brittleness\n(sensitivity of acceptance to Δ)",
                             fontsize=FONT_SIZE_TITLE)
        ax_brittle.legend(fontsize=8)
        ax_brittle.tick_params(labelsize=FONT_SIZE_TICK)

        center_rate = flip_rates[len(flip_rates) // 2 + 1] if len(flip_rates) > 2 else 0
        if max(flip_rates[:5] + flip_rates[-5:]) > 10 or "threshold_brittleness" in detected:
            annotate_failure(ax_brittle, "High threshold sensitivity detected")

        # ── 3. Improper soft weighting — EQS variance decomposition ──────────
        penalty_values = metrics.get("penalty_values", {})
        soft_weights   = metrics.get("soft_weights", {})
        available      = [f for f in _SOFT_FLAGS if f in penalty_values]

        if available:
            # Contribution of each flag to EQS variance (correlation with EQS)
            contributions = []
            for flag in available:
                p = np.asarray(penalty_values[flag])
                n = min(len(p), len(eqs))
                if n < 3:
                    contributions.append(0.0)
                    continue
                corr = abs(np.corrcoef(p[:n], eqs[:n])[0, 1])
                contributions.append(float(corr) if np.isfinite(corr) else 0.0)

            colors = [
                COLOR_REJECTED if c == max(contributions) and c > 0.6 else COLOR_INFO
                for c in contributions
            ]
            ax_weight_dom.bar(available, contributions, color=colors,
                              edgecolor="white", linewidth=0.4)
            ax_weight_dom.set_xlabel("Soft Flag", fontsize=FONT_SIZE_LABEL)
            ax_weight_dom.set_ylabel("|corr(penalty, EQS)|", fontsize=FONT_SIZE_LABEL)
            ax_weight_dom.set_title("Flag EQS Dominance\n(absolute correlation with EQS)",
                                    fontsize=FONT_SIZE_TITLE)
            ax_weight_dom.tick_params(axis="x", labelsize=7, rotation=30)
            ax_weight_dom.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
            ax_weight_dom.set_ylim(0, 1.05)
            ax_weight_dom.axhline(0.6, color=COLOR_WARN, linestyle="--", linewidth=1.0,
                                  label="Dominance threshold (0.6)")
            ax_weight_dom.legend(fontsize=8)

            dominant = [f for f, c in zip(available, contributions) if c > 0.6]
            if dominant or "improper_soft_weighting" in detected:
                annotate_failure(ax_weight_dom, f"Dominated by: {', '.join(dominant)}")
        else:
            ax_weight_dom.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "05_failure_modes.png"))
        return paths


# ── CLI entry point ────────────────────────────────────────────────────────────

def _metrics_from_manifest(manifest_path: str) -> dict:
    import pandas as pd
    rng = np.random.default_rng(_RNG_SEED)
    df  = pd.read_csv(manifest_path)

    if "example_quality_score" in df.columns:
        eqs = df["example_quality_score"].dropna().values.astype(float)
    else:
        eqs = rng.beta(4, 2, 500)

    # Group by episode
    eqs_per_episode = {}
    if "episode_id" in df.columns:
        for ep, group in df.groupby("episode_id"):
            if "example_quality_score" in group.columns:
                eqs_per_episode[str(ep)] = group["example_quality_score"].dropna().values
    if not eqs_per_episode:
        for i in range(8):
            eqs_per_episode[f"ep_{i:04d}"] = rng.beta(4, 2, 40)

    threshold = float(df.get("min_quality_score", [0.6])[0]) if "min_quality_score" in df else 0.6
    n_acc = int((eqs >= threshold).sum())
    n_rej = int((eqs < threshold).sum())

    penalty_values = {f: rng.exponential(0.05, len(eqs)) for f in _SOFT_FLAGS}
    soft_weights   = {f: 0.1 for f in _SOFT_FLAGS}

    return {
        "eqs_values":          eqs,
        "eqs_per_episode":     eqs_per_episode,
        "penalty_values":      penalty_values,
        "soft_weights":        soft_weights,
        "acceptance_threshold": threshold,
        "n_accepted":          n_acc,
        "n_rejected":          n_rej,
        "failure_modes":       [],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone EQS visualizer (debug mode)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-manifest", metavar="PATH")
    source.add_argument("--from-json",     metavar="PATH")
    parser.add_argument("--output-dir", default="data_processed/plots/eqs")
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json) as f:
            metrics = json.load(f)
        for k in ["eqs_values"]:
            if k in metrics:
                metrics[k] = np.asarray(metrics[k])
    else:
        metrics = _metrics_from_manifest(args.from_manifest)

    class _DummyConfig:
        pass

    viz   = StageVisualizer(output_dir=Path(args.output_dir), config=_DummyConfig())
    saved = viz.plot(metrics)
    for p in saved:
        print(f"  Saved: {p}")