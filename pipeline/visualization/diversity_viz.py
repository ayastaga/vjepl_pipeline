"""
pipeline/visualization/diversity_viz.py

Dataset-Level Distribution & Diversity Visualization
======================================================
Produces plots to: data_processed/plots/diversity/

Metrics dict contract:
    embeddings           : np.ndarray   — (N, D) latent embeddings (pre-computed)
    tsne_coords          : np.ndarray   — (N, 2) pre-computed t-SNE coordinates (optional)
    episode_labels       : np.ndarray   — (N,) integer episode index per window
    inter_episode_distances : np.ndarray — (E, E) pairwise mean embedding distance
    proxy_fid            : float        — proxy FID score (lower=more diverse)
    env_drift_timeline   : np.ndarray   — per-episode mean embedding shift
    episode_ids          : list[str]
    failure_modes        : list[str]

CLI:
    python -m pipeline.visualization.diversity_viz --from-manifest data_processed/manifest.csv
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
    annotate_failure, make_figure,
)

_RNG_SEED = 42


class StageVisualizer(BaseStageVisualizer):
    STAGE_NAME = "diversity"

    def plot_primary_diagnostics(self, metrics: dict) -> list[Path]:
        """
        t-SNE embedding projection coloured by episode.
        """
        paths = []
        tsne_coords   = np.asarray(metrics.get("tsne_coords", []))
        episode_labels = np.asarray(metrics.get("episode_labels", []))

        if not tsne_coords.size or tsne_coords.ndim != 2:
            # Try to compute t-SNE from raw embeddings
            embeddings = np.asarray(metrics.get("embeddings", []))
            if not embeddings.size or embeddings.ndim != 2:
                return paths
            # Lightweight PCA projection as fallback (no sklearn needed)
            cov   = np.cov(embeddings.T)
            vals, vecs = np.linalg.eigh(cov)
            idx   = np.argsort(vals)[::-1][:2]
            tsne_coords = embeddings @ vecs[:, idx]

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.suptitle("Dataset Diversity — Embedding Projection (PCA/t-SNE)",
                     fontsize=FONT_SIZE_TITLE + 1, fontweight="bold")

        if episode_labels.size == tsne_coords.shape[0]:
            unique_eps = np.unique(episode_labels)
            cmap = plt.get_cmap("tab20", max(len(unique_eps), 1))
            for i, ep in enumerate(unique_eps[:20]):  # cap at 20 colours
                mask = episode_labels == ep
                ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                           s=5, alpha=0.5, color=cmap(i), linewidths=0,
                           label=f"ep {ep}" if len(unique_eps) <= 10 else None)
            if len(unique_eps) <= 10:
                ax.legend(fontsize=7, markerscale=2)
        else:
            ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                       s=5, alpha=0.5, color=COLOR_INFO, linewidths=0)

        ax.set_xlabel("Dim 1", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Dim 2", fontsize=FONT_SIZE_LABEL)
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        fig.tight_layout()
        paths.append(self._save(fig, "01_embedding_projection.png"))
        return paths

    def plot_distributions(self, metrics: dict) -> list[Path]:
        """
        Inter-episode distance heatmap + proxy FID annotation.
        """
        paths = []
        dist_matrix = np.asarray(metrics.get("inter_episode_distances", []))
        proxy_fid   = metrics.get("proxy_fid", None)
        env_drift   = np.asarray(metrics.get("env_drift_timeline", []))

        n_panels = int(dist_matrix.size > 0) + int(env_drift.size > 0)
        if n_panels == 0:
            return paths

        fig, axes = make_figure(1, n_panels, title="Diversity — Inter-Episode Analysis")
        flat = np.array(axes).flatten() if n_panels > 1 else [axes]
        idx  = 0

        if dist_matrix.size and dist_matrix.ndim == 2:
            import seaborn as sns
            n = min(dist_matrix.shape[0], 30)
            sns.heatmap(dist_matrix[:n, :n], ax=flat[idx], cmap="Blues",
                        linewidths=0.2, linecolor="white", square=True,
                        cbar_kws={"shrink": 0.7})
            title = "Inter-Episode Embedding Distance"
            if proxy_fid is not None:
                title += f"\nProxy FID = {proxy_fid:.2f}"
            flat[idx].set_title(title, fontsize=FONT_SIZE_TITLE)
            flat[idx].tick_params(labelsize=6)
            # Low off-diagonal distances = low diversity
            off_diag = dist_matrix[~np.eye(n, dtype=bool)]
            if off_diag.size and off_diag.mean() < 0.3:
                annotate_failure(flat[idx], "Low inter-episode diversity")
            idx += 1

        if env_drift.size:
            flat[idx].plot(env_drift, color=COLOR_WARN, linewidth=1.0, alpha=0.85)
            flat[idx].set_xlabel("Episode index", fontsize=FONT_SIZE_LABEL)
            flat[idx].set_ylabel("Mean embedding shift", fontsize=FONT_SIZE_LABEL)
            flat[idx].set_title("Environment Drift Tracker", fontsize=FONT_SIZE_TITLE)
            flat[idx].tick_params(labelsize=FONT_SIZE_TICK)
            drift_range = env_drift.max() - env_drift.min()
            if drift_range > 0.5:
                annotate_failure(flat[idx], f"Large env drift detected (Δ={drift_range:.2f})")
            idx += 1

        fig.tight_layout()
        paths.append(self._save(fig, "02_inter_episode_diversity.png"))
        return paths

    def plot_failure_modes(self, metrics: dict) -> list[Path]:
        """
        Env distribution shift, lighting change, lab layout drift.
        """
        paths = []
        env_drift   = np.asarray(metrics.get("env_drift_timeline", []))
        dist_matrix = np.asarray(metrics.get("inter_episode_distances", []))
        detected    = metrics.get("failure_modes", [])

        fig, axes = make_figure(1, 3, title="Diversity — Failure Mode Diagnostics")
        ax_shift, ax_light, ax_layout = axes

        # 1. Distribution shift: running mean of embedding coordinate
        if env_drift.size > 5:
            window = max(3, len(env_drift) // 10)
            rolling = np.convolve(env_drift, np.ones(window) / window, mode="valid")
            ax_shift.plot(env_drift, color=COLOR_INFO, linewidth=0.7, alpha=0.6,
                          label="raw")
            ax_shift.plot(range(window - 1, len(env_drift)), rolling,
                          color=COLOR_REJECTED, linewidth=1.5, label=f"rolling mean (w={window})")
            ax_shift.set_xlabel("Episode", fontsize=FONT_SIZE_LABEL)
            ax_shift.set_ylabel("Embedding shift", fontsize=FONT_SIZE_LABEL)
            ax_shift.set_title("Environmental Distribution Shift", fontsize=FONT_SIZE_TITLE)
            ax_shift.legend(fontsize=8)
            ax_shift.tick_params(labelsize=FONT_SIZE_TICK)
            if rolling[-1] - rolling[0] > 0.3 or "env_distribution_shift" in detected:
                annotate_failure(ax_shift, "Progressive distribution shift")
        else:
            ax_shift.set_visible(False)

        # 2. Lighting change proxy: sudden jump in env_drift
        if env_drift.size > 5:
            jumps = np.abs(np.diff(env_drift))
            jump_threshold = jumps.mean() + 3 * jumps.std()
            jump_indices   = np.where(jumps > jump_threshold)[0]
            ax_light.plot(env_drift, color=COLOR_WARN, linewidth=0.9)
            for ji in jump_indices:
                ax_light.axvline(ji, color=COLOR_REJECTED, linewidth=1.2,
                                 linestyle=":", alpha=0.7)
            ax_light.set_xlabel("Episode", fontsize=FONT_SIZE_LABEL)
            ax_light.set_ylabel("Embedding shift", fontsize=FONT_SIZE_LABEL)
            ax_light.set_title("Lighting Change Detection\n(sudden embedding jump)",
                               fontsize=FONT_SIZE_TITLE)
            ax_light.tick_params(labelsize=FONT_SIZE_TICK)
            if jump_indices.size or "lighting_change" in detected:
                annotate_failure(ax_light, f"{jump_indices.size} lighting change(s) detected")
        else:
            ax_light.set_visible(False)

        # 3. Lab layout drift: cluster spread over time
        if dist_matrix.size and dist_matrix.ndim == 2:
            # Proxy: mean distance to episode 0 over time
            first_row = dist_matrix[0, :]
            ax_layout.plot(first_row, color=COLOR_INFO, linewidth=1.0)
            ax_layout.set_xlabel("Episode index", fontsize=FONT_SIZE_LABEL)
            ax_layout.set_ylabel("Distance from episode 0", fontsize=FONT_SIZE_LABEL)
            ax_layout.set_title("Lab Layout Drift\n(distance from initial episode embedding)",
                                fontsize=FONT_SIZE_TITLE)
            ax_layout.tick_params(labelsize=FONT_SIZE_TICK)
            if first_row[-1] > first_row.mean() + 2 * first_row.std() or "lab_layout_drift" in detected:
                annotate_failure(ax_layout, "Lab layout drift in late episodes")
        else:
            ax_layout.set_visible(False)

        fig.tight_layout()
        paths.append(self._save(fig, "03_failure_modes.png"))
        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone diversity visualizer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-manifest", metavar="PATH")
    source.add_argument("--from-json",     metavar="PATH")
    parser.add_argument("--output-dir", default="data_processed/plots/diversity")
    args = parser.parse_args()

    rng = np.random.default_rng(_RNG_SEED)
    N, E, D = 300, 20, 32
    emb = rng.normal(0, 1, (N, D))
    labels = np.repeat(np.arange(E), N // E)

    metrics = {
        "embeddings":              emb,
        "episode_labels":          labels,
        "inter_episode_distances": np.abs(rng.normal(0.5, 0.2, (E, E))),
        "proxy_fid":               12.3,
        "env_drift_timeline":      np.cumsum(rng.normal(0, 0.02, E)),
        "failure_modes":           [],
    }

    class _DummyConfig: pass
    viz = StageVisualizer(Path(args.output_dir), _DummyConfig())
    for p in viz.plot(metrics):
        print(f"  Saved: {p}")