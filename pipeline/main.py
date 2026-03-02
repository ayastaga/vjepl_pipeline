import sys
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Make sure pipeline/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.log_generator import LogGenerator
from pipeline.validator import EpisodeValidator
from pipeline.aligner import DriftCorrector, TimestampAligner
from pipeline.sampler import WindowSampler
from pipeline.quality import QualityFlagger
from pipeline.scorer import ExampleScorer
from pipeline.exporter import DatasetExporter


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict):
    plots_dir = Path("data_processed/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # STAGE 1: GENERATE RAW LOGS
    # ============================================================
    print("\n" + "="*60)
    print("STAGE 1: Generating synthetic robot logs...")
    print("="*60)
    generator = LogGenerator(config, out_dir="data_raw")
    episode_dirs = generator.generate_all()

    # ============================================================
    # STAGE 2: VALIDATE EPISODES
    # ============================================================
    print("\n" + "="*60)
    print("STAGE 2: Validating episodes...")
    print("="*60)
    validator = EpisodeValidator(config)
    valid_episodes = validator.validate_all(episode_dirs)
    print(f"  -> {len(valid_episodes)}/{len(episode_dirs)} episodes passed validation")

    # ============================================================
    # STAGE 3-7: PROCESS EACH EPISODE
    # ============================================================
    print("\n" + "="*60)
    print("STAGES 3-7: Aligning, sampling, flagging, scoring...")
    print("="*60)

    drift_corrector = DriftCorrector()
    aligner = TimestampAligner(config)
    sampler = WindowSampler(config)
    flagger = QualityFlagger(config)
    scorer = ExampleScorer(config)
    exporter = DatasetExporter(config, out_dir="data_processed")

    all_scores = []
    drift_records = []
    total_windows = 0

    for ep_info in valid_episodes:
        ep_dir = Path(ep_info["episode_dir"])
        ep_id = ep_info["episode_id"]
        event_tag = ep_info.get("event_tag", "unknown")

        # Load raw data
        cam_ts_df = pd.read_csv(ep_dir / "camera_timestamps.csv")
        actions_df = pd.read_csv(ep_dir / "actions.csv")
        states_df = pd.read_csv(ep_dir / "states.csv")

        cam_timestamps = cam_ts_df["camera_timestamp_s"].values
        ctrl_timestamps = actions_df["timestamp_s"].values

        # Stage 3a: Drift correction
        corrected_cam_ts, calib = drift_corrector.correct(cam_timestamps, ctrl_timestamps)
        drift_records.append({
            "episode_id": ep_id,
            "raw_timestamps": cam_timestamps.copy(),
            "corrected_timestamps": corrected_cam_ts.copy(),
            "max_drift_ms": calib["max_drift_ms"]
        })

        # Stage 3b: Timestamp alignment
        aligned = aligner.align(corrected_cam_ts, actions_df, states_df)

        # Stage 4: Window sampling
        windows = sampler.sample_windows(aligned, ep_dir / "frames", ep_id)
        print(f"  {ep_id}: {len(windows)} windows sampled")

        # Stages 5-7: Flag + Score + Export each window
        for window in windows:
            bitmask, details = flagger.flag_window(window, corrected_cam_ts)
            score_result = scorer.score(bitmask, details)
            eqs = score_result["cis"]
            exporter.export_window(window, bitmask, details, score_result, event_tag)
            all_scores.append(eqs)
            total_windows += 1

    print(f"\n  Total windows processed: {total_windows}")

    # ============================================================
    # STAGE 8: EXPORT MANIFEST
    # ============================================================
    print("\n" + "="*60)
    print("STAGE 8: Exporting manifest...")
    print("="*60)
    manifest_path = exporter.finalize()

    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    _plot_drift(drift_records, plots_dir)
    _plot_quality_histogram(all_scores, config["quality"]["min_quality_score"], plots_dir)
    _plot_sample_windows(exporter.manifest, plots_dir)
    _plot_flag_cooccurrence(exporter.manifest, plots_dir)

    # ============================================================
    # QA SIMULATION (2% manual review)
    # ============================================================
    print("\n" + "="*60)
    print("QA SIMULATION: 2% manual review strategy")
    print("="*60)
    _simulate_qa(exporter.manifest, config)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    accepted_ct  = sum(1 for r in exporter.manifest if r["accepted"])
    rejected_ct  = sum(1 for r in exporter.manifest if not r["accepted"])
    uncertain_ct = sum(1 for r in exporter.manifest if r.get("uncertain", False))
    print(f"  Accepted examples:  {accepted_ct}")
    print(f"  Rejected examples:  {rejected_ct}")
    print(f"  Uncertain (review): {uncertain_ct}")
    print(f"  Mean CIS: {np.mean(all_scores):.3f}  Std: {np.std(all_scores):.3f}")
    print(f"  Shards: {manifest_path.replace('manifest.csv','shards/')}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Plots: {plots_dir}/  (drift_correction, quality_histogram, sample_windows, flag_cooccurrence)")
    print("="*60)


def _plot_drift(drift_records: list, plots_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Clock Drift: Before vs After Correction", fontsize=14, fontweight="bold")

    for rec in drift_records[:3]:
        raw = rec["raw_timestamps"]
        cor = rec["corrected_timestamps"]
        n = min(len(raw), len(cor))
        diff_before = raw[:n] - np.linspace(raw[0], raw[n-1], n)
        diff_after = cor[:n] - np.linspace(cor[0], cor[n-1], n)
        axes[0].plot(raw[:n], diff_before * 1000, alpha=0.7, label=rec["episode_id"])
        axes[1].plot(cor[:n], diff_after * 1000, alpha=0.7, label=rec["episode_id"])

    for ax, title in zip(axes, ["Before Correction (ms)", "After Correction (ms)"]):
        ax.set_xlabel("Episode Time (s)")
        ax.set_ylabel("Timestamp Deviation (ms)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "drift_correction.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plots_dir}/drift_correction.png")


def _plot_quality_histogram(scores: list, threshold: float, plots_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    scores_arr = np.array(scores)
    n_bins = 30
    ax.hist(scores_arr[scores_arr >= threshold], bins=n_bins, color="#2ecc71",
            alpha=0.8, label=f"Accepted (≥{threshold})")
    ax.hist(scores_arr[scores_arr < threshold], bins=n_bins, color="#e74c3c",
            alpha=0.8, label=f"Rejected (<{threshold})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold = {threshold}")
    # Uncertainty band
    unc_lo = max(0, threshold - 0.15)
    ax.axvspan(unc_lo, threshold, alpha=0.15, color="orange", label="Uncertainty band (priority review)")
    ax.set_xlabel("Causal Integrity Score (CIS)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Causal Integrity Scores\n(SYNC_ERR weighted 3×)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    stats_text = f"n={len(scores_arr)}\nμ={scores_arr.mean():.3f}\nσ={scores_arr.std():.3f}\nmin={scores_arr.min():.3f}"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(plots_dir / "quality_histogram.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plots_dir}/quality_histogram.png")


def _plot_sample_windows(manifest: list, plots_dir: Path):
    """Show context vs target frame grids for best and worst CIS examples."""
    if not manifest:
        return

    sorted_m = sorted(manifest, key=lambda r: r["example_quality_score"])
    worst = sorted_m[0]
    best  = sorted_m[-1]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Sample Training Windows: Context vs Target Frames\n(left = highest CIS, right = lowest CIS)",
                 fontsize=13, fontweight="bold")

    for col_idx, (label, row) in enumerate([("HIGH CIS", best), ("LOW CIS", worst)]):
        shard_name = row.get("shard", "")
        example_key = row.get("example_key", "")
        sub = "shards" if row.get("accepted", True) else "rejected"
        shard_path = Path("data_processed") / sub / shard_name
        if not shard_path.exists() or not example_key:
            continue

        # Read arrays back from tar shard
        try:
            import tarfile, io
            with tarfile.open(str(shard_path), "r") as tf:
                ctx = np.load(io.BytesIO(tf.extractfile(f"{example_key}/ctx_video.npy").read()))
                tgt = np.load(io.BytesIO(tf.extractfile(f"{example_key}/tgt_video.npy").read()))
        except Exception:
            continue

        n_show = 4
        for i in range(n_show):
            ax = fig.add_subplot(4, 8, col_idx * 4 + i + 1)
            frame_idx = int(i * len(ctx) / n_show)
            ax.imshow(ctx[frame_idx][:, :, ::-1])
            ax.set_title(f"ctx {i}", fontsize=7)
            ax.axis("off")
            if i == 0:
                flags_short = row.get("quality_flags","")[:25]
                ax.set_ylabel(f"{label}\nCIS={row['example_quality_score']:.2f}\n{flags_short}",
                              fontsize=7, rotation=0, ha='right')

        for i in range(n_show):
            ax = fig.add_subplot(4, 8, col_idx * 4 + i + 1 + 16)
            frame_idx = int(i * len(tgt) / n_show)
            ax.imshow(tgt[frame_idx][:, :, ::-1])
            ax.set_title(f"tgt {i}", fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "sample_windows.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plots_dir}/sample_windows.png")


def _plot_flag_cooccurrence(manifest: list, plots_dir: Path):
    """
    Flag co-occurrence heatmap.
    Cell [i,j] = fraction of examples where both flag_i and flag_j are set.
    Diagonal = individual flag prevalence.
    Useful for dashboard monitoring: correlated flags suggest a shared root cause
    (e.g., STALL + ACT_SAT always co-occurring → robot hitting limit, not just slow).
    """
    if not manifest:
        return

    from collections import Counter
    # Collect all flag names present in the dataset
    all_flag_sets = []
    for row in manifest:
        flags_str = row.get("quality_flags", "")
        flags = set(f for f in flags_str.split("|") if f) if flags_str else set()
        all_flag_sets.append(flags)

    all_flags_seen = sorted({f for fs in all_flag_sets for f in fs})
    if len(all_flags_seen) < 2:
        return

    n = len(all_flags_seen)
    matrix = np.zeros((n, n))
    total = len(all_flag_sets)

    for i, fi in enumerate(all_flags_seen):
        for j, fj in enumerate(all_flags_seen):
            count = sum(1 for fs in all_flag_sets if fi in fs and fj in fs)
            matrix[i, j] = count / max(total, 1)

    fig, ax = plt.subplots(figsize=(max(7, n), max(6, n - 1)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_flags_seen, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(all_flags_seen, fontsize=9)
    ax.set_title("Flag Co-occurrence Matrix\n(fraction of examples where both flags are set)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Co-occurrence fraction")

    # Annotate cells with values
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(plots_dir / "flag_cooccurrence.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plots_dir}/flag_cooccurrence.png")


def _simulate_qa(manifest: list, config: dict):
    """Simulate the 2% manual review strategy with uncertainty-prioritised sampling."""
    df = pd.DataFrame(manifest)
    n_total = len(df)
    review_frac = config["qa"]["manual_review_fraction"]
    n_review = max(1, int(n_total * review_frac))
    outlier_thr = config["qa"]["outlier_low_eqs_threshold"]

    n_random  = n_review // 3
    n_outlier = n_review // 3
    n_uncertain = n_review - n_random - n_outlier

    random_sample   = df.sample(n=min(n_random, len(df)), random_state=42)
    outlier_pool    = df[df["example_quality_score"] < outlier_thr]
    outlier_sample  = outlier_pool.sample(n=min(n_outlier, len(outlier_pool)), random_state=42)
    uncertain_pool  = df[df.get("uncertain", pd.Series([False]*len(df))).astype(bool)] if "uncertain" in df.columns else pd.DataFrame()
    uncertain_sample = uncertain_pool.sample(n=min(n_uncertain, len(uncertain_pool)), random_state=42) if len(uncertain_pool) > 0 else pd.DataFrame()

    combined = pd.concat([random_sample, outlier_sample, uncertain_sample]).drop_duplicates()

    print(f"\n  QA REVIEW PLAN (2% = {n_review} examples from {n_total} total):")
    print(f"    Random sample:    {len(random_sample)} examples (baseline coverage)")
    print(f"    Outlier sample:   {len(outlier_sample)} examples (CIS < {outlier_thr})")
    print(f"    Uncertain sample: {len(uncertain_sample)} examples (near acceptance boundary)")
    print(f"    Combined unique:  {len(combined)} examples")
    print(f"\n  WHAT HUMANS CHECK:")
    print(f"    - SYNC_ERR: does the robot actually move as commanded? (most critical)")
    print(f"    - DUP_FRAME: are consecutive frames suspiciously identical?")
    print(f"    - ACT_SAT: is the robot hitting joint limits and ignoring commands?")
    print(f"    - Visual blur/exposure artifacts not caught by Laplacian threshold")
    print(f"    - Label utility: does this window teach useful causal dynamics?")
    print(f"\n  AUTOMATED DASHBOARD METRICS:")
    print(f"    - CIS distribution per episode and overall (with hard/soft breakdown)")
    print(f"    - SYNC_ERR residual histogram (most critical — 3× weight in CIS)")
    print(f"    - Flag co-occurrence heatmap (STALL + ACT_SAT often co-occur)")
    print(f"    - Per-shard CIS variance (high variance = inhomogeneous episode quality)")
    print(f"    - DUP_FRAME rate per camera (sudden spike = encoder stall)")

    # Print flag prevalence
    if len(df) > 0 and "quality_flags" in df.columns:
        from collections import Counter
        all_flags = []
        for flags_str in df["quality_flags"].dropna():
            all_flags.extend([f for f in flags_str.split("|") if f])
        flag_counts = Counter(all_flags)
        print(f"\n  FLAG PREVALENCE:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"    {flag:<14} {count:4d}  ({100*count/max(n_total,1):.1f}%)")


if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)