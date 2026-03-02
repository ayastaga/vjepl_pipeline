import sys
import yaml
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
            eqs = scorer.score(bitmask, details)
            exporter.export_window(window, bitmask, details, eqs, event_tag)
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

    # ============================================================
    # QA SIMULATION (2% manual review)
    # ============================================================
    print("\n" + "="*60)
    print("QA SIMULATION: 2% manual review strategy")
    print("="*60)
    _simulate_qa(exporter.manifest, config)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"  Accepted examples: {sum(1 for r in exporter.manifest if r['accepted'])}")
    print(f"  Rejected examples: {sum(1 for r in exporter.manifest if not r['accepted'])}")
    print(f"  Mean EQS: {np.mean(all_scores):.3f}  Std: {np.std(all_scores):.3f}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Plots: {plots_dir}/")
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
    ax.set_xlabel("Example Quality Score (EQS)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Example Quality Scores", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Stats box
    stats_text = f"n={len(scores_arr)}\nμ={scores_arr.mean():.3f}\nσ={scores_arr.std():.3f}\nmin={scores_arr.min():.3f}"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(plots_dir / "quality_histogram.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plots_dir}/quality_histogram.png")


def _plot_sample_windows(manifest: list, plots_dir: Path):
    """Show context vs target frame grids for best and worst EQS examples."""
    if not manifest:
        return

    sorted_m = sorted(manifest, key=lambda r: r["example_quality_score"])
    worst = sorted_m[0]
    best = sorted_m[-1]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Sample Training Windows: Context vs Target Frames", fontsize=14, fontweight="bold")

    for col_idx, (label, row) in enumerate([("HIGH EQS", best), ("LOW EQS", worst)]):
        npz_path = Path("data_processed") / row["npz_path"]
        if not npz_path.exists():
            continue
        data = np.load(str(npz_path))
        ctx = data["ctx_video"]
        tgt = data["tgt_video"]

        n_show_ctx = min(4, len(ctx))
        n_show_tgt = min(4, len(tgt))
        n_show = n_show_ctx + n_show_tgt

        for i in range(n_show_ctx):
            ax = fig.add_subplot(4, 8, col_idx * 4 + i + 1)
            ax.imshow(ctx[int(i * len(ctx) / n_show_ctx)][:, :, ::-1])  # BGR->RGB
            ax.set_title(f"ctx {i}", fontsize=7)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(f"{label}\nEQS={row['example_quality_score']:.2f}\n{row['quality_flags'][:20]}",
                              fontsize=7, rotation=0, ha='right')

        for i in range(n_show_tgt):
            ax = fig.add_subplot(4, 8, col_idx * 4 + i + 1 + 16)
            ax.imshow(tgt[int(i * len(tgt) / n_show_tgt)][:, :, ::-1])
            ax.set_title(f"tgt {i}", fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "sample_windows.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plots_dir}/sample_windows.png")


def _simulate_qa(manifest: list, config: dict):
    """Simulate the 2% manual review strategy."""
    df = pd.DataFrame(manifest)
    n_total = len(df)
    review_frac = config["qa"]["manual_review_fraction"]
    n_review = max(1, int(n_total * review_frac))
    outlier_thr = config["qa"]["outlier_low_eqs_threshold"]

    n_random = n_review // 2
    n_outlier = n_review - n_random

    random_sample = df.sample(n=min(n_random, len(df)), random_state=42)
    outlier_pool = df[df["example_quality_score"] < outlier_thr]
    outlier_sample = outlier_pool.sample(n=min(n_outlier, len(outlier_pool)), random_state=42)

    combined = pd.concat([random_sample, outlier_sample]).drop_duplicates()

    print(f"\n  QA REVIEW PLAN (2% = {n_review} examples from {n_total} total):")
    print(f"    Random sample:  {len(random_sample)} examples")
    print(f"    Outlier sample: {len(outlier_sample)} examples (EQS < {outlier_thr})")
    print(f"    Combined unique: {len(combined)} examples")
    print(f"\n  WHAT HUMANS CHECK:")
    print(f"    - Visual coherence of ctx/tgt frame pairs")
    print(f"    - Action/state synchrony (do joints move when commanded?)")
    print(f"    - Exposure or blur artifacts not caught by automated flags")
    print(f"    - Label utility (does this window teach useful dynamics?)")
    print(f"\n  AUTOMATED DASHBOARD METRICS:")
    print(f"    - EQS distribution per episode and overall")
    print(f"    - Flag co-occurrence heatmap")
    print(f"    - Per-flag prevalence rates")
    print(f"    - Shard-level variance of quality scores")

    # Print flag prevalence
    if len(df) > 0 and "quality_flags" in df.columns:
        from collections import Counter
        all_flags = []
        for flags_str in df["quality_flags"].dropna():
            all_flags.extend([f for f in flags_str.split("|") if f])
        flag_counts = Counter(all_flags)
        print(f"\n  FLAG PREVALENCE:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"    {flag:<12} {count:4d}  ({100*count/max(n_total,1):.1f}%)")


if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)