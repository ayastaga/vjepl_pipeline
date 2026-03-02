import numpy as np
import pandas as pd
from pathlib import Path


class DatasetExporter:
    def __init__(self, config: dict, out_dir: str = "data_processed"):
        self.out_dir = Path(out_dir).resolve()
        self.min_eqs = config["quality"]["min_quality_score"]
        (self.out_dir / "examples").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "rejected").mkdir(parents=True, exist_ok=True)
        self.manifest = []

    def export_window(self, window: dict, bitmask: int, details: dict, eqs: float, event_tag: str = "unknown") -> str:
        """Serialize one training example. Returns output path."""
        ep_id = window["episode_id"]
        anchor_t = window["anchor_time"]
        safe_anchor = f"{anchor_t:.4f}".replace(".", "_")
        fname = f"{ep_id}__t{safe_anchor}.npz"

        folder = self.out_dir / ("examples" if eqs >= self.min_eqs else "rejected")
        out_path = folder / fname

        # np.savez_compressed appends .npz automatically so write directly
        np.savez_compressed(
            str(out_path.with_suffix("")),  # strip .npz, savez will add it
            ctx_video=window["ctx_video"],
            tgt_video=window["tgt_video"],
            ctx_actions=window["ctx_actions"],
            ctx_states=window["ctx_states"],
        )

        # Build manifest row
        row = {
            "episode_id": ep_id,
            "anchor_time": anchor_t,
            "example_quality_score": eqs,
            "quality_bitmask": bitmask,
            "quality_flags": "|".join(details.get("flag_names", [])),
            "action_alignment_method": window.get("action_alignment_method", ""),
            "event_tag": event_tag,
            "npz_path": str(out_path.relative_to(self.out_dir)),
            "accepted": eqs >= self.min_eqs,
            "ctx_frames": window["ctx_video"].shape[0],
            "tgt_frames": window["tgt_video"].shape[0],
            "mean_laplacian_var": details.get("mean_laplacian_var", 0),
            "max_exposure_shift": details.get("max_exposure_shift", 0),
            "sync_residual": details.get("sync_residual", 0),
        }
        self.manifest.append(row)
        return str(out_path)

    def finalize(self) -> str:
        """Write Parquet manifest and return path."""
        df = pd.DataFrame(self.manifest)
        # Write CSV (parquet requires pyarrow; CSV is universally readable)
        csv_path = self.out_dir / "manifest.csv"
        df.to_csv(str(csv_path), index=False)

        # Also write JSON for easy inspection
        json_path = self.out_dir / "manifest.json"
        df.to_json(str(json_path), orient="records", indent=2)

        accepted = df["accepted"].sum()
        total = len(df)
        print(f"\n  [Exporter] Manifest written: {total} examples total, {accepted} accepted ({100*accepted/max(total,1):.1f}%)")
        print(f"  [Exporter] CSV manifest: {csv_path}")
        print(f"  [Exporter] JSON manifest: {json_path}")
        print(f"  [Exporter] NOTE: In production, swap to df.to_parquet() with pyarrow installed.")
        return str(csv_path)