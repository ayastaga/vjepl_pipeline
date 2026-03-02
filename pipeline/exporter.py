import io
import json
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path


class DatasetExporter:
    def __init__(self, config: dict, out_dir: str = "data_processed"):
        self.out_dir          = Path(out_dir).resolve()
        self.min_cis          = config["quality"]["min_quality_score"]
        self.examples_per_shard = config.get("export", {}).get("examples_per_shard", 50)

        (self.out_dir / "shards"   ).mkdir(parents=True, exist_ok=True)
        (self.out_dir / "rejected" ).mkdir(parents=True, exist_ok=True)

        self.manifest      = []
        self._shard_idx    = 0       # accepted shard counter
        self._rej_idx      = 0       # rejected shard counter
        self._shard_count  = 0       # examples in current accepted shard
        self._rej_count    = 0       # examples in current rejected shard
        self._shard_tf     = None    # open TarFile for accepted
        self._rej_tf       = None    # open TarFile for rejected
        self._open_shard(accepted=True)
        self._open_shard(accepted=False)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def export_window(self, window: dict, bitmask: int, details: dict,
                      score_result: dict, event_tag: str = "unknown") -> str:
        """
        Write one example into the appropriate shard.
        score_result is the dict returned by ExampleScorer.score().
        Returns the shard path the example was written to.
        """
        cis      = score_result["cis"]
        accepted = score_result["accepted"]
        uncertain = score_result.get("uncertain", False)

        ep_id    = window["episode_id"]
        anchor_t = window["anchor_time"]
        example_key = f"{ep_id}__t{anchor_t:.4f}".replace(".", "_")

        tf = self._shard_tf if accepted else self._rej_tf
        shard_path = self._current_shard_path(accepted)

        # Write arrays as .npy bytes inside the tar
        self._add_array(tf, example_key, "ctx_video",   window["ctx_video"])
        self._add_array(tf, example_key, "tgt_video",   window["tgt_video"])
        self._add_array(tf, example_key, "ctx_actions", window["ctx_actions"])
        self._add_array(tf, example_key, "ctx_states",  window["ctx_states"])

        # Write metadata as JSON
        meta = {
            "episode_id":             ep_id,
            "anchor_time":            float(anchor_t),
            "cis":                    float(cis),
            "uncertain":              bool(uncertain),
            "quality_bitmask":        int(bitmask),
            "quality_flags":          details.get("flag_names", []),
            "action_alignment_method": window.get("action_alignment_method", ""),
            "event_tag":              event_tag,
            "ctx_shape":              list(window["ctx_video"].shape),
            "tgt_shape":              list(window["tgt_video"].shape),
        }
        self._add_json(tf, example_key, meta)

        # Rotate shard if full
        if accepted:
            self._shard_count += 1
            if self._shard_count >= self.examples_per_shard:
                self._close_shard(accepted=True)
                self._shard_idx += 1
                self._shard_count = 0
                self._open_shard(accepted=True)
        else:
            self._rej_count += 1
            if self._rej_count >= self.examples_per_shard:
                self._close_shard(accepted=False)
                self._rej_idx += 1
                self._rej_count = 0
                self._open_shard(accepted=False)

        # Manifest row
        self.manifest.append({
            "episode_id":             ep_id,
            "anchor_time":            float(anchor_t),
            "example_quality_score":  float(cis),
            "uncertain":              bool(uncertain),
            "quality_bitmask":        int(bitmask),
            "quality_flags":          "|".join(details.get("flag_names", [])),
            "action_alignment_method": window.get("action_alignment_method", ""),
            "event_tag":              event_tag,
            "accepted":               bool(accepted),
            "shard":                  str(shard_path.name),
            "example_key":            example_key,
            "mean_laplacian_var":     details.get("mean_laplacian_var", 0),
            "max_exposure_shift":     details.get("max_exposure_shift", 0),
            "sync_residual":          details.get("sync_residual", 0),
            "actuator_sat_frac":      details.get("actuator_saturation_frac", 0),
            "max_ssim_consecutive":   details.get("max_ssim_consecutive", 0),
        })
        return str(shard_path)

    def finalize(self) -> str:
        """Close open shards and write manifest. Returns manifest CSV path."""
        self._close_shard(accepted=True)
        self._close_shard(accepted=False)

        df       = pd.DataFrame(self.manifest)
        csv_path = self.out_dir / "manifest.csv"
        df.to_csv(str(csv_path), index=False)

        json_path = self.out_dir / "manifest.json"
        df.to_json(str(json_path), orient="records", indent=2)

        accepted = int(df["accepted"].sum())
        total    = len(df)
        uncertain_count = int(df["uncertain"].sum()) if "uncertain" in df.columns else 0
        n_shards = self._shard_idx + (1 if self._shard_count > 0 else 0)
        print(f"\n  [Exporter] {total} examples: {accepted} accepted in {n_shards} shard(s), "
              f"{total-accepted} rejected, {uncertain_count} uncertain (priority review)")
        print(f"  [Exporter] Shards:   {self.out_dir / 'shards'}/")
        print(f"  [Exporter] Manifest: {csv_path}")
        print(f"  [Exporter] Format: WebDataset tar shards ({self.examples_per_shard} examples/shard)")
        return str(csv_path)

    # ------------------------------------------------------------------ #
    #  Shard management                                                    #
    # ------------------------------------------------------------------ #

    def _shard_filename(self, accepted: bool, idx: int) -> str:
        prefix = "shard" if accepted else "rejected"
        return f"{prefix}_{idx:06d}.tar"

    def _current_shard_path(self, accepted: bool) -> Path:
        idx = self._shard_idx if accepted else self._rej_idx
        sub = "shards" if accepted else "rejected"
        return self.out_dir / sub / self._shard_filename(accepted, idx)

    def _open_shard(self, accepted: bool):
        path = self._current_shard_path(accepted)
        tf   = tarfile.open(str(path), "w")
        if accepted:
            self._shard_tf = tf
        else:
            self._rej_tf = tf

    def _close_shard(self, accepted: bool):
        tf = self._shard_tf if accepted else self._rej_tf
        if tf is not None:
            tf.close()
        if accepted:
            self._shard_tf = None
        else:
            self._rej_tf = None

    # ------------------------------------------------------------------ #
    #  Tar write helpers                                                   #
    # ------------------------------------------------------------------ #

    def _add_array(self, tf: tarfile.TarFile, key: str, field: str, arr: np.ndarray):
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        data = buf.read()
        info = tarfile.TarInfo(name=f"{key}/{field}.npy")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    def _add_json(self, tf: tarfile.TarFile, key: str, obj: dict):
        data = json.dumps(obj, indent=2).encode()
        info = tarfile.TarInfo(name=f"{key}/meta.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))