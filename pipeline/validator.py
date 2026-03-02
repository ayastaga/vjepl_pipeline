import json
import shutil
import pandas as pd
from pathlib import Path


class EpisodeValidator:
    REQUIRED_FILES = ["camera_timestamps.csv", "actions.csv", "states.csv", "meta.json"]

    def __init__(self, config: dict):
        self.cfg = config
        ctx = config["pipeline"]["context_seconds"]
        tgt = config["pipeline"]["target_seconds"]
        self.min_duration = ctx + tgt  # need at least one window

    def validate(self, episode_dir: str) -> dict:
        """
        Validate one episode directory.
        Returns a result dict with keys: episode_id, valid, reason.
        """
        ep_dir = Path(episode_dir)
        result = {"episode_dir": str(ep_dir), "valid": True, "reason": "ok"}

        # 1. Required files present
        for fname in self.REQUIRED_FILES:
            if not (ep_dir / fname).exists():
                return self._fail(result, f"Missing file: {fname}")

        # 2. Meta parseable
        try:
            with open(ep_dir / "meta.json") as f:
                meta = json.load(f)
        except Exception as e:
            return self._fail(result, f"meta.json parse error: {e}")

        result["episode_id"] = meta.get("episode_id", ep_dir.name)
        result["event_tag"] = meta.get("event_tag", "unknown")
        result["duration_s"] = meta.get("duration_s", 0)

        # 3. Minimum duration
        if result["duration_s"] < self.min_duration:
            return self._fail(result, f"Duration {result['duration_s']:.1f}s < minimum {self.min_duration}s")

        # 4. CSV parseable with minimum rows
        for fname in ["camera_timestamps.csv", "actions.csv", "states.csv"]:
            try:
                df = pd.read_csv(ep_dir / fname)
                if len(df) < 10:
                    return self._fail(result, f"{fname} has only {len(df)} rows")
            except Exception as e:
                return self._fail(result, f"{fname} parse error: {e}")

        # 5. Frames directory not empty
        frames_dir = ep_dir / "frames"
        if not frames_dir.exists() or len(list(frames_dir.glob("*.png"))) < 10:
            return self._fail(result, "Insufficient video frames")

        return result

    def validate_all(self, episode_dirs: list[str], quarantine_dir: str = "data_raw/quarantine") -> list[dict]:
        """Validate all episodes; quarantine failures."""
        q_dir = Path(quarantine_dir)
        valid_results = []
        for ep_dir in episode_dirs:
            res = self.validate(ep_dir)
            if not res["valid"]:
                q_dir.mkdir(parents=True, exist_ok=True)
                dest = q_dir / Path(ep_dir).name
                if not dest.exists():
                    shutil.copytree(ep_dir, dest)
                print(f"  [Validator] QUARANTINE {Path(ep_dir).name}: {res['reason']}")
            else:
                valid_results.append(res)
                print(f"  [Validator] OK {res['episode_id']} ({res['duration_s']:.1f}s)")
        return valid_results

    def _fail(self, result: dict, reason: str) -> dict:
        result["valid"] = False
        result["reason"] = reason
        return result