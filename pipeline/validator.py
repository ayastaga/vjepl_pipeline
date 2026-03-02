import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path


class EpisodeValidator:
    REQUIRED_FILES = ["camera_timestamps.csv", "actions.csv", "states.csv", "meta.json"]

    def __init__(self, config: dict):
        self.cfg = config
        ctx = config["pipeline"]["context_seconds"]
        tgt = config["pipeline"]["target_seconds"]
        self.min_duration = ctx + tgt
        self.expected_action_hz  = config["pipeline"]["action_hz"]
        self.expected_video_fps  = config["pipeline"]["video_fps"]
        # Tolerance: actual rate must be within ±20% of nominal
        self.rate_tolerance = 0.20

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
        result["event_tag"]  = meta.get("event_tag", "unknown")
        result["duration_s"] = meta.get("duration_s", 0)

        # 3. Minimum duration
        if result["duration_s"] < self.min_duration:
            return self._fail(result, f"Duration {result['duration_s']:.1f}s < minimum {self.min_duration}s")

        # 4. CSV parseable with minimum rows + deep timestamp checks
        for fname, expected_hz in [
            ("camera_timestamps.csv", self.expected_video_fps),
            ("actions.csv",           self.expected_action_hz),
            ("states.csv",            self.expected_action_hz),
        ]:
            try:
                df = pd.read_csv(ep_dir / fname)
            except Exception as e:
                return self._fail(result, f"{fname} parse error: {e}")

            if len(df) < 10:
                return self._fail(result, f"{fname} has only {len(df)} rows")

            # Identify timestamp column (first column named *timestamp* or first column)
            ts_col = next((c for c in df.columns if "timestamp" in c.lower()), df.columns[0])
            ts = df[ts_col].dropna().values.astype(float)

            if len(ts) < 2:
                return self._fail(result, f"{fname} insufficient timestamp rows")

            # 4a. Monotonicity: no backwards jumps
            diffs = np.diff(ts)
            n_violations = int((diffs <= 0).sum())
            if n_violations > 0:
                return self._fail(result, f"{fname} has {n_violations} non-monotonic timestamp(s)")

            # 4b. Sampling rate aliasing detection
            #   Compute median inter-sample interval and compare to nominal.
            #   Also check for bimodal intervals (hallmark of duplicated samples):
            #   if >40% of intervals are within 1% of the minimum interval,
            #   and the median interval is ~2× the minimum, it's a duplicated stream.
            median_interval_s = float(np.median(diffs))
            if median_interval_s <= 0:
                return self._fail(result, f"{fname} zero median interval (constant timestamps)")

            actual_hz = 1.0 / median_interval_s
            lo = expected_hz * (1 - self.rate_tolerance)
            hi = expected_hz * (1 + self.rate_tolerance)
            if not (lo <= actual_hz <= hi):
                return self._fail(
                    result,
                    f"{fname} apparent rate {actual_hz:.1f} Hz outside expected "
                    f"{expected_hz} Hz ±{int(self.rate_tolerance*100)}% window. "
                    f"Possible aliased/duplicated stream."
                )

            # 4c. Duplication check: min interval should be ~median interval.
            min_interval_s = float(np.min(diffs[diffs > 1e-6]))
            if min_interval_s > 0 and (median_interval_s / min_interval_s) > 1.8:
                frac_at_min = float((diffs < min_interval_s * 1.01).mean())
                if frac_at_min > 0.35:
                    return self._fail(
                        result,
                        f"{fname} bimodal intervals suggest duplicated/aliased stream "
                        f"(min={min_interval_s*1000:.1f}ms, median={median_interval_s*1000:.1f}ms)"
                    )

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