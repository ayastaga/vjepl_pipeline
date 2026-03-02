import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path


class LogGenerator:
    def __init__(self, config: dict, out_dir: str = "data_raw"):
        self.cfg = config
        self.gen = config["generation"]
        self.noise = config["noise"]
        self.out_dir = Path(out_dir)
        self.rng = np.random.default_rng(self.gen["seed"])

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_all(self) -> list[str]:
        """Generate all synthetic episodes. Returns list of episode dirs."""
        episode_dirs = []
        for ep_idx in range(self.gen["num_episodes"]):
            ep_id = f"ep_{ep_idx:04d}"
            ep_dir = self.out_dir / ep_id
            ep_dir.mkdir(parents=True, exist_ok=True)
            self._generate_episode(ep_id, ep_dir)
            episode_dirs.append(str(ep_dir))
            print(f"  [LogGenerator] Generated {ep_id}")
        return episode_dirs

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _generate_episode(self, ep_id: str, ep_dir: Path):
        duration = self.rng.uniform(self.gen["min_duration"], self.gen["max_duration"])
        fps = self.cfg["pipeline"]["video_fps"]
        action_hz = self.cfg["pipeline"]["action_hz"]

        # ---- camera timestamps (may have jitter + drift) ----
        n_frames_ideal = int(duration * fps)
        cam_timestamps = self._make_camera_timestamps(n_frames_ideal, fps)

        # ---- controller timestamps (separate clock domain) ----
        n_ctrl = int(duration * action_hz)
        ctrl_timestamps = self._make_controller_timestamps(n_ctrl, action_hz, duration)

        # ---- simulate frame drops ----
        cam_timestamps, drop_mask = self._simulate_frame_drops(cam_timestamps)

        # ---- generate video frames ----
        frames_dir = ep_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        self._generate_video_frames(cam_timestamps, drop_mask, frames_dir, duration)

        # ---- generate actions & states ----
        actions_df = self._generate_actions(ctrl_timestamps, duration)
        states_df = self._generate_states(ctrl_timestamps, duration, actions_df)

        # ---- simulate packet loss in states ----
        states_df = self._simulate_packet_loss(states_df)

        # ---- write to disk ----
        cam_ts_df = pd.DataFrame({
            "frame_index": np.arange(len(cam_timestamps)),
            "camera_timestamp_s": cam_timestamps,
        })
        cam_ts_df.to_csv(ep_dir / "camera_timestamps.csv", index=False)
        actions_df.to_csv(ep_dir / "actions.csv", index=False)
        states_df.to_csv(ep_dir / "states.csv", index=False)

        meta = {
            "episode_id": ep_id,
            "duration_s": float(duration),
            "n_frames": int(len(cam_timestamps)),
            "n_ctrl_steps": int(n_ctrl),
            "event_tag": self.rng.choice(["success", "fail", "unknown"],
                                          p=[0.6, 0.25, 0.15])
        }
        with open(ep_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _make_camera_timestamps(self, n_frames: int, fps: float) -> np.ndarray:
        ideal = np.arange(n_frames) / fps
        jitter = self.rng.normal(0, self.noise["jitter_std_ms"] / 1000.0, n_frames)

        # Linear base drift
        linear_drift = ideal * self.noise["clock_drift_rate"]

        # Non-linear thermal component: CPU throttling causes sinusoidal slow-down
        # ~0.5–2 Hz oscillation with amplitude proportional to drift rate.
        # This creates the non-linear residual that breaks pure linear regression
        # on long episodes and requires piecewise correction.
        thermal_freq = self.rng.uniform(0.3, 1.5)   # Hz
        thermal_amp  = self.noise["clock_drift_rate"] * self.noise.get("thermal_amplitude_scale", 0.4)
        thermal_drift = thermal_amp * np.sin(2 * np.pi * thermal_freq * ideal)

        return ideal + jitter + linear_drift + thermal_drift

    def _make_controller_timestamps(self, n_ctrl: int, hz: float, duration: float) -> np.ndarray:
        ideal = np.linspace(0, duration, n_ctrl)
        jitter = self.rng.normal(0, self.noise["jitter_std_ms"] / 1000.0 * 0.5, n_ctrl)
        return np.clip(ideal + jitter, 0, None)

    def _simulate_frame_drops(self, timestamps: np.ndarray):
        n = len(timestamps)
        drop_mask = self.rng.random(n) < self.noise["drop_frame_prob"]
        # Keep first and last frame
        drop_mask[0] = False
        drop_mask[-1] = False
        kept_timestamps = timestamps[~drop_mask]
        return kept_timestamps, drop_mask

    def _generate_video_frames(self, timestamps, drop_mask, frames_dir, duration):
        H = self.gen["video_height"]
        W = self.gen["video_width"]
        # Decide stall / blur / exposure windows
        stall_start = (self.rng.random() < self.noise["stall_prob"])
        blur_start = self.rng.random() < self.noise["blur_prob"]
        expo_shift = self.rng.random() < self.noise["exposure_shift_prob"]
        expo_shift_time = self.rng.uniform(0.3, 0.7) * duration

        for i, t in enumerate(timestamps):
            # Base: animated gradient
            x = np.linspace(0, 1, W)
            y = np.linspace(0, 1, H)
            xx, yy = np.meshgrid(x, y)
            phase = t / duration
            r = np.clip((xx + phase) % 1.0 * 255, 0, 255).astype(np.uint8)
            g = np.clip((yy + phase * 0.7) % 1.0 * 255, 0, 255).astype(np.uint8)
            b = np.clip((0.5 + 0.5 * np.sin(phase * 2 * np.pi)) * 200, 0, 255) * np.ones((H, W), np.uint8)
            frame = np.stack([b, g, r], axis=2)

            # Add noise
            noise_img = self.rng.integers(0, 30, (H, W, 3), dtype=np.uint8)
            frame = np.clip(frame.astype(int) + noise_img, 0, 255).astype(np.uint8)

            # Exposure shift
            if expo_shift and t > expo_shift_time:
                frame = np.clip(frame.astype(int) + 80, 0, 255).astype(np.uint8)

            # Blur
            if blur_start and (0.3 * duration < t < 0.5 * duration):
                frame = cv2.GaussianBlur(frame, (15, 15), 5)

            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), frame)

    def _generate_actions(self, ctrl_timestamps, duration) -> pd.DataFrame:
        """
        Generate joint-space delta commands (radians/step) at 50 Hz.

        Actions are the numerical derivative of the underlying joint trajectory,
        so the SYNC_ERR physics residual ||s_{t+1} - s_t - a_t|| is realistically
        small (~0.001–0.003 rad) for good data. The residual grows large only when:
          - stall events freeze the state while actions remain non-zero
          - bursty packet loss creates NaN gaps in the state stream
        Uses the same JOINT_MIDPOINTS and JOINT_AMPLITUDES as _generate_states
        so the coordinate spaces match exactly.
        """
        n = len(ctrl_timestamps)
        t = ctrl_timestamps

        JOINT_MIDPOINTS = np.array([0.0, 0.0, 0.0, -1.527, 0.0, 1.867, 0.0])
        JOINT_AMPLITUDES = np.array([0.3, 0.3, 0.3, 0.20, 0.3, 0.25, 0.3])

        # Reconstruct the same 6-joint (overlap with action dims) trajectory
        joints_underlying = np.zeros((n, 6))
        for j in range(6):
            phase_offset = j * 0.0  # must match _generate_states phase=0 (seed-independent)
            joints_underlying[:, j] = JOINT_MIDPOINTS[j] + JOINT_AMPLITUDES[j] * np.sin(
                2 * np.pi * t / duration * (j * 0.3 + 0.5)
            )

        # Numerical derivative: action_t ≈ joints[t+1] - joints[t]
        actions = np.zeros((n, 6))
        actions[:-1] = np.diff(joints_underlying, axis=0)
        actions[-1]  = actions[-2]

        # Small command noise (controller quantisation, ~1/10th of typical delta)
        noise = self.rng.normal(0, 0.0005, (n, 6))
        actions = actions + noise

        # Simulate action lag (shift actions forward in time)
        lag = self.noise["action_lag_frames"]
        actions = np.roll(actions, lag, axis=0)
        actions[:lag] = 0.0

        cols = [f"action_{i}" for i in range(6)]
        df = pd.DataFrame(actions, columns=cols)
        df.insert(0, "timestamp_s", ctrl_timestamps)
        return df

    def _generate_states(self, ctrl_timestamps, duration, actions_df) -> pd.DataFrame:
        """
        Generate 7-DOF joint angles + 6-DOF EEF pose.

        Joint angles are kept well within the ±2.897 rad limits (amplitude ≤ 0.8 rad,
        offset to mid-range) so ACT_SAT flags only fire on genuine limit events, not
        on all windows. This makes ACT_SAT a meaningful signal rather than constant noise.
        """
        n = len(ctrl_timestamps)
        t = ctrl_timestamps

        # Joint limits (Franka Panda-style 7-DOF) — pre-computed midpoints
        # so synthetic motion is centered in the valid range and never near limits.
        JOINT_MIDPOINTS = np.array([0.0, 0.0, 0.0, -1.527, 0.0, 1.867, 0.0])
        JOINT_AMPLITUDES = np.array([0.3, 0.3, 0.3, 0.20, 0.3, 0.25, 0.3])  # stay well inside limits

        # 7 joint angles oscillating around mid-range with episode-unique phases
        joints = np.zeros((n, 7))
        for j in range(7):
            phase_offset = self.rng.uniform(0, 2 * np.pi)
            joints[:, j] = JOINT_MIDPOINTS[j] + JOINT_AMPLITUDES[j] * np.sin(
                2 * np.pi * t / duration * (j * 0.3 + 0.5) + phase_offset
            )

        # EEF: xyz translation + euler orientation
        eef = np.zeros((n, 6))
        eef[:, 0] = 0.3 + 0.05 * np.sin(2 * np.pi * t / duration)
        eef[:, 1] = 0.0 + 0.05 * np.cos(2 * np.pi * t / duration)
        eef[:, 2] = 0.5

        # Stall region: freeze joints for ~1.5s
        if self.rng.random() < self.noise["stall_prob"]:
            stall_t = self.rng.uniform(0.3, 0.7) * duration
            stall_mask = (t > stall_t) & (t < stall_t + 1.5)
            if stall_mask.any():
                freeze_val = joints[np.where(stall_mask)[0][0]]
                joints[stall_mask] = freeze_val

        state_data = np.hstack([joints, eef])
        cols = [f"joint_{i}" for i in range(7)] + [f"eef_{i}" for i in range(6)]
        df = pd.DataFrame(state_data, columns=cols)
        df.insert(0, "timestamp_s", ctrl_timestamps)
        return df

    def _simulate_packet_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bursty packet loss model (USB bus saturation).

        Real USB bus saturation arrives in ~200ms bursts every few seconds,
        not as independent Bernoulli drops. We model this as a Poisson process
        of burst events, each lasting burst_duration_s, within which every
        packet is dropped. This creates the correlated-loss pattern that
        stresses interpolation robustness in aligner.py far more than i.i.d. loss.
        """
        n = len(df)
        ts = df["timestamp_s"].values
        duration = float(ts[-1] - ts[0]) if n > 1 else 1.0

        loss_mask = np.zeros(n, dtype=bool)

        # Uniform background loss (cable noise)
        bg_mask = self.rng.random(n) < (self.noise["packet_loss_prob"] * 0.3)
        loss_mask |= bg_mask

        # Bursty foreground loss (USB saturation events)
        burst_rate   = self.noise.get("burst_loss_rate_hz", 0.3)    # bursts per second
        burst_dur    = self.noise.get("burst_loss_duration_s", 0.15) # seconds per burst
        expected_bursts = max(1, int(duration * burst_rate))
        burst_starts = self.rng.uniform(ts[0], max(ts[0] + 0.1, ts[-1] - burst_dur), expected_bursts)

        for t_start in burst_starts:
            t_end = t_start + burst_dur
            burst_idx = np.where((ts >= t_start) & (ts <= t_end))[0]
            loss_mask[burst_idx] = True

        df = df.copy()
        df.loc[loss_mask, df.columns[1:]] = np.nan
        return df