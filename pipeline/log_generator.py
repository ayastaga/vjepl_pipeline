import os
import json
import numpy as np
import pandas as pd
import cv2
import yaml
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
        # Cumulative drift: drift rate * time
        drift = np.arange(n_frames) / fps * self.noise["clock_drift_rate"]
        return ideal + jitter + drift

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
        n = len(ctrl_timestamps)
        # 6-DOF delta end-effector commands
        t = ctrl_timestamps
        actions = np.zeros((n, 6))
        actions[:, 0] = 0.01 * np.sin(2 * np.pi * t / duration)
        actions[:, 1] = 0.01 * np.cos(2 * np.pi * t / duration * 1.3)
        actions[:, 2] = 0.005 * np.sin(4 * np.pi * t / duration)
        noise = self.rng.normal(0, 0.002, (n, 6))
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
        n = len(ctrl_timestamps)
        t = ctrl_timestamps
        # 7 joint angles + 6 end-effector pose
        joints = np.zeros((n, 7))
        for j in range(7):
            joints[:, j] = (j + 1) * 0.1 * np.sin(2 * np.pi * t / duration * (j * 0.3 + 0.5))
        eef = np.zeros((n, 6))
        eef[:, 0] = 0.3 + 0.05 * np.sin(2 * np.pi * t / duration)
        eef[:, 1] = 0.0 + 0.05 * np.cos(2 * np.pi * t / duration)
        eef[:, 2] = 0.5

        # Introduce stall region
        if self.rng.random() < self.noise["stall_prob"]:
            stall_t = self.rng.uniform(0.3, 0.7) * duration
            stall_mask = (t > stall_t) & (t < stall_t + 1.5)
            joints[stall_mask] = joints[np.where(stall_mask)[0][0]] if stall_mask.any() else joints

        state_data = np.hstack([joints, eef])
        cols = [f"joint_{i}" for i in range(7)] + [f"eef_{i}" for i in range(6)]
        df = pd.DataFrame(state_data, columns=cols)
        df.insert(0, "timestamp_s", ctrl_timestamps)
        return df

    def _simulate_packet_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        mask = self.rng.random(n) < self.noise["packet_loss_prob"]
        df.loc[mask, df.columns[1:]] = np.nan
        return df