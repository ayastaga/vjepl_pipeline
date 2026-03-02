import numpy as np
from pathlib import Path
import cv2


class WindowSampler:
    def __init__(self, config: dict):
        p = config["pipeline"]
        self.ctx_s = p["context_seconds"]
        self.tgt_s = p["target_seconds"]
        self.fps = p["video_fps"]
        self.action_hz = p["action_hz"]
        self.stride_s = p["stride_seconds"]
        self.blind_gap_frames = 2  # action leakage prevention

        self.ctx_frames = int(self.ctx_s * self.fps)
        self.tgt_frames = int(self.tgt_s * self.fps)
        # Action window: t-ctx_s to t+tgt_s at action_hz
        # We'll index into aligned arrays which are at video fps (resampled)

    def sample_windows(self, aligned: dict, frames_dir: Path, episode_id: str) -> list[dict]:
        """
        Slide a window over aligned episode data and return list of window dicts.

        Each window dict contains:
          - ctx_frames: np.ndarray (ctx_frames, H, W, 3) uint8
          - tgt_frames: np.ndarray (tgt_frames, H, W, 3) uint8
          - ctx_actions: np.ndarray (ctx_action_steps + tgt_action_steps, 6)
          - ctx_states: np.ndarray (ctx_frames, state_dim)
          - anchor_time: float
          - episode_id: str
        """
        ts = aligned["timestamps"]
        n = len(ts)
        total_frames_needed = self.ctx_frames + self.tgt_frames + self.blind_gap_frames
        windows = []

        # stride in frames
        stride_frames = max(1, int(self.stride_s * self.fps))

        # Load all frames
        all_frames = self._load_frames(frames_dir, n)
        if all_frames is None:
            return []

        # Stride through valid anchor indices
        ctx_f = self.ctx_frames
        tgt_f = self.tgt_frames
        gap = self.blind_gap_frames

        for anchor_idx in range(ctx_f, n - tgt_f - gap, stride_frames):
            ctx_start = anchor_idx - ctx_f
            ctx_end = anchor_idx           # exclusive
            tgt_start = anchor_idx + gap
            tgt_end = tgt_start + tgt_f

            if tgt_end > n:
                break

            ctx_video = all_frames[ctx_start:ctx_end]   # (ctx_f, H, W, 3)
            tgt_video = all_frames[tgt_start:tgt_end]   # (tgt_f, H, W, 3)

            # States (at video fps)
            state_data = np.hstack([aligned["joints"], aligned["eef_xyz"], aligned["eef_quat"]])
            ctx_states = state_data[ctx_start:ctx_end]

            # Actions: full window t-ctx to t+tgt (include future for conditioning)
            # Map back to aligned array indices
            action_end = min(tgt_end, len(aligned["actions"]))
            ctx_actions = aligned["actions"][ctx_start:action_end]

            window = {
                "episode_id": episode_id,
                "anchor_time": float(ts[anchor_idx]),
                "anchor_idx": int(anchor_idx),
                "ctx_video": ctx_video,
                "tgt_video": tgt_video,
                "ctx_actions": ctx_actions,
                "ctx_states": ctx_states,
                "action_alignment_method": aligned.get("action_alignment_method", "nn"),
            }
            windows.append(window)

        return windows

    def _load_frames(self, frames_dir: Path, n_aligned: int) -> np.ndarray | None:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            return None
        frames = []
        for fp in frame_files[:n_aligned]:
            img = cv2.imread(str(fp))
            if img is None:
                # Use black frame as placeholder for dropped frame
                img = np.zeros((64, 64, 3), dtype=np.uint8)
            frames.append(img)
        # Pad if needed
        while len(frames) < n_aligned:
            frames.append(np.zeros_like(frames[-1]))
        return np.array(frames[:n_aligned], dtype=np.uint8)