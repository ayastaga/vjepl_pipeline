import numpy as np
from pathlib import Path
import cv2

# Extra seconds to ingest beyond the exported window (Runge boundary buffer)
INTERP_PAD_S = 0.1


class WindowSampler:
    def __init__(self, config: dict):
        p = config["pipeline"]
        self.ctx_s   = p["context_seconds"]
        self.tgt_s   = p["target_seconds"]
        self.fps     = p["video_fps"]
        self.stride_s = p["stride_seconds"]
        self.blind_gap_frames = 2  # action leakage prevention

        self.ctx_frames = int(self.ctx_s  * self.fps)
        self.tgt_frames = int(self.tgt_s  * self.fps)
        # Padding in frames: discard this many at each boundary after interpolation
        self.pad_frames = max(1, int(INTERP_PAD_S * self.fps))

    def sample_windows(self, aligned: dict, frames_dir: Path, episode_id: str) -> list[dict]:
        """
        Slide a window over aligned episode data and return list of window dicts.
        Boundary frames (pad_frames on each side) are never used as anchor points
        to ensure all spline evaluations come from the stable interior.
        """
        ts = aligned["timestamps"]
        n  = len(ts)
        stride_frames = max(1, int(self.stride_s * self.fps))
        windows = []

        all_frames = self._load_frames(frames_dir, n)
        if all_frames is None:
            return []

        ctx_f = self.ctx_frames
        tgt_f = self.tgt_frames
        gap   = self.blind_gap_frames
        pad   = self.pad_frames   # Runge boundary buffer — skip first/last pad_frames as anchors

        # Valid anchor range: must have ctx_f context frames, gap, tgt_f target frames,
        # and stay pad_frames away from episode boundaries on both sides.
        anchor_start = ctx_f + pad
        anchor_end   = n - tgt_f - gap - pad

        for anchor_idx in range(anchor_start, anchor_end, stride_frames):
            ctx_start = anchor_idx - ctx_f
            ctx_end   = anchor_idx           # exclusive
            tgt_start = anchor_idx + gap
            tgt_end   = tgt_start + tgt_f

            if tgt_end > n:
                break

            ctx_video = all_frames[ctx_start:ctx_end]
            tgt_video = all_frames[tgt_start:tgt_end]

            state_data  = np.hstack([aligned["joints"], aligned["eef_xyz"], aligned["eef_quat"]])
            ctx_states  = state_data[ctx_start:ctx_end]

            action_end  = min(tgt_end, len(aligned["actions"]))
            ctx_actions = aligned["actions"][ctx_start:action_end]

            window = {
                "episode_id":  episode_id,
                "anchor_time": float(ts[anchor_idx]),
                "anchor_idx":  int(anchor_idx),
                "ctx_video":   ctx_video,
                "tgt_video":   tgt_video,
                "ctx_actions": ctx_actions,
                "ctx_states":  ctx_states,
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
                img = np.zeros((64, 64, 3), dtype=np.uint8)
            frames.append(img)
        while len(frames) < n_aligned:
            frames.append(np.zeros_like(frames[-1]))
        return np.array(frames[:n_aligned], dtype=np.uint8)