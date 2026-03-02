import numpy as np
import cv2


# Bitmask definitions
FLAG_BITS = {
    "MISSING_V":  1 << 0,   # Missing or black video frames
    "JITTER_A":   1 << 1,   # High action timestamp jitter
    "BLUR":       1 << 2,   # Excessive motion blur
    "STALL":      1 << 3,   # Robot not moving
    "EXPO_S":     1 << 4,   # Sudden exposure shift
    "SYNC_ERR":   1 << 5,   # Action/state physics mismatch
    "DROP_P":     1 << 6,   # Packet loss in state stream
    "COMP_A":     1 << 7,   # Compression artifacts (DCT blocking)
    "JITTER_V":   1 << 8,   # Video timestamp jitter (bonus flag)
}

FLAG_NAMES = {v: k for k, v in FLAG_BITS.items()}


class QualityFlagger:
    """
    Computes quality flags for a single training window.
    Returns (bitmask: int, flag_details: dict).
    """

    def __init__(self, config: dict):
        q = config["quality"]
        self.missing_frame_gap_ms = q["missing_frame_gap_ms"]
        self.max_consecutive_drops = q["missing_frame_max_consecutive"]
        self.jitter_var_thr = q["jitter_var_threshold_ms2"]
        self.blur_thr = q["blur_laplacian_threshold"]
        self.stall_thr = q["stall_joint_delta_threshold"]
        self.expo_thr = q["exposure_shift_threshold"]
        self.sync_thr = q["sync_error_threshold"]
        self.packet_loss_thr = q["packet_loss_max_fraction"]
        self.dct_thr = q["compression_dct_threshold"]

    def flag_window(self, window: dict, cam_timestamps: np.ndarray | None = None) -> tuple[int, dict]:
        """
        Evaluate all quality detectors for a window.
        Returns (bitmask, details_dict).
        """
        ctx_video = window["ctx_video"]   # (T, H, W, 3)
        tgt_video = window["tgt_video"]
        all_video = np.concatenate([ctx_video, tgt_video], axis=0)
        actions = window["ctx_actions"]   # (T, 6)
        states = window["ctx_states"]     # (T, state_dim)

        bitmask = 0
        details = {}
        hard_flags = []

        # ---- 1. MISSING_V: black / missing frames ----
        black_mask = np.all(all_video < 5, axis=(1, 2, 3))
        n_black = int(black_mask.sum())
        consecutive = self._max_consecutive(black_mask)
        details["missing_frames"] = int(n_black)
        details["max_consecutive_missing"] = int(consecutive)
        if consecutive >= self.max_consecutive_drops or n_black > 0.15 * len(all_video):
            bitmask |= FLAG_BITS["MISSING_V"]
            hard_flags.append("MISSING_V")

        # ---- 2. JITTER_A: action timestamp jitter ----
        # Proxy: variance of action magnitudes as jitter indicator
        action_norms = np.linalg.norm(actions, axis=1)
        jitter_var = float(np.var(np.diff(action_norms)) * 1000)  # scale to ms^2 proxy
        details["action_jitter_var"] = jitter_var
        if jitter_var > self.jitter_var_thr:
            bitmask |= FLAG_BITS["JITTER_A"]

        # ---- 3. BLUR: Laplacian variance on video frames ----
        lap_vars = []
        for frame in all_video:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_vars.append(float(np.var(lap)))
        mean_lap = float(np.mean(lap_vars))
        details["mean_laplacian_var"] = mean_lap
        if mean_lap < self.blur_thr:
            bitmask |= FLAG_BITS["BLUR"]

        # ---- 4. STALL: robot not moving ----
        if states.shape[0] > 1:
            joint_deltas = np.abs(np.diff(states[:, :7], axis=0)).sum()
            details["joint_cumulative_delta"] = float(joint_deltas)
            if joint_deltas < self.stall_thr * states.shape[0]:
                bitmask |= FLAG_BITS["STALL"]
        else:
            details["joint_cumulative_delta"] = 0.0

        # ---- 5. EXPO_S: sudden exposure shift ----
        frame_means = np.array([frame.mean() / 255.0 for frame in all_video])
        if len(frame_means) > 1:
            mean_diff = float(np.abs(np.diff(frame_means)).max())
        else:
            mean_diff = 0.0
        details["max_exposure_shift"] = mean_diff
        if mean_diff > self.expo_thr:
            bitmask |= FLAG_BITS["EXPO_S"]

        # ---- 6. SYNC_ERR: action/state physics mismatch ----
        # Heuristic: if robot states barely change despite non-zero actions
        if states.shape[0] > 1 and actions.shape[0] > 1:
            action_energy = float(np.linalg.norm(actions) / (len(actions) + 1e-8))
            state_change = float(np.linalg.norm(np.diff(states[:, :7], axis=0)) / (len(states) + 1e-8))
            sync_residual = action_energy / (state_change + 1e-6)
            details["sync_residual"] = sync_residual
            # Very high ratio = actions commanded but robot didn't move
            if sync_residual > (1.0 / self.sync_thr):
                bitmask |= FLAG_BITS["SYNC_ERR"]
        else:
            details["sync_residual"] = 0.0

        # ---- 7. DROP_P: NaN fraction in states (packet loss) ----
        if hasattr(states, 'dtype') and states.dtype == float:
            nan_frac = float(np.isnan(states).mean())
        else:
            nan_frac = 0.0
        details["nan_fraction"] = nan_frac
        if nan_frac > self.packet_loss_thr:
            bitmask |= FLAG_BITS["DROP_P"]
            hard_flags.append("DROP_P")

        # ---- 8. COMP_A: DCT compression artifacts ----
        dct_score = self._dct_artifact_score(all_video)
        details["dct_artifact_score"] = dct_score
        if dct_score > self.dct_thr:
            bitmask |= FLAG_BITS["COMP_A"]

        # ---- 9. JITTER_V: video frame interval variance (bonus) ----
        if cam_timestamps is not None and len(cam_timestamps) > 2:
            intervals_ms = np.diff(cam_timestamps) * 1000
            jitter_v = float(np.var(intervals_ms))
            details["video_jitter_var_ms2"] = jitter_v
            if jitter_v > self.jitter_var_thr:
                bitmask |= FLAG_BITS["JITTER_V"]

        details["hard_flags"] = hard_flags
        details["flag_names"] = self._decode_flags(bitmask)
        return bitmask, details

    def _max_consecutive(self, mask: np.ndarray) -> int:
        max_run, run = 0, 0
        for v in mask:
            run = run + 1 if v else 0
            max_run = max(max_run, run)
        return max_run

    def _dct_artifact_score(self, frames: np.ndarray) -> float:
        """Measure 8x8 block discontinuity as proxy for compression artifacts."""
        scores = []
        for frame in frames[::max(1, len(frames) // 5)]:  # sample every Nth frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            if h < 8 or w < 8:
                continue
            # Compute horizontal and vertical 8-block boundary discontinuities
            h_rows = gray[7::8, :]
            h_next = gray[8::8, :]
            min_h = min(len(h_rows), len(h_next))
            h_bound = np.abs(h_rows[:min_h] - h_next[:min_h]).mean() if min_h > 0 and h > 8 else 0
            v_cols = gray[:, 7::8]
            v_next = gray[:, 8::8]
            min_v = min(v_cols.shape[1], v_next.shape[1])
            v_bound = np.abs(v_cols[:, :min_v] - v_next[:, :min_v]).mean() if min_v > 0 and w > 8 else 0
            interior_var = np.var(gray)
            score = (h_bound + v_bound) / (interior_var + 1e-6)
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0

    def _decode_flags(self, bitmask: int) -> list[str]:
        return [name for name, bit in FLAG_BITS.items() if bitmask & bit]