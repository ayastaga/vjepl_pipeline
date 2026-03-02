import numpy as np
import cv2


# ------------------------------------------------------------------ #
#  Bitmask definitions                                                #
# ------------------------------------------------------------------ #
FLAG_BITS = {
    "MISSING_V":  1 << 0,   # Missing / black video frames        [HARD]
    "JITTER_A":   1 << 1,   # High action timestamp jitter        [soft]
    "BLUR":       1 << 2,   # Excessive motion blur               [soft]
    "STALL":      1 << 3,   # Robot not moving                    [soft]
    "EXPO_S":     1 << 4,   # Sudden exposure shift               [soft]
    "SYNC_ERR":   1 << 5,   # Physics residual mismatch           [soft, high-weight]
    "DROP_P":     1 << 6,   # Packet loss in state stream         [HARD]
    "COMP_A":     1 << 7,   # Compression / DCT artifacts         [soft]
    "JITTER_V":   1 << 8,   # Video timestamp jitter              [soft]
    "DUP_FRAME":  1 << 9,   # Duplicate consecutive frames (SSIM) [soft]
    "ACT_SAT":    1 << 10,  # Actuator saturation at joint limits [soft]
}

FLAG_NAMES = {v: k for k, v in FLAG_BITS.items()}

# Joint limits (radians) — representative 7-DOF arm; configurable
DEFAULT_JOINT_LIMITS = np.array([
    [-2.897, 2.897],  # joint 0
    [-1.763, 1.763],  # joint 1
    [-2.897, 2.897],  # joint 2
    [-3.072, 0.017],  # joint 3
    [-2.897, 2.897],  # joint 4
    [-0.018, 3.752],  # joint 5
    [-2.897, 2.897],  # joint 6
])


class QualityFlagger:
    """
    Computes quality flags for a single training window.
    Returns (bitmask: int, flag_details: dict).
    """

    def __init__(self, config: dict):
        q = config["quality"]
        self.missing_frame_gap_ms    = q["missing_frame_gap_ms"]
        self.max_consecutive_drops   = q["missing_frame_max_consecutive"]
        self.jitter_var_thr          = q["jitter_var_threshold_ms2"]
        self.blur_thr                = q["blur_laplacian_threshold"]
        self.stall_thr               = q["stall_joint_delta_threshold"]
        self.expo_thr                = q["exposure_shift_threshold"]
        self.sync_thr                = q["sync_error_threshold"]        # physics residual θ
        self.packet_loss_thr         = q["packet_loss_max_fraction"]
        self.dct_thr                 = q["compression_dct_threshold"]
        self.ssim_dup_thr            = q.get("ssim_duplicate_threshold", 0.98)
        self.sat_margin              = q.get("actuator_saturation_margin", 0.05)  # fraction of range
        self.joint_limits            = np.array(q.get("joint_limits", DEFAULT_JOINT_LIMITS.tolist()))

    def flag_window(self, window: dict, cam_timestamps: np.ndarray | None = None) -> tuple[int, dict]:
        """
        Evaluate all quality detectors for one window.
        Returns (bitmask, details_dict).
        """
        ctx_video = window["ctx_video"]
        tgt_video = window["tgt_video"]
        all_video = np.concatenate([ctx_video, tgt_video], axis=0)
        actions   = window["ctx_actions"]  # (T, 6)
        states    = window["ctx_states"]   # (T, state_dim) — joint_0..6, eef_0..9

        bitmask    = 0
        details    = {}
        hard_flags = []

        # ---- 1. MISSING_V: black / missing frames ----
        black_mask   = np.all(all_video < 5, axis=(1, 2, 3))
        n_black      = int(black_mask.sum())
        consecutive  = self._max_consecutive(black_mask)
        details["missing_frames"]            = n_black
        details["max_consecutive_missing"]   = consecutive
        if consecutive >= self.max_consecutive_drops or n_black > 0.15 * len(all_video):
            bitmask |= FLAG_BITS["MISSING_V"]
            hard_flags.append("MISSING_V")

        # ---- 2. JITTER_A: action command jitter ----
        action_norms = np.linalg.norm(actions, axis=1)
        jitter_var   = float(np.var(np.diff(action_norms)) * 1000)
        details["action_jitter_var"] = jitter_var
        if jitter_var > self.jitter_var_thr:
            bitmask |= FLAG_BITS["JITTER_A"]

        # ---- 3. BLUR: Laplacian variance ----
        lap_vars = []
        for frame in all_video:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap  = cv2.Laplacian(gray, cv2.CV_64F)
            lap_vars.append(float(np.var(lap)))
        mean_lap = float(np.mean(lap_vars))
        details["mean_laplacian_var"] = mean_lap
        if mean_lap < self.blur_thr:
            bitmask |= FLAG_BITS["BLUR"]

        # ---- 4. STALL: robot not moving ----
        if states.shape[0] > 1:
            joint_deltas = float(np.abs(np.diff(states[:, :7], axis=0)).sum())
            details["joint_cumulative_delta"] = joint_deltas
            if joint_deltas < self.stall_thr * states.shape[0]:
                bitmask |= FLAG_BITS["STALL"]
        else:
            details["joint_cumulative_delta"] = 0.0

        # ---- 5. EXPO_S: exposure shift ----
        frame_means = np.array([f.mean() / 255.0 for f in all_video])
        mean_diff   = float(np.abs(np.diff(frame_means)).max()) if len(frame_means) > 1 else 0.0
        details["max_exposure_shift"] = mean_diff
        if mean_diff > self.expo_thr:
            bitmask |= FLAG_BITS["EXPO_S"]

        # ---- 6. SYNC_ERR: physics residual ||s_{t+1} - s_t - a_t|| > θ ----
        #   The previous heuristic (action_energy / state_change ratio) was wrong:
        #   it could not distinguish a stall from a small deliberate motion.
        #   The correct check computes the per-step prediction error of a naive
        #   first-order integrator and flags when the mean exceeds threshold θ.
        #   a_t must be in the same coordinate space as s_t (both are joint-space
        #   deltas here). action dims 0-5, state joints 0-6; we use the overlap (6).
        sync_residual = 0.0
        n_joints_overlap = min(actions.shape[1], states.shape[1], 6)
        if states.shape[0] > 1 and actions.shape[0] > 1 and n_joints_overlap > 0:
            s_curr = states[:-1, :n_joints_overlap]         # s_t
            s_next = states[1:,  :n_joints_overlap]         # s_{t+1}
            # Actions aligned to ctx window (may be longer due to tgt future);
            # take the matching prefix
            a_t = actions[:len(s_curr), :n_joints_overlap]  # a_t
            min_len = min(len(s_curr), len(s_next), len(a_t))
            if min_len > 0:
                residuals    = np.linalg.norm(s_next[:min_len] - s_curr[:min_len] - a_t[:min_len], axis=1)
                sync_residual = float(np.mean(residuals))
        details["sync_residual"] = sync_residual
        if sync_residual > self.sync_thr:
            bitmask |= FLAG_BITS["SYNC_ERR"]

        # ---- 7. DROP_P: NaN fraction in states (packet / bursty USB loss) ----
        nan_frac = float(np.isnan(states).mean()) if np.issubdtype(states.dtype, np.floating) else 0.0
        details["nan_fraction"] = nan_frac
        if nan_frac > self.packet_loss_thr:
            bitmask |= FLAG_BITS["DROP_P"]
            hard_flags.append("DROP_P")

        # ---- 8. COMP_A: DCT block artifacts ----
        dct_score = self._dct_artifact_score(all_video)
        details["dct_artifact_score"] = dct_score
        if dct_score > self.dct_thr:
            bitmask |= FLAG_BITS["COMP_A"]

        # ---- 9. JITTER_V: video timestamp variance ----
        if cam_timestamps is not None and len(cam_timestamps) > 2:
            intervals_ms = np.diff(cam_timestamps) * 1000
            jitter_v     = float(np.var(intervals_ms))
            details["video_jitter_var_ms2"] = jitter_v
            if jitter_v > self.jitter_var_thr:
                bitmask |= FLAG_BITS["JITTER_V"]

        # ---- 10. DUP_FRAME: SSIM-based duplicate / encoder stall detection ----
        dup_score = self._duplicate_frame_score(all_video)
        details["max_ssim_consecutive"] = dup_score
        if dup_score > self.ssim_dup_thr:
            bitmask |= FLAG_BITS["DUP_FRAME"]

        # ---- 11. ACT_SAT: actuator saturation at joint limits ----
        sat_frac = self._actuator_saturation_fraction(states)
        details["actuator_saturation_frac"] = sat_frac
        if sat_frac > 0.0:   # any saturation event is flagged
            bitmask |= FLAG_BITS["ACT_SAT"]

        details["hard_flags"] = hard_flags
        details["flag_names"] = self._decode_flags(bitmask)
        return bitmask, details

    # ------------------------------------------------------------------ #
    #  Helper methods                                                      #
    # ------------------------------------------------------------------ #

    def _max_consecutive(self, mask: np.ndarray) -> int:
        max_run, run = 0, 0
        for v in mask:
            run = run + 1 if v else 0
            max_run = max(max_run, run)
        return max_run

    def _dct_artifact_score(self, frames: np.ndarray) -> float:
        """Measure 8×8 block boundary discontinuity as proxy for DCT artifacts."""
        scores = []
        for frame in frames[::max(1, len(frames) // 5)]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            if h < 16 or w < 16:
                continue
            h_rows = gray[7::8, :]
            h_next = gray[8::8, :]
            min_h  = min(len(h_rows), len(h_next))
            h_bound = np.abs(h_rows[:min_h] - h_next[:min_h]).mean() if min_h > 0 else 0
            v_cols = gray[:, 7::8]
            v_next = gray[:, 8::8]
            min_v  = min(v_cols.shape[1], v_next.shape[1])
            v_bound = np.abs(v_cols[:, :min_v] - v_next[:, :min_v]).mean() if min_v > 0 else 0
            interior_var = np.var(gray)
            scores.append((h_bound + v_bound) / (interior_var + 1e-6))
        return float(np.mean(scores)) if scores else 0.0

    def _duplicate_frame_score(self, frames: np.ndarray) -> float:
        """
        Compute the maximum SSIM between consecutive frames.
        SSIM ≈ 1.0 means frames are nearly identical (encoder stall).

        Uses a lightweight approximation: normalised mean absolute difference
        inverted to [0,1], avoiding the full SSIM kernel for speed.
        Perfect duplicate → score 1.0; completely different → score ~0.
        """
        if len(frames) < 2:
            return 0.0
        max_sim = 0.0
        step = max(1, len(frames) // 20)   # sample at most ~20 pairs
        for i in range(0, len(frames) - 1, step):
            f1 = frames[i].astype(np.float32)
            f2 = frames[i + 1].astype(np.float32)
            mad  = np.mean(np.abs(f1 - f2))
            sim  = 1.0 - min(mad / 255.0, 1.0)
            max_sim = max(max_sim, float(sim))
        return max_sim

    def _actuator_saturation_fraction(self, states: np.ndarray) -> float:
        """
        Fraction of timesteps where any joint is within sat_margin of its limit.
        states[:, :7] are joint angles in radians.
        """
        if states.shape[0] == 0 or states.shape[1] < 7:
            return 0.0
        joints = states[:, :7]
        n_joints = min(joints.shape[1], len(self.joint_limits))
        sat_events = 0
        for j in range(n_joints):
            lo, hi  = self.joint_limits[j]
            margin  = (hi - lo) * self.sat_margin
            at_lo   = joints[:, j] < (lo + margin)
            at_hi   = joints[:, j] > (hi - margin)
            sat_events += int((at_lo | at_hi).sum())
        return float(sat_events) / max(states.shape[0] * n_joints, 1)

    def _decode_flags(self, bitmask: int) -> list[str]:
        return [name for name, bit in FLAG_BITS.items() if bitmask & bit]