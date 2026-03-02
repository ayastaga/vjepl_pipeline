import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import linregress
from pathlib import Path


# Episodes longer than this use piecewise-linear drift correction
PIECEWISE_THRESHOLD_S = 30.0
# Number of equal-length segments for piecewise correction
PIECEWISE_N_SEGMENTS  = 4


class DriftCorrector:
    """
    Estimates and removes clock drift from camera timestamps.

    Short episodes (<30s): single global linear regression.
    Long episodes (>=30s): piecewise-linear correction — fits independent
      linear models per segment, then stitches with continuity constraints.
      This handles non-linear thermal drift that a single regression misses.
    """

    def correct(self, cam_timestamps: np.ndarray, ctrl_timestamps: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Returns (corrected_cam_timestamps, calib_dict).
        """
        duration = float(cam_timestamps[-1] - cam_timestamps[0])

        if duration >= PIECEWISE_THRESHOLD_S:
            corrected, calib = self._piecewise_correct(cam_timestamps, ctrl_timestamps)
        else:
            corrected, calib = self._linear_correct(cam_timestamps, ctrl_timestamps)

        # Enforce strict monotonicity (clock correction can introduce tiny inversions)
        for i in range(1, len(corrected)):
            if corrected[i] <= corrected[i - 1]:
                corrected[i] = corrected[i - 1] + 1e-5

        calib["max_drift_ms"] = float(np.abs(corrected - cam_timestamps).max() * 1000)
        calib["method"] = "piecewise_linear" if duration >= PIECEWISE_THRESHOLD_S else "linear"
        return corrected, calib

    # ------------------------------------------------------------------ #

    def _linear_correct(self, cam_ts: np.ndarray, ctrl_ts: np.ndarray) -> tuple[np.ndarray, dict]:
        n = min(len(cam_ts), len(ctrl_ts))
        slope, intercept, r_value, _, _ = linregress(cam_ts[:n], ctrl_ts[:n])
        corrected = (cam_ts - intercept) / max(slope, 1e-9)
        return corrected, {"alpha": float(slope), "delta_t": float(intercept),
                           "r_squared": float(r_value ** 2)}

    def _piecewise_correct(self, cam_ts: np.ndarray, ctrl_ts: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Divide the episode into PIECEWISE_N_SEGMENTS equal-length segments.
        Fit independent linear models per segment, then apply continuously:
        each segment's correction is blended at the boundary to avoid jumps.
        """
        n = min(len(cam_ts), len(ctrl_ts))
        seg_size = n // PIECEWISE_N_SEGMENTS
        corrected = cam_ts.copy().astype(float)

        seg_params = []
        for s in range(PIECEWISE_N_SEGMENTS):
            i0 = s * seg_size
            i1 = (s + 1) * seg_size if s < PIECEWISE_N_SEGMENTS - 1 else n
            c = cam_ts[i0:i1]
            t = ctrl_ts[i0:i1]
            if len(c) < 4:
                seg_params.append((1.0, 0.0))
                continue
            slope, intercept, _, _, _ = linregress(c, t)
            slope = max(slope, 1e-9)
            seg_params.append((float(slope), float(intercept)))

        # Apply per-segment correction
        for s in range(PIECEWISE_N_SEGMENTS):
            i0 = s * seg_size
            i1 = (s + 1) * seg_size if s < PIECEWISE_N_SEGMENTS - 1 else len(cam_ts)
            slope, intercept = seg_params[s]
            corrected[i0:i1] = (cam_ts[i0:i1] - intercept) / slope

        # Enforce continuity: stitch segment boundaries with linear blend over 5 frames
        for s in range(1, PIECEWISE_N_SEGMENTS):
            boundary = s * seg_size
            blend_half = 5
            i0 = max(0, boundary - blend_half)
            i1 = min(len(corrected), boundary + blend_half)
            if i1 > i0:
                ideal = np.linspace(corrected[i0], corrected[i1 - 1], i1 - i0)
                corrected[i0:i1] = ideal

        return corrected, {"n_segments": PIECEWISE_N_SEGMENTS}


class TimestampAligner:
    """
    Aligns 50Hz controller signals to the 30Hz video grid.

    Strategy per signal type:
      - Joint angles / EEF xyz: CubicSpline (C2 continuity for smooth dynamics)
      - EEF orientation (quaternion): SLERP in SO(3)
      - Actions: nearest-neighbor (avoids synthesizing commands that were never issued)
    """

    def __init__(self, config: dict):
        self.cfg = config

    def align(self, cam_ts: np.ndarray, actions_df: pd.DataFrame, states_df: pd.DataFrame) -> dict:
        """
        Interpolate all controller signals onto cam_ts grid.
        Returns dict of aligned arrays.
        """
        ctrl_ts  = actions_df["timestamp_s"].values
        state_ts = states_df["timestamp_s"].values

        # Fill NaN state rows before interpolation, respecting burst-loss gaps.
        # Gaps longer than 10 steps are left as NaN so DROP_P can catch them.
        states_df = states_df.copy()
        states_df = states_df.interpolate(limit=10, limit_direction="both")

        # Clamp target timestamps to the valid overlap of all streams
        t_min = max(ctrl_ts[0], state_ts[0])
        t_max = min(ctrl_ts[-1], state_ts[-1])
        valid_mask = (cam_ts >= t_min) & (cam_ts <= t_max)
        target_ts  = cam_ts[valid_mask]

        # Joint angles + EEF xyz: CubicSpline (C2 continuity)
        joint_cols   = [c for c in states_df.columns if c.startswith("joint_")]
        eef_xyz_cols = ["eef_0", "eef_1", "eef_2"]
        joint_data   = states_df[joint_cols].values
        eef_xyz      = states_df[eef_xyz_cols].values

        aligned_joints  = self._cubic_interp(state_ts, joint_data, target_ts)
        aligned_eef_xyz = self._cubic_interp(state_ts, eef_xyz,    target_ts)

        # EEF orientation: Euler -> quaternion -> SLERP
        eef_euler_cols = ["eef_3", "eef_4", "eef_5"]
        eef_euler      = states_df[eef_euler_cols].values
        aligned_eef_quat = self._slerp_interp(state_ts, eef_euler, target_ts)

        # Actions: nearest-neighbor (only correct choice — see module docstring)
        action_cols  = [c for c in actions_df.columns if c.startswith("action_")]
        action_data  = actions_df[action_cols].values
        aligned_actions = self._nearest_interp(ctrl_ts, action_data, target_ts)

        return {
            "timestamps": target_ts,
            "valid_mask": valid_mask,
            "joints":     aligned_joints,     # (T, 7)
            "eef_xyz":    aligned_eef_xyz,    # (T, 3)
            "eef_quat":   aligned_eef_quat,   # (T, 4)
            "actions":    aligned_actions,    # (T, 6)
            "action_alignment_method": "cubicspline_joints_slerp_quat_nn_actions",
        }

    def _cubic_interp(self, src_ts: np.ndarray, data: np.ndarray, tgt_ts: np.ndarray) -> np.ndarray:
        out = np.zeros((len(tgt_ts), data.shape[1]))
        for j in range(data.shape[1]):
            valid = np.isfinite(data[:, j])
            if valid.sum() < 4:
                out[:, j] = 0.0
                continue
            cs = CubicSpline(src_ts[valid], data[valid, j])
            out[:, j] = cs(tgt_ts)
        return out

    def _slerp_interp(self, src_ts: np.ndarray, euler_data: np.ndarray, tgt_ts: np.ndarray) -> np.ndarray:
        quats = Rotation.from_euler("xyz", euler_data).as_quat()  # (N, 4)
        _, unique_idx = np.unique(src_ts, return_index=True)
        times = src_ts[unique_idx]
        qs    = quats[unique_idx]
        if len(times) < 2:
            return np.tile([0, 0, 0, 1], (len(tgt_ts), 1))
        slerp = Slerp(times, Rotation.from_quat(qs))
        tgt_clipped = np.clip(tgt_ts, times[0], times[-1])
        return slerp(tgt_clipped).as_quat()

    def _nearest_interp(self, src_ts: np.ndarray, data: np.ndarray, tgt_ts: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(src_ts, tgt_ts, side="left")
        indices = np.clip(indices, 0, len(src_ts) - 1)
        return data[indices]