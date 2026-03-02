import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import linregress
from pathlib import Path


class DriftCorrector:
    """Estimates and removes linear clock drift from camera timestamps."""

    def correct(self, cam_timestamps: np.ndarray, ctrl_timestamps: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Fit T_ctrl = alpha * T_cam + delta_t.
        Returns corrected camera timestamps and calibration params.
        """
        n_ref = min(len(cam_timestamps), len(ctrl_timestamps))
        cam_ref = cam_timestamps[:n_ref]
        ctrl_ref = ctrl_timestamps[:n_ref]

        slope, intercept, r_value, _, _ = linregress(cam_ref, ctrl_ref)
        corrected = (cam_timestamps - intercept) / slope

        # Enforce monotonicity
        for i in range(1, len(corrected)):
            if corrected[i] <= corrected[i - 1]:
                corrected[i] = corrected[i - 1] + 1e-4

        calib = {
            "alpha": float(slope),
            "delta_t": float(intercept),
            "r_squared": float(r_value ** 2),
            "max_drift_ms": float(np.abs(corrected - cam_timestamps).max() * 1000)
        }
        return corrected, calib


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
        ctrl_ts = actions_df["timestamp_s"].values
        state_ts = states_df["timestamp_s"].values

        # Fill NaN state rows before interpolation
        states_df = states_df.interpolate(limit=10, limit_direction="both")

        # Clamp target timestamps to valid range
        t_min = max(ctrl_ts[0], state_ts[0])
        t_max = min(ctrl_ts[-1], state_ts[-1])
        valid_mask = (cam_ts >= t_min) & (cam_ts <= t_max)
        target_ts = cam_ts[valid_mask]

        # Joint angles: CubicSpline
        joint_cols = [c for c in states_df.columns if c.startswith("joint_")]
        eef_xyz_cols = ["eef_0", "eef_1", "eef_2"]
        joint_data = states_df[joint_cols].values
        eef_xyz = states_df[eef_xyz_cols].values

        aligned_joints = self._cubic_interp(state_ts, joint_data, target_ts)
        aligned_eef_xyz = self._cubic_interp(state_ts, eef_xyz, target_ts)

        # EEF orientation: treat eef_3,4,5 as Euler angles -> convert to quat -> SLERP
        eef_euler_cols = ["eef_3", "eef_4", "eef_5"]
        eef_euler = states_df[eef_euler_cols].values
        aligned_eef_quat = self._slerp_interp(state_ts, eef_euler, target_ts)

        # Actions: nearest-neighbor
        action_cols = [c for c in actions_df.columns if c.startswith("action_")]
        action_data = actions_df[action_cols].values
        aligned_actions = self._nearest_interp(ctrl_ts, action_data, target_ts)

        return {
            "timestamps": target_ts,
            "valid_mask": valid_mask,
            "joints": aligned_joints,          # (T, 7)
            "eef_xyz": aligned_eef_xyz,        # (T, 3)
            "eef_quat": aligned_eef_quat,      # (T, 4)
            "actions": aligned_actions,        # (T, 6)
            "action_alignment_method": "nearest_neighbor_joints_cubicspline_actions_nn_quat_slerp"
        }

    def _cubic_interp(self, src_ts, data, tgt_ts):
        out = np.zeros((len(tgt_ts), data.shape[1]))
        for j in range(data.shape[1]):
            valid = np.isfinite(data[:, j])
            if valid.sum() < 4:
                out[:, j] = 0.0
                continue
            cs = CubicSpline(src_ts[valid], data[valid, j])
            out[:, j] = cs(tgt_ts)
        return out

    def _slerp_interp(self, src_ts, euler_data, tgt_ts):
        # Convert Euler to quaternion
        quats = Rotation.from_euler("xyz", euler_data).as_quat()  # (N, 4)
        # SLERP needs sorted, unique times
        _, unique_idx = np.unique(src_ts, return_index=True)
        times = src_ts[unique_idx]
        qs = quats[unique_idx]
        if len(times) < 2:
            return np.tile([0, 0, 0, 1], (len(tgt_ts), 1))
        slerp = Slerp(times, Rotation.from_quat(qs))
        tgt_clipped = np.clip(tgt_ts, times[0], times[-1])
        return slerp(tgt_clipped).as_quat()

    def _nearest_interp(self, src_ts, data, tgt_ts):
        indices = np.searchsorted(src_ts, tgt_ts, side="left")
        indices = np.clip(indices, 0, len(src_ts) - 1)
        return data[indices]