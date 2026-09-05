"""Tests for AMP motion-dataset sampling."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import torch

from rsl_rl.utils.motion_loader import AMPLoader


def test_combined_npz_clip_weights_follow_frame_counts(tmp_path, monkeypatch):
    math_utils = ModuleType("isaaclab.utils.math")
    math_utils.convert_quat = lambda quat, to: quat
    math_utils.subtract_frame_transforms = (
        lambda anchor_pos, anchor_quat, target_pos, target_quat: (
            target_pos - anchor_pos,
            target_quat,
        )
    )
    math_utils.matrix_from_quat = lambda quat: torch.eye(3).expand(*quat.shape[:-1], 3, 3)
    math_utils.quat_apply_inverse = lambda quat, vector: vector
    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.math = math_utils
    isaaclab = ModuleType("isaaclab")
    isaaclab.utils = isaaclab_utils
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", isaaclab_utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_utils)

    num_frames = 8
    num_bodies = 2
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[..., 3] = 1.0
    body_velocity_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    motion_path = tmp_path / "combined.npz"
    np.savez(
        motion_path,
        fps=np.asarray([50.0]),
        clip_names=np.asarray(["short", "long"]),
        clip_lengths=np.asarray([2, 6]),
        clip_fps=np.asarray([50.0, 50.0]),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_velocity_w,
        body_ang_vel_w=body_velocity_w,
    )

    loader = AMPLoader(
        device=torch.device("cpu"),
        time_between_frames=0.02,
        motion_files=str(motion_path),
        loader_type="body_kinematics_npz",
        body_names=("foot",),
        anchor_name="base",
        motion_body_names=("base", "foot"),
        all_body_names=("base", "foot"),
        preload_transitions=False,
    )

    np.testing.assert_allclose(loader.trajectory_weights, np.asarray([0.25, 0.75]))
