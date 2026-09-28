"""Tests for AMP motion-dataset sampling."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
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


@pytest.mark.parametrize("convention", ("wxyz", "xyzw"))
def test_npz_rotations_match_wxyz_environment_features(tmp_path, monkeypatch, convention):
    """Check known 90/180-degree rotations, including positions and velocities."""
    math_utils = ModuleType("isaaclab.utils.math")

    def matrix_from_quat(quat):
        w, x, y, z = quat.unbind(-1)
        return torch.stack((
            1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
            2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
            2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
        ), dim=-1).reshape(*quat.shape[:-1], 3, 3)

    def apply_inverse(quat, vector):
        return (matrix_from_quat(quat).transpose(-1, -2) @ vector.unsqueeze(-1)).squeeze(-1)

    def subtract_frames(anchor_pos, anchor_quat, target_pos, target_quat):
        inverse = anchor_quat.clone()
        inverse[..., 1:] *= -1
        a, b = inverse[..., :1], target_quat[..., :1]
        av, bv = inverse[..., 1:], target_quat[..., 1:]
        relative_quat = torch.cat((
            a*b - (av*bv).sum(-1, keepdim=True),
            a*bv + b*av + torch.linalg.cross(av, bv, dim=-1),
        ), dim=-1)
        return apply_inverse(anchor_quat, target_pos - anchor_pos), relative_quat

    math_utils.convert_quat = lambda quat, to: torch.roll(quat, 1 if to == "wxyz" else -1, dims=-1)
    math_utils.matrix_from_quat = matrix_from_quat
    math_utils.quat_apply_inverse = apply_inverse
    math_utils.subtract_frame_transforms = subtract_frames
    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.math = math_utils
    isaaclab = ModuleType("isaaclab")
    isaaclab.utils = isaaclab_utils
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", isaaclab_utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_utils)

    quats = np.asarray([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)], [0, 0, 0, 1]], dtype=np.float32)
    if convention == "xyzw":
        quats = np.roll(quats, -1, axis=-1)
    motion_path = tmp_path / "rotated.npz"
    np.savez(
        motion_path,
        fps=np.asarray([50.0]),
        body_pos_w=np.tile([[3, 4, 5], [3, 5, 5]], (2, 1, 1)).astype(np.float32),
        body_quat_w=np.tile(quats, (2, 1, 1)),
        body_lin_vel_w=np.tile([[0, 0, 0], [1, 2, 3]], (2, 1, 1)).astype(np.float32),
        body_ang_vel_w=np.tile([[0, 0, 0], [4, 5, 6]], (2, 1, 1)).astype(np.float32),
    )
    loader = AMPLoader(
        device="cpu", time_between_frames=0.02, motion_files=str(motion_path),
        loader_type="body_kinematics_npz", body_names=("foot",), anchor_name="base",
        motion_body_names=("base", "foot"), motion_quat_convention=convention,
        preload_transitions=False,
    )
    expected = torch.tensor([1, 0, 0, 0, -1, 1, 0, 0, 0, -1, -2, 3, -4, -5, 6], dtype=torch.float32)
    torch.testing.assert_close(loader.trajectories[0], expected.expand(2, -1), atol=1e-6, rtol=1e-6)
