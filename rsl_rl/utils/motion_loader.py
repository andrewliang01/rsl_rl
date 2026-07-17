# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion loader for Adversarial Motion Prior (AMP)."""

import json
import os
from collections.abc import Sequence

import numpy as np
import torch


class AMPLoader:
    """Loads and samples motion data for AMP training.

    This loader supports loading motion clips from JSON files and sampling
    transitions for discriminator training.

    Args:
        device: Device to load data on.
        time_between_frames: Time step between frames (should match env step_dt).
        motion_files: List of paths to motion JSON files.
        preload_transitions: Whether to preload transitions into memory.
        num_preload_transitions: Number of transitions to preload.
    """

    # Default observation dimensions for G1 robot
    JOINT_POS_SIZE = 29
    JOINT_VEL_SIZE = 29
    END_EFFECTOR_POS_SIZE = 12  # 4 end effectors * 3 coordinates

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    END_POS_START_IDX = JOINT_VEL_END_IDX
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE

    JOINT_EE_LOADER_TYPE = "joint_ee_json"
    BODY_KINEMATICS_LOADER_TYPE = "body_kinematics_npz"

    def __init__(
        self,
        device: torch.device,
        time_between_frames: float,
        motion_files: list[str] | str,
        loader_type: str = JOINT_EE_LOADER_TYPE,
        body_names: Sequence[str] = (),
        anchor_name: str = "",
        motion_body_names: Sequence[str] = (),
        all_body_names: Sequence[str] = (),
        preload_transitions: bool = True,
        num_preload_transitions: int = 100000,
        motion_quat_convention: str = "xyzw",
        expert_sampling_mode: str = "continuous",
        expert_trajectory_sampling_mode: str = "weighted_random",
    ) -> None:
        self.device = device
        self.time_between_frames = time_between_frames
        self.preload_transitions = preload_transitions
        self.num_preload_transitions = num_preload_transitions
        self.loader_type = self._normalize_loader_type(loader_type)
        if expert_sampling_mode not in ("continuous", "adjacent"):
            raise ValueError(
                "[AMPLoader] expert_sampling_mode must be 'continuous' or 'adjacent', "
                f"got {expert_sampling_mode!r}"
            )
        self.expert_sampling_mode = expert_sampling_mode
        if expert_trajectory_sampling_mode not in ("weighted_random", "round_robin"):
            raise ValueError(
                "[AMPLoader] expert_trajectory_sampling_mode must be 'weighted_random' or "
                f"'round_robin', got {expert_trajectory_sampling_mode!r}"
            )
        if expert_trajectory_sampling_mode == "round_robin" and preload_transitions:
            raise ValueError(
                "[AMPLoader] round_robin trajectory sampling requires preload_transitions=False "
                "so each minibatch can be sampled from one selected motion clip."
            )
        self.expert_trajectory_sampling_mode = expert_trajectory_sampling_mode
        self._amp_obs_dim = 0

        # Load trajectories from motion files
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Trajectory length in seconds
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        if self.loader_type == self.BODY_KINEMATICS_LOADER_TYPE:
            self._load_body_kinematics_npz(
                motion_files,
                body_names,
                anchor_name,
                motion_body_names,
                all_body_names,
                motion_quat_convention,
            )
        else:
            self._load_joint_ee_json(motion_files)

        if len(self.trajectories) == 0:
            raise ValueError(f"[AMPLoader] No motion files loaded for loader_type={self.loader_type}")

        # Normalize trajectory weights for sampling
        self.trajectory_weights = np.array(self.trajectory_weights)
        self.trajectory_weights = self.trajectory_weights / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions if requested
        if self.preload_transitions:
            print(f"[AMPLoader] Preloading {num_preload_transitions} transitions...")
            self._preload_transitions()
            print("[AMPLoader] Preloading complete")

    @staticmethod
    def _normalize_loader_type(loader_type: str) -> str:
        aliases = {
            "legacy": AMPLoader.JOINT_EE_LOADER_TYPE,
            "joint_ee": AMPLoader.JOINT_EE_LOADER_TYPE,
            "joint_ee_json": AMPLoader.JOINT_EE_LOADER_TYPE,
            "json": AMPLoader.JOINT_EE_LOADER_TYPE,
            "txt": AMPLoader.JOINT_EE_LOADER_TYPE,
            "mimic": AMPLoader.BODY_KINEMATICS_LOADER_TYPE,
            "body_npz": AMPLoader.BODY_KINEMATICS_LOADER_TYPE,
            "body_kinematics": AMPLoader.BODY_KINEMATICS_LOADER_TYPE,
            "body_kinematics_npz": AMPLoader.BODY_KINEMATICS_LOADER_TYPE,
            "npz": AMPLoader.BODY_KINEMATICS_LOADER_TYPE,
        }
        normalized = aliases.get(str(loader_type), str(loader_type))
        if normalized not in (AMPLoader.JOINT_EE_LOADER_TYPE, AMPLoader.BODY_KINEMATICS_LOADER_TYPE):
            raise ValueError(f"[AMPLoader] Unknown loader_type: {loader_type}")
        return normalized

    @staticmethod
    def _expand_motion_files(motion_files: list[str] | str, extensions: tuple[str, ...]) -> list[str]:
        raw_files = [motion_files] if isinstance(motion_files, str) else list(motion_files)
        expanded: list[str] = []
        for path in raw_files:
            if os.path.isdir(path):
                for root, _dirs, filenames in os.walk(path):
                    for filename in filenames:
                        if filename.endswith(extensions):
                            expanded.append(os.path.join(root, filename))
            elif path.endswith(extensions):
                expanded.append(path)
            else:
                expanded.append(path)
        return sorted(expanded)

    def _load_joint_ee_json(self, motion_files: list[str] | str) -> None:
        self._amp_obs_dim = AMPLoader.END_POS_END_IDX - AMPLoader.JOINT_POSE_START_IDX
        for i, motion_file in enumerate(self._expand_motion_files(motion_files, (".txt", ".json"))):
            self.trajectory_names.append(os.path.splitext(motion_file)[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"], dtype=np.float32)

            if motion_data.shape[1] < AMPLoader.END_POS_END_IDX:
                raise ValueError(
                    f"[AMPLoader] {motion_file} has {motion_data.shape[1]} dims, "
                    f"expected at least {AMPLoader.END_POS_END_IDX}"
                )

            trajectory = torch.tensor(
                motion_data[:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.END_POS_END_IDX],
                dtype=torch.float32,
                device=self.device,
            )
            self.trajectories.append(trajectory)
            self.trajectories_full.append(trajectory)
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(float(motion_json.get("MotionWeight", 1.0)))

            frame_duration = float(motion_json.get("FrameDuration", 0.02))
            self.trajectory_frame_durations.append(frame_duration)
            traj_len = max(0.0, (motion_data.shape[0] - 1) * frame_duration)
            self.trajectory_lens.append(traj_len)
            self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"[AMPLoader] Loaded {traj_len:.2f}s {self.loader_type} motion from {motion_file}")

    def _load_body_kinematics_npz(
        self,
        motion_files: list[str] | str,
        body_names: Sequence[str],
        anchor_name: str,
        motion_body_names: Sequence[str],
        all_body_names: Sequence[str],
        motion_quat_convention: str,
    ) -> None:
        if not body_names:
            raise ValueError("[AMPLoader] body_kinematics_npz requires non-empty body_names")
        if not anchor_name:
            raise ValueError("[AMPLoader] body_kinematics_npz requires anchor_name")

        import isaaclab.utils.math as math_utils

        requested_names = tuple(body_names) + (anchor_name,)
        if all_body_names:
            missing_robot_names = [name for name in requested_names if name not in all_body_names]
            if missing_robot_names:
                raise ValueError(f"[AMPLoader] body names not found in robot model: {missing_robot_names}")

        motion_body_names = list(motion_body_names or all_body_names)
        if not motion_body_names:
            raise ValueError(
                "[AMPLoader] body_kinematics_npz requires motion_body_names. "
                "This list must match the body axis order stored in the npz files."
            )
        missing_motion_names = [name for name in requested_names if name not in motion_body_names]
        if missing_motion_names:
            raise ValueError(f"[AMPLoader] body names not found in npz motion body list: {missing_motion_names}")

        body_indexes = [motion_body_names.index(name) for name in body_names]
        anchor_index = motion_body_names.index(anchor_name)
        num_bodies = len(body_indexes)
        self._amp_obs_dim = num_bodies * (3 + 6 + 3 + 3)

        for motion_file in self._expand_motion_files(motion_files, (".npz",)):
            data = np.load(motion_file)
            required_keys = (
                "fps",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
            missing_keys = [key for key in required_keys if key not in data.files]
            if missing_keys:
                raise ValueError(f"[AMPLoader] {motion_file} missing npz keys: {missing_keys}")

            body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=self.device)
            body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=self.device)
            if motion_quat_convention == "wxyz":
                body_quat_w = math_utils.convert_quat(body_quat_w, to="xyzw")
            elif motion_quat_convention != "xyzw":
                raise ValueError(
                    "[AMPLoader] motion_quat_convention must be 'xyzw' or 'wxyz', "
                    f"got {motion_quat_convention!r}"
                )
            body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=self.device)
            body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=self.device)

            required_body_count = max(body_indexes + [anchor_index]) + 1
            if body_pos_w.shape[1] < required_body_count:
                raise ValueError(
                    f"[AMPLoader] {motion_file} has {body_pos_w.shape[1]} bodies, "
                    f"but motion_body_names indexes require at least {required_body_count}"
                )

            target_pos_w = body_pos_w[:, body_indexes, :]
            target_quat_w = body_quat_w[:, body_indexes, :]
            target_lin_vel_w = body_lin_vel_w[:, body_indexes, :]
            target_ang_vel_w = body_ang_vel_w[:, body_indexes, :]
            anchor_pos_w = body_pos_w[:, anchor_index, None, :].expand(-1, num_bodies, -1)
            anchor_quat_w = body_quat_w[:, anchor_index, None, :].expand(-1, num_bodies, -1)

            body_pos_b, body_quat_b = math_utils.subtract_frame_transforms(
                anchor_pos_w,
                anchor_quat_w,
                target_pos_w,
                target_quat_w,
            )
            body_ori_b = math_utils.matrix_from_quat(body_quat_b)[..., :, :2].reshape(body_pos_w.shape[0], num_bodies, 6)
            body_lin_vel_b = math_utils.quat_apply_inverse(
                target_quat_w.reshape(-1, 4),
                target_lin_vel_w.reshape(-1, 3),
            ).reshape(body_pos_w.shape[0], num_bodies, 3)
            body_ang_vel_b = math_utils.quat_apply_inverse(
                target_quat_w.reshape(-1, 4),
                target_ang_vel_w.reshape(-1, 3),
            ).reshape(body_pos_w.shape[0], num_bodies, 3)

            trajectory = torch.cat(
                [
                    body_pos_b.reshape(body_pos_w.shape[0], -1),
                    body_ori_b.reshape(body_pos_w.shape[0], -1),
                    body_lin_vel_b.reshape(body_pos_w.shape[0], -1),
                    body_ang_vel_b.reshape(body_pos_w.shape[0], -1),
                ],
                dim=-1,
            )

            default_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if "clip_lengths" in data.files:
                clip_lengths = np.asarray(data["clip_lengths"], dtype=np.int64).reshape(-1)
                if clip_lengths.size == 0 or np.any(clip_lengths <= 0):
                    raise ValueError(f"[AMPLoader] {motion_file} has invalid clip_lengths: {clip_lengths}")
                if int(clip_lengths.sum()) != int(trajectory.shape[0]):
                    raise ValueError(
                        f"[AMPLoader] {motion_file} clip_lengths sum to {clip_lengths.sum()}, "
                        f"but the concatenated arrays contain {trajectory.shape[0]} frames"
                    )
                clip_fps = (
                    np.asarray(data["clip_fps"], dtype=np.float64).reshape(-1)
                    if "clip_fps" in data.files
                    else np.full(clip_lengths.shape, default_fps, dtype=np.float64)
                )
                clip_names = (
                    np.asarray(data["clip_names"]).astype(str).reshape(-1)
                    if "clip_names" in data.files
                    else np.asarray([f"clip_{clip_idx:04d}" for clip_idx in range(len(clip_lengths))])
                )
                if len(clip_fps) != len(clip_lengths) or len(clip_names) != len(clip_lengths):
                    raise ValueError(
                        f"[AMPLoader] {motion_file} clip metadata lengths do not match: "
                        f"lengths={len(clip_lengths)}, fps={len(clip_fps)}, names={len(clip_names)}"
                    )
            else:
                clip_lengths = np.asarray([trajectory.shape[0]], dtype=np.int64)
                clip_fps = np.asarray([default_fps], dtype=np.float64)
                clip_names = np.asarray([os.path.basename(os.path.splitext(motion_file)[0])])

            frame_start = 0
            for clip_name, clip_length, fps in zip(clip_names, clip_lengths, clip_fps):
                clip_name = str(clip_name)
                frame_end = frame_start + int(clip_length)
                clip_trajectory = trajectory[frame_start:frame_end]
                trajectory_name = os.path.splitext(motion_file)[0]
                if len(clip_lengths) > 1:
                    trajectory_name = f"{trajectory_name}::{clip_name}"

                self.trajectory_names.append(trajectory_name)
                self.trajectories.append(clip_trajectory)
                self.trajectories_full.append(clip_trajectory)
                self.trajectory_idxs.append(len(self.trajectory_idxs))
                self.trajectory_weights.append(1.0)

                frame_duration = 1.0 / float(fps)
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = max(0.0, (clip_trajectory.shape[0] - 1) * frame_duration)
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(clip_trajectory.shape[0]))

                print(
                    f"[AMPLoader] Loaded {traj_len:.2f}s {self.loader_type} motion "
                    f"{clip_name!r} from {motion_file} with amp_obs_dim={self._amp_obs_dim}"
                )
                frame_start = frame_end

    def _preload_transitions(self) -> None:
        """Preload transitions into memory for faster sampling."""
        traj_idxs = self.weighted_traj_idx_sample_batch(self.num_preload_transitions)
        if self.expert_sampling_mode == "adjacent":
            self.preloaded_s, self.preloaded_s_next = self.get_adjacent_frame_batch(traj_idxs)
        else:
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames
            )

    def get_adjacent_frame_batch(self, traj_idxs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample exact ``frame[i] -> frame[i + 1]`` transitions from each selected trajectory."""
        traj_idxs = np.asarray(traj_idxs, dtype=np.int64)
        frame_idxs = np.zeros(len(traj_idxs), dtype=np.int64)

        for traj_idx in np.unique(traj_idxs):
            traj_mask = traj_idxs == traj_idx
            num_frames = int(self.trajectory_num_frames[traj_idx])
            if num_frames > 1:
                # The exclusive upper bound keeps frame + 1 inside the trajectory
                # and avoids introducing a duplicated final-frame transition.
                frame_idxs[traj_mask] = np.random.randint(0, num_frames - 1, size=int(traj_mask.sum()))

        states = torch.zeros(len(traj_idxs), self.amp_obs_dim, device=self.device)
        next_states = torch.zeros_like(states)
        for traj_idx in np.unique(traj_idxs):
            traj_mask = traj_idxs == traj_idx
            trajectory = self.trajectories_full[int(traj_idx)]
            current_idxs = frame_idxs[traj_mask]
            next_idxs = np.minimum(current_idxs + 1, trajectory.shape[0] - 1)
            states[traj_mask] = trajectory[current_idxs]
            next_states[traj_mask] = trajectory[next_idxs]

        return states, next_states

    def weighted_traj_idx_sample(self) -> int:
        """Sample a trajectory index based on weights."""
        return int(np.random.choice(self.trajectory_idxs, p=self.trajectory_weights))

    def weighted_traj_idx_sample_batch(self, size: int) -> np.ndarray:
        """Sample a batch of trajectory indices."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx: int) -> float:
        """Sample a random time for a trajectory."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0.0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs: np.ndarray) -> np.ndarray:
        """Sample random times for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    @staticmethod
    def slerp(frame1: torch.Tensor, frame2: torch.Tensor, blend: torch.Tensor | float) -> torch.Tensor:
        """Spherical linear interpolation between frames."""
        return (1.0 - blend) * frame1 + blend * frame2

    def get_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
        """Get a single frame at a specific time from a trajectory."""
        n = self.trajectories[traj_idx].shape[0]
        frame = np.clip(float(time) / float(self.trajectory_frame_durations[traj_idx]), 0.0, n - 1)
        idx_low = int(np.floor(frame))
        idx_high = min(idx_low + 1, n - 1)
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = frame - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        """Get frames at specific times for multiple trajectories."""
        n = self.trajectory_num_frames[traj_idxs].astype(np.int64)
        frame = np.clip(times / self.trajectory_frame_durations[traj_idxs], 0.0, n - 1)
        idx_low = np.floor(frame).astype(np.int64)
        idx_high = np.minimum(idx_low + 1, n - 1)

        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]

        blend = torch.tensor(frame - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
        """Get full AMP frame at a specific time."""
        n = self.trajectories_full[traj_idx].shape[0]
        frame = np.clip(float(time) / float(self.trajectory_frame_durations[traj_idx]), 0.0, n - 1)
        idx_low = int(np.floor(frame))
        idx_high = min(idx_low + 1, n - 1)
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = frame - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        """Get full AMP frames at specific times for multiple trajectories."""
        n = self.trajectory_num_frames[traj_idxs].astype(np.int64)
        frame = np.clip(times / self.trajectory_frame_durations[traj_idxs], 0.0, n - 1)
        idx_low = np.floor(frame).astype(np.int64)
        idx_high = np.minimum(idx_low + 1, n - 1)

        amp_obs_dim = self.amp_obs_dim
        all_frame_starts = torch.zeros(len(traj_idxs), amp_obs_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), amp_obs_dim, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]

        blend = torch.tensor(frame - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Generate batches of AMP transitions for discriminator training.

        Yields:
            Tuple of (state, next_state) tensors.
        """
        for batch_idx in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs]
                s_next = self.preloaded_s_next[idxs]
            else:
                if self.expert_trajectory_sampling_mode == "round_robin":
                    # Match AMP_mjlab: each discriminator expert minibatch comes
                    # from one clip, and clips are traversed deterministically.
                    traj_idx = self.trajectory_idxs[batch_idx % len(self.trajectory_idxs)]
                    traj_idxs = np.full(mini_batch_size, traj_idx, dtype=np.int64)
                else:
                    traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                if self.expert_sampling_mode == "adjacent":
                    s, s_next = self.get_adjacent_frame_batch(traj_idxs)
                else:
                    times = self.traj_time_sample_batch(traj_idxs)
                    s = self.get_full_frame_at_time_batch(traj_idxs, times)
                    s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)

            yield s, s_next

    @property
    def observation_dim(self) -> int:
        """Dimension of AMP observations (joint pos + joint vel + end effector pos)."""
        return self.trajectories[0].shape[1]

    @property
    def amp_obs_dim(self) -> int:
        """Dimension of AMP-specific observations."""
        return self._amp_obs_dim

    @property
    def num_motions(self) -> int:
        """Number of loaded motion files."""
        return len(self.trajectory_names)

    # Utility methods for extracting specific parts of observations
    @staticmethod
    def get_joint_pose(pose: torch.Tensor) -> torch.Tensor:
        """Extract joint positions from pose."""
        return pose[AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses: torch.Tensor) -> torch.Tensor:
        """Extract joint positions from batch of poses."""
        return poses[:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_vel(pose: torch.Tensor) -> torch.Tensor:
        """Extract joint velocities from pose."""
        return pose[AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses: torch.Tensor) -> torch.Tensor:
        """Extract joint velocities from batch of poses."""
        return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_end_pos(pose: torch.Tensor) -> torch.Tensor:
        """Extract end effector positions from pose."""
        return pose[AMPLoader.END_POS_START_IDX:AMPLoader.END_POS_END_IDX]

    @staticmethod
    def get_end_pos_batch(poses: torch.Tensor) -> torch.Tensor:
        """Extract end effector positions from batch of poses."""
        return poses[:, AMPLoader.END_POS_START_IDX:AMPLoader.END_POS_END_IDX]
