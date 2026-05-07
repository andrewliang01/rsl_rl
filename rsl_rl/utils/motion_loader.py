# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion loader for Adversarial Motion Prior (AMP)."""

import json
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

    def __init__(
        self,
        device: torch.device,
        time_between_frames: float,
        motion_files: list[str],
        preload_transitions: bool = True,
        num_preload_transitions: int = 100000,
    ) -> None:
        self.device = device
        self.time_between_frames = time_between_frames
        self.preload_transitions = preload_transitions
        self.num_preload_transitions = num_preload_transitions

        # Load trajectories from motion files
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Trajectory length in seconds
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                # Store full trajectory (joint pos, joint vel, end effector pos)
                self.trajectories.append(
                    torch.tensor(motion_data[:, :AMPLoader.END_POS_END_IDX], dtype=torch.float32, device=device)
                )
                self.trajectories_full.append(
                    torch.tensor(motion_data[:, :AMPLoader.END_POS_END_IDX], dtype=torch.float32, device=device)
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json.get("MotionWeight", 1.0)))

                frame_duration = float(motion_json.get("FrameDuration", 0.02))
                self.trajectory_frame_durations.append(frame_duration)

                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

                print(f"[AMPLoader] Loaded {traj_len:.2f}s motion from {motion_file}")

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

    def _preload_transitions(self) -> None:
        """Preload transitions into memory for faster sampling."""
        traj_idxs = self.weighted_traj_idx_sample_batch(self.num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)

        self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
        self.preloaded_s_next = self.get_full_frame_at_time_batch(
            traj_idxs, times + self.time_between_frames
        )

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
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low = int(np.floor(p * n))
        idx_high = int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        """Get frames at specific times for multiple trajectories."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = np.floor(p * n).astype(np.int64)
        idx_high = np.ceil(p * n).astype(np.int64)

        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
        """Get full AMP frame at a specific time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low = int(np.floor(p * n))
        idx_high = int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low

        # Extract AMP observations (joint pos, joint vel, end effector pos)
        joints0 = frame_start[AMPLoader.JOINT_POSE_START_IDX:AMPLoader.END_POS_END_IDX]
        joints1 = frame_end[AMPLoader.JOINT_POSE_START_IDX:AMPLoader.END_POS_END_IDX]
        return self.slerp(joints0, joints1, blend)

    def get_full_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        """Get full AMP frames at specific times for multiple trajectories."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = np.floor(p * n).astype(np.int64)
        idx_high = np.ceil(p * n).astype(np.int64)

        amp_obs_dim = AMPLoader.END_POS_END_IDX - AMPLoader.JOINT_POSE_START_IDX
        all_frame_starts = torch.zeros(len(traj_idxs), amp_obs_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), amp_obs_dim, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask], AMPLoader.JOINT_POSE_START_IDX:AMPLoader.END_POS_END_IDX]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask], AMPLoader.JOINT_POSE_START_IDX:AMPLoader.END_POS_END_IDX]

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Generate batches of AMP transitions for discriminator training.

        Yields:
            Tuple of (state, next_state) tensors.
        """
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = torch.randint(self.preloaded_s.shape[0], (mini_batch_size,), device=self.device)
                s = self.preloaded_s.index_select(0, idxs)
                s_next = self.preloaded_s_next.index_select(0, idxs)
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
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
        return AMPLoader.END_POS_END_IDX - AMPLoader.JOINT_POSE_START_IDX

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
