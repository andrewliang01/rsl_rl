# SPDX-License-Identifier: BSD-3-Clause
"""Chitu Max's original 61-column mocap and 43-dimensional AMP contract.

No Isaac runtime dependency. Quaternions use xyzw, as in the source mocap and EA3.
"""

from pathlib import Path
import json
import numpy as np
import torch

JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf"))
FK_PARAMS = {
    "hip_offsets": [[0.301, 0.10557, 0], [0.301, -0.10557, 0], [-0.301, 0.10557, 0], [-0.301, -0.10557, 0]],
    "hfe_offsets": [[0.0795, 0.0272, 0], [0.0795, -0.0272, 0], [-0.0795, 0.0272, 0], [-0.0795, -0.0272, 0]],
    "kfe_offsets": [
        [-0.24748, 0.12015, -0.24752],
        [-0.24748, -0.12015, -0.24752],
        [-0.24748, 0.12015, -0.24752],
        [-0.24748, -0.12015, -0.24752],
    ],
    "foot_offsets": [
        [0.31382, -0.00056, -0.1566],
        [0.31376, -0.00266, -0.15669],
        [0.31382, -0.00056, -0.1566],
        [0.31375, -0.00256, -0.1567],
    ],
}


def quat_apply(q, v):
    xyz, w = q[..., :3], q[..., 3:4]
    return v + 2 * (w * torch.cross(xyz, v, dim=-1) + torch.cross(xyz, torch.cross(xyz, v, dim=-1), dim=-1))


def quat_inverse_apply(q, v):
    return quat_apply(torch.cat((-q[..., :3], q[..., 3:4]), -1), v)


def foot_fk(joint_pos, params=FK_PARAMS):
    """Source Rx(hip) Ry(thigh) Ry(calf) chain, in FL/FR/RL/RR order."""
    q = joint_pos.reshape(*joint_pos.shape[:-1], 4, 3)
    offsets = [
        torch.as_tensor(params[k], device=q.device, dtype=q.dtype)
        for k in ("hip_offsets", "hfe_offsets", "kfe_offsets", "foot_offsets")
    ]

    def ry(v, a):
        x, y, z = v.unbind(-1)
        c, s = a.cos(), a.sin()
        return torch.stack((c * x + s * z, y.expand_as(a), -s * x + c * z), -1)

    def rx(v, a):
        x, y, z = v.unbind(-1)
        c, s = a.cos(), a.sin()
        return torch.stack((x, c * y - s * z, s * y + c * z), -1)

    local = offsets[1] + ry(offsets[2], q[..., 1]) + ry(offsets[3], q[..., 1] + q[..., 2])
    return (offsets[0] + rx(local, q[..., 0])).flatten(-2)


class GMQMotionLoader:
    """Online sampling with the source branch's time distribution and interpolation.

    Source caches two million sampled transitions; this loader samples the same
    distribution online to retain the workspace's AMP memory model.
    """

    def __init__(self, device, time_between_frames, motion_files, **_):
        self.device = device
        self.time_between_frames = time_between_frames
        files = [motion_files] if isinstance(motion_files, str) else motion_files
        self.trajectory_files = sorted(
            str(f) for p in files for f in (Path(p).glob("*.txt") if Path(p).is_dir() else [Path(p)])
        )
        self.trajectory_names = [Path(p).stem for p in self.trajectory_files]
        self.trajectories_full = []
        weights = []
        durations = []
        for path in self.trajectory_files:
            data = json.loads(Path(path).read_text())
            a = np.asarray(data["Frames"], dtype=np.float64)
            dt = float(data["FrameDuration"])
            if a.ndim != 2 or a.shape[1] < 49 or len(a) < 2 or dt <= 0 or not np.isfinite(a).all():
                raise ValueError(f"Invalid GMQ motion: {path}")
            for start in (7, 19, 37, 49):
                if a.shape[1] >= start + 12:
                    a[:, start : start + 12] = (
                        a[:, start : start + 12].reshape(-1, 4, 3)[:, [1, 0, 3, 2]].reshape(-1, 12)
                    )
            a[:, 37:49] = np.gradient(a[:, 7:19], dt, axis=0)
            q = a[:, 3:7]
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            q[q[:, 3] < 0] *= -1
            frame = torch.as_tensor(a[:, :49], dtype=torch.float32, device=device)
            frame[:, 19:31] = foot_fk(frame[:, 7:19])
            self.trajectories_full.append(frame)
            durations.append(dt)
            weights.append(float(data["MotionWeight"]))
        if not self.trajectories_full:
            raise ValueError("No Chitu Max mocap files found")
        self.trajectory_frame_durations = np.asarray(durations)
        self.trajectory_num_frames = np.asarray([len(f) for f in self.trajectories_full])
        self.trajectory_lens = (self.trajectory_num_frames - 1) * self.trajectory_frame_durations
        if np.any(self.trajectory_lens < time_between_frames):
            raise ValueError("GMQ clips must span one policy step")
        self.trajectory_weights = np.asarray(weights, dtype=float)
        if (
            not np.isfinite(self.trajectory_weights).all()
            or np.any(self.trajectory_weights < 0)
            or self.trajectory_weights.sum() <= 0
        ):
            raise ValueError("Invalid GMQ motion weights")
        self.trajectory_weights /= self.trajectory_weights.sum()
        self.trajectory_idxs = np.arange(len(weights))
        self._offsets = np.r_[0, np.cumsum(self.trajectory_num_frames[:-1])]
        self._packed = torch.cat(self.trajectories_full)
        self.amp_obs_dim = self.observation_dim = 43

    def weighted_traj_idx_sample_batch(self, size):
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights)

    def traj_time_sample_batch(self, ids):
        return np.maximum(
            0,
            self.trajectory_lens[ids] * np.random.uniform(size=len(ids))
            - self.time_between_frames
            - self.trajectory_frame_durations[ids],
        )

    def full_frame_batch(self, ids, times):
        # Preserve source p*n interpolation (rather than silently correcting it to time/frame_dt).
        f = np.clip(
            times / self.trajectory_lens[ids] * self.trajectory_num_frames[ids], 0, self.trajectory_num_frames[ids] - 1
        )
        lo = np.floor(f).astype(int)
        hi = np.minimum(np.ceil(f).astype(int), self.trajectory_num_frames[ids] - 1)
        indices = torch.as_tensor(np.stack((self._offsets[ids] + lo, self._offsets[ids] + hi)), device=self.device)
        a, b = self._packed[indices]
        t = torch.as_tensor(f - lo, dtype=a.dtype, device=self.device).unsqueeze(1)
        frame = (1 - t) * a + t * b
        # Root reset uses shortest-arc quaternion slerp, matching source full-frame interpolation.
        qa, qb = a[:, 3:7], b[:, 3:7]
        dot = (qa * qb).sum(-1, keepdim=True)
        qb = torch.where(dot < 0, -qb, qb)
        dot = dot.abs().clamp(0, 1)
        theta = dot.acos()
        sin = theta.sin().clamp_min(1e-8)
        qs = ((1 - t) * theta).sin() / sin * qa + (t * theta).sin() / sin * qb
        qs = torch.where(dot > 0.9995, (1 - t) * qa + t * qb, qs)
        frame[:, 3:7] = torch.nn.functional.normalize(qs, dim=-1)
        return frame

    @staticmethod
    def amp_features(frame):
        return torch.cat((frame[:, 7:49], frame[:, 2:3]), dim=-1)

    def get_full_frame_batch(self, size):
        ids = self.weighted_traj_idx_sample_batch(size)
        return self.full_frame_batch(ids, self.traj_time_sample_batch(ids))

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = self.weighted_traj_idx_sample_batch(mini_batch_size)
            times = self.traj_time_sample_batch(ids)
            yield (
                self.amp_features(self.full_frame_batch(ids, times)),
                self.amp_features(self.full_frame_batch(ids, times + self.time_between_frames)),
            )

    @property
    def num_motions(self):
        return len(self.trajectories_full)


class GMQNormalizer(torch.nn.Module):
    """Source running moments and sqrt(var + epsilon), checkpointable on device."""

    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("var", torch.ones(size, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1.0e-4, dtype=torch.float64))

    @property
    def std(self):
        return (self.var + 1.0e-4).sqrt()

    def forward(self, x):
        return (x - self.mean.float()) / self.std.float()

    @torch.no_grad()
    def update(self, x):
        x = x.double()
        n = x.shape[0]
        delta = x.mean(0) - self.mean
        total = self.count + n
        self.var.copy_(
            (self.var * self.count + x.var(0, unbiased=False) * n + delta.square() * self.count * n / total) / total
        )
        self.mean.add_(delta * n / total)
        self.count.copy_(total)
