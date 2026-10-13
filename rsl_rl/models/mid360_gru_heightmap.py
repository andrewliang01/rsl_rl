"""MSE-trained recurrent mapping in front of an immutable five-map ECMM actor."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from rsl_rl.modules.elevation_2D_cnn_encoder import Elevation2DCNNEncoder


def build_ecmm_actor(obs, config):
    """Construct the original actor, dropping only deprecated runner metadata."""
    from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
    config = dict(config)
    for key in ("class_name", "stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
        config.pop(key, None)
    return PropMLPElevationFusionModel(
        obs, {"actor": ["policy", "height_scan_policy"]}, "actor", 29, **config,
    ).to(obs["policy"].device)


class CaptureAlignedHistory:
    """Atomically retain depth, proprioception and time at distinct captures.

    The policy receives one complete observation snapshot per control step.
    Never recompute proprioception (or its noise) to populate this history.
    Startup padding repeats the entire first packet, including its timestamp.
    """
    def __init__(self, frames, stride):
        if frames < 1 or stride < 1:
            raise ValueError("Capture history frames and stride must be positive.")
        self.frames, self.stride = frames, stride
        self.capacity = (frames - 1) * stride + 1
        self.ready = None

    @torch.no_grad()
    def update(self, depth, proprio, capture_id, time_s):
        capture_id = capture_id.reshape(-1).long()
        time_s = time_s.reshape(-1).double()
        if self.ready is None:
            batch, device = depth.shape[0], depth.device
            self.depth = depth.new_zeros(batch, self.capacity, *depth.shape[1:])
            self.proprio = proprio.new_zeros(batch, self.capacity, proprio.shape[-1])
            self.times = time_s.new_zeros(batch, self.capacity)
            self.ready = torch.zeros(batch, device=device, dtype=torch.bool)
            self.last_capture = torch.full((batch,), -1, device=device, dtype=torch.long)
            self.write = torch.zeros(batch, device=device, dtype=torch.long)
        delta = capture_id - self.last_capture
        if ((delta < 0) & self.ready).any() or ((delta > 1) & self.ready).any():
            raise ValueError("Missed/reset Livox captures: call policy each control step and reset done environments.")
        first = ~self.ready
        self.depth[first] = depth[first, None]
        self.proprio[first] = proprio[first, None]
        self.times[first] = time_s[first, None]
        fresh = first | (delta != 0)
        ids = fresh.nonzero(as_tuple=True)[0]
        index = self.write[ids]
        self.depth[ids, index] = depth[ids]
        self.proprio[ids, index] = proprio[ids]
        self.times[ids, index] = time_s[ids]
        self.write[ids] = (index + 1) % self.capacity
        self.last_capture[ids] = capture_id[ids]
        self.ready[ids] = True
        offsets = torch.arange(self.frames, device=depth.device) * self.stride
        indices = (self.write[:, None] + offsets[None]) % self.capacity
        rows = torch.arange(depth.shape[0], device=depth.device)[:, None]
        timestamps = self.times[rows, indices]
        ages = (time_s[:, None] - timestamps).to(depth.dtype)
        if (ages < -1.0e-6).any():
            raise ValueError("Capture history contains a future observation timestamp.")
        return self.depth[rows, indices], self.proprio[rows, indices], ages, timestamps

    def reset(self, dones):
        if self.ready is not None:
            self.ready[dones] = False
            self.last_capture[dones] = -1
            self.write[dones] = 0
            self.depth[dones] = 0
            self.proprio[dones] = 0
            self.times[dones] = 0


class Mid360GRUHeightmap(nn.Module):
    def __init__(self, depth_frames=10, proprio_history=5, proprio_dim=96,
                 hidden_size=256, num_layers=1, cnn_hidden_dims=(16, 32),
                 cnn_kernel_sizes=(3, 3), cnn_strides=(2, 2), capture_stride=1):
        super().__init__()
        self.proprio_history = proprio_history
        self.proprio_dim = proprio_dim
        self.depth_frames = depth_frames
        self.capture_stride = capture_stride
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if min(depth_frames, proprio_history, proprio_dim, hidden_size, num_layers, capture_stride) < 1:
            raise ValueError("All estimator dimensions must be positive.")
        self.depth_encoder = Elevation2DCNNEncoder(
            in_channels=depth_frames, hidden_dims=cnn_hidden_dims,
            kernel_sizes=cnn_kernel_sizes, strides=cnn_strides,
            out_dim=64, vision_spatial_size=(16, 96),
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_history * proprio_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(),
        )
        self.aligned_proprio_encoder = nn.Sequential(
            nn.Linear(depth_frames * (proprio_dim + 1), 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(),
        )
        self.gru = nn.GRU(192, hidden_size, num_layers=num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, 256), nn.ELU(), nn.Linear(256, 28 * 20))

    def forward(self, depth, proprio_history, aligned_proprio, frame_ages, hidden=None):
        if depth.shape[1:] != (self.depth_frames, 16, 96):
            raise ValueError(f"Unexpected depth history shape: {tuple(depth.shape)}")
        if proprio_history.shape[1:] != (self.proprio_history, self.proprio_dim):
            raise ValueError(f"Unexpected proprio history shape: {tuple(proprio_history.shape)}")
        if aligned_proprio.shape[1:] != (self.depth_frames, self.proprio_dim):
            raise ValueError(f"Unexpected aligned proprio shape: {tuple(aligned_proprio.shape)}")
        if frame_ages.shape != depth.shape[:2]:
            raise ValueError(f"Unexpected capture ages shape: {tuple(frame_ages.shape)}")
        # Range images are distances in metres, with misses filled by 40 m.
        visual = self.depth_encoder(depth.clamp(0.05, 40.0) / 40.0)
        proprio = self.proprio_encoder(proprio_history.flatten(1))
        aligned = self.aligned_proprio_encoder(torch.cat((aligned_proprio, frame_ages[..., None]), -1).flatten(1))
        recurrent, hidden = self.gru(torch.cat((visual, aligned, proprio), -1).unsqueeze(0), hidden)
        return self.decoder(recurrent[0]).reshape(-1, 28, 20), hidden


def load_frozen_actor(actor, checkpoint_path, actor_key="actor_state_dict"):
    """Require an actual compatible actor including its normalization buffers."""
    if not checkpoint_path:
        raise ValueError("checkpoint_path is empty; supply the trained ECMM .pt before training.")
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or actor_key not in checkpoint:
        raise ValueError(f"ECMM checkpoint must contain {actor_key!r}.")
    actor.load_state_dict(checkpoint[actor_key], strict=True)
    actor.requires_grad_(False)
    actor.eval()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest


class FrozenECMMMapPolicy(nn.Module):
    """Only estimator predictions have gradients; action generation never does.

    Call once per 50 Hz control step. Reset done environments after env.step,
    and detach recurrent state at TBPTT boundaries, without clearing memory.
    """
    def __init__(self, actor, estimator):
        super().__init__()
        if actor.elevation_history_length != 5 or tuple(actor.vision_spatial_size) != (28, 20):
            raise ValueError("Frozen ECMM must consume five 28x20 maps.")
        self.actor = actor.requires_grad_(False).eval()
        self.estimator = estimator
        self.capture_history = CaptureAlignedHistory(estimator.depth_frames, estimator.capture_stride)
        self.hidden = None
        self.proprio_buffer = None
        self.map_buffer = None
        self.ready = None

    def train(self, mode=True):
        super().train(mode)
        self.actor.eval()  # also freeze BatchNorm statistics and observation normalization
        return self

    def forward(self, obs, use_teacher=False):
        # Clone inputs: the simulator can overwrite its observation tensors
        # before the accumulated supervised losses are backpropagated.
        with torch.no_grad():
            current = self.actor.obs_normalizer(obs["policy"].detach().clone())
            if self.ready is None:
                self.ready = torch.zeros(current.shape[0], device=current.device, dtype=torch.bool)
                self.proprio_buffer = current[:, None].repeat(1, self.estimator.proprio_history, 1)
                self.map_buffer = current.new_zeros(current.shape[0], 5, 28, 20)
            first = ~self.ready
            self.proprio_buffer = torch.cat((self.proprio_buffer[:, 1:], current[:, None]), dim=1)
            self.proprio_buffer[first] = current[first, None]
            depth, aligned, ages, _ = self.capture_history.update(
                obs["depth_history"][:, -1], current, obs["depth_capture_id"], obs["observation_time"],
            )
        prediction, self.hidden = self.estimator(
            depth, self.proprio_buffer.clone(), aligned, ages, self.hidden,
        )
        with torch.no_grad():
            self.map_buffer = torch.cat((self.map_buffer[:, 1:], prediction.detach()[:, None]), dim=1)
            self.map_buffer[first] = prediction.detach()[first, None]
            self.ready[:] = True
            maps = obs["height_scan_policy"] if use_teacher else self.map_buffer
            actions = self.actor({"policy": obs["policy"], "height_scan_policy": maps})
        return actions, prediction

    def reset(self, dones):
        if self.ready is None:
            return
        dones = dones.bool().reshape(-1)
        self.ready[dones] = False
        self.capture_history.reset(dones)
        if self.hidden is not None:
            # Out-of-place mask preserves the graph for nonterminal sequences.
            self.hidden = self.hidden * (~dones)[None, :, None]
        self.proprio_buffer[dones] = 0
        self.map_buffer[dones] = 0

    def detach_hidden(self):
        if self.hidden is not None:
            self.hidden = self.hidden.detach()


@torch.no_grad()
def broadcast_estimator(estimator):
    """Start every worker from rank zero's parameters and normalization state."""
    for tensor in estimator.state_dict().values():
        dist.broadcast(tensor, src=0)


def update_estimator(policy, optimizer, losses, *, distributed=False):
    """One supervised update; no PPO, critic or action loss is involved."""
    if not losses:
        raise ValueError("Cannot update without supervised losses.")
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack(losses).mean()
    finite = torch.isfinite(loss).to(torch.int32)
    if distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError("Non-finite heightmap MSE.")
    loss.backward()
    if distributed:
        # TBPTT retains multiple forwards in a single graph. Reduce once after
        # backward, before clipping/Adam, so each rank applies the same update.
        parameters = [p for p in policy.estimator.parameters() if p.requires_grad]
        gradient = torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in parameters
        ])
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(dist.get_world_size())
        offset = 0
        for parameter in parameters:
            parameter.grad = gradient[offset:offset + parameter.numel()].view_as(parameter)
            offset += parameter.numel()
    torch.nn.utils.clip_grad_norm_(policy.estimator.parameters(), 1.0, error_if_nonfinite=True)
    if any(p.requires_grad or p.grad is not None for p in policy.actor.parameters()):
        raise RuntimeError("Frozen actor gradient boundary was violated.")
    optimizer.step()
    policy.detach_hidden()
    return float(loss.detach())
