"""Original AMP-PPO with a detached, independently learned range-JEPA front end.

PPO storage contains the original scalar observations plus range_features[512].
The auxiliary image sequences have their own fresh-capture storage and optimizer;
there is no perception network evaluation inside the original PPO minibatches.
"""
from __future__ import annotations

from copy import deepcopy
import time
import inspect
import json
import torch
from torch import nn
from tensordict import TensorDict

from .amp_ppo import AMPPPO
from .ppo_builders import construct_multi_critic_algorithm
from rsl_rl.models.range_jepa import RangeJEPA
from rsl_rl.storage.perception_storage import PerceptionStorage
from rsl_rl.utils.range_aux_profiler import RangeAuxProfiler
from rsl_rl.utils.range_online_profiler import RangeOnlineProfiler

def _model_configuration(perception_cfg):
    cfg = deepcopy(perception_cfg or {})
    for key in ("num_learning_epochs", "tbptt_steps", "sequence_batch_size", "learning_rate", "max_grad_norm", "weights", "profile_timing", "profile_aux_breakdown", "profile_online_breakdown"):
        cfg.pop(key, None)
    model_cfg = cfg.pop("model", {})
    if set(model_cfg) & set(cfg):
        raise ValueError("Duplicate perception model configuration keys.")
    bound = inspect.signature(RangeJEPA).bind(**{**cfg, **model_cfg})
    bound.apply_defaults()
    return dict(bound.arguments)


RAW_RANGE_KEYS = frozenset(("range_obs", "range_map", "range_sim", "range_capture_id", "range_episode_id"))


def compact_range_observations(obs: TensorDict, features: torch.Tensor) -> TensorDict:
    """Create a small observation dictionary without cloning or retaining images."""
    return TensorDict({**{k: v for k, v in obs.items() if k not in RAW_RANGE_KEYS},
                       "range_features": features.detach()}, batch_size=obs.batch_size, device=obs.device)


class RangeJEPAInferencePolicy(nn.Module):
    """Stateful raw-depth/mask → cached detached features → original actor.

    Call reset(dones) when the environment does not supply range_episode_id.
    Episode IDs reset automatically. Held captures reuse features and hidden
    state; negative capture IDs represent no capture yet. The map/sim targets are
    never needed for inference. Runtime memory is intentionally not a checkpoint
    tensor: loading a policy starts a new environment rollout.
    """
    def __init__(self, actor: nn.Module, perception: RangeJEPA) -> None:
        super().__init__()
        self.actor = actor
        self.perception = perception
        for name in ("hidden", "features", "last_capture", "episode"):
            self.register_buffer(name, None, persistent=False)

    @property
    def output_std(self):
        return self.actor.output_std

    @property
    def is_recurrent(self):
        return True  # the deployed depth front end has state even though PPO actor is feedforward

    @torch.inference_mode(False)
    @torch.no_grad()
    def clear_memory(self) -> None:
        self.hidden = self.features = self.last_capture = self.episode = None

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset_memory(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self.clear_memory()
            return
        if self.hidden is not None:
            mask = dones.to(device=self.hidden.device, dtype=torch.bool).reshape(-1)
            self.hidden[mask] = 0
            self.features[mask] = 0
            self.last_capture[mask] = -1
            # Without an explicit episode ID this is the episode identity source.
            self.episode[mask] += 1
    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.reset_memory(dones)
        self.actor.reset(dones)

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare(self, obs: TensorDict, record=None, record_indexed=None, profile=None) -> TensorDict:
        if profile is not None:
            profile.mark("validate_select")
        if record is not None and record_indexed is not None:
            raise ValueError("Choose either a selected-row or indexed capture recorder.")
        if "range_obs" not in obs:
            if "range_features" not in obs:
                raise KeyError("Expected raw range_obs/capture_id or cached range_features.")
            if profile is not None:
                profile.mark("publish")
            result = compact_range_observations(obs, obs["range_features"])
            if profile is not None:
                profile.mark()
            return result
        image = obs["range_obs"]
        if image.ndim != 4 or image.shape[1] != 2:
            raise ValueError("range_obs must be [B,2,H,W] depth and effective-validity mask.")
        b = image.shape[0]
        capture = obs["range_capture_id"]
        if capture.dtype != torch.long or capture.shape != (b, 1):
            raise ValueError("range_capture_id must be int64 [B,1].")
        capture = capture.reshape(-1)
        explicit_episode = obs.get("range_episode_id", None)
        if explicit_episode is not None:
            if explicit_episode.dtype != torch.long or explicit_episode.shape != (b, 1):
                raise ValueError("range_episode_id must be int64 [B,1].")
            explicit_episode = explicit_episode.reshape(-1)
            if torch.any(explicit_episode < 0):
                raise ValueError("range_episode_id cannot be negative.")
        if self.hidden is None:
            self.hidden = self.perception.initial_hidden(b, device=image.device, dtype=image.dtype)
            self.features = image.new_zeros(b, 512)
            self.last_capture = torch.full((b,), -1, device=image.device, dtype=torch.long)
            self.episode = torch.zeros(b, device=image.device, dtype=torch.long)
            if explicit_episode is not None:
                self.episode.copy_(explicit_episode)
        if self.hidden.shape[0] != b:
            raise ValueError("Inference batch size changed; call policy.reset() first.")
        if self.hidden.device != image.device:
            raise ValueError("Observation and perception state devices disagree.")
        changed = torch.zeros(b, device=image.device, dtype=torch.bool) if explicit_episode is None else explicit_episode != self.episode
        if torch.any((~changed) & (capture >= 0) & (capture < self.last_capture)):
            raise ValueError("Capture ID moved backwards within the same episode.")
        self.hidden[changed] = 0
        self.features[changed] = 0
        self.last_capture[changed] = -1
        if explicit_episode is not None:
            self.episode.copy_(explicit_episode)
        fresh = (capture >= 0) & (capture != self.last_capture)
        indices = fresh.nonzero(as_tuple=False).flatten()
        if indices.numel():
            if profile is not None:
                profile.mark("cache_enqueue")
            if record is not None or record_indexed is not None:
                for key in ("range_map", "range_sim"):
                    if key not in obs or obs[key].shape != image.shape:
                        raise ValueError(f"Training requires aligned {key} with range_obs shape.")
            if record_indexed is not None:
                # The recorder owns one independent selection of each source.
                # Encoding is read-only, so online inference can reuse its obs
                # and PRE hidden snapshots without another gather or clone.
                saved = record_indexed(image, obs["range_map"], obs["range_sim"],
                                       self.hidden, indices, self.episode, capture)
                fresh_image, pre_hidden = saved["observed"], saved["pre_hidden"]
            else:
                fresh_image = image.index_select(0, indices)
                pre_hidden = self.hidden.index_select(0, indices)
                if record is not None:
                    record(fresh_image, obs["range_map"][indices], obs["range_sim"][indices],
                           pre_hidden, indices, self.episode[indices], capture[indices])
            if profile is None:
                prediction = self.perception.step(fresh_image, pre_hidden)
            else:
                profile.mark("online_cnn")
                encoded = self.perception.encode(fresh_image)
                profile.mark("memory_attention")
                prediction = self.perception._step_encoded(encoded, pre_hidden)
            if profile is not None:
                profile.mark("publish")
            if prediction["features"].shape != (indices.numel(), 512):
                raise ValueError("RangeJEPA must produce [fresh,512] cached features.")
            self.hidden[indices] = prediction["hidden"].detach()
            self.features[indices] = prediction["features"].detach()
            self.last_capture[indices] = capture[indices]
        if profile is not None and not indices.numel():
            profile.mark("publish")
        # New tensor ownership is essential: env.step/reset must not mutate PPO's
        # previous transition observations or inference tensors saved for autograd.
        result = compact_range_observations(obs, self.features.clone())
        if profile is not None:
            profile.mark()
        return result

    def forward(self, obs, *args, **kwargs):
        return self.actor(self.prepare(obs), *args, **kwargs)

    def as_jit(self):
        raise NotImplementedError("Stateful raw-range export needs explicit hidden/capture I/O; export actor and perception separately.")

    def as_onnx(self, **kwargs):
        raise NotImplementedError("Stateful raw-range export needs explicit hidden/capture I/O; export actor and perception separately.")


class RangeJEPAAMPPPO(AMPPPO):
    """Add separate range memory learning to AMPPPO without replacing its loops."""
    def __init__(self, actor, critic, storage, perception_cfg: dict | None = None, **kwargs):
        super().__init__(actor, critic, storage, **kwargs)
        cfg = deepcopy(perception_cfg or {})
        self.perception_cfg = deepcopy(cfg)
        self.perception_epochs = cfg.pop("num_learning_epochs", 5)
        self.perception_tbptt_steps = cfg.pop("tbptt_steps", 8)
        self.perception_sequence_batch_size = cfg.pop("sequence_batch_size", 128)
        for name in ("perception_epochs", "perception_tbptt_steps", "perception_sequence_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        lr = cfg.pop("learning_rate", 1e-3)
        self.perception_max_grad_norm = cfg.pop("max_grad_norm", 1.0)
        if lr <= 0 or self.perception_max_grad_norm <= 0:
            raise ValueError("Perception learning rate and max_grad_norm must be positive.")
        self.profile_timing = bool(cfg.pop("profile_timing", False))
        self.profile_aux_breakdown = bool(cfg.pop("profile_aux_breakdown", False))
        self.profile_online_breakdown = bool(cfg.pop("profile_online_breakdown", False))
        self._online_profile = RangeOnlineProfiler(self.profile_online_breakdown, self.device)
        self.perception_weights = cfg.pop("weights", None)
        model_cfg = cfg.pop("model", {})
        if set(model_cfg) & set(cfg):
            raise ValueError("Duplicate perception model configuration keys.")
        self.perception_model_cfg = _model_configuration(perception_cfg)
        self.perception = RangeJEPA(**self.perception_model_cfg).to(self.device)
        self.perception_optimizer = torch.optim.Adam(self.perception.parameters(), lr=lr)
        ppo_params = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        if ppo_params & {id(p) for p in self.perception.parameters()}:
            raise RuntimeError("Perception parameters must be disjoint from original PPO/AMP optimizers.")
        self.perception_storage = PerceptionStorage()
        self._live_policy = RangeJEPAInferencePolicy(self.actor, self.perception)
        self._inference_policy = RangeJEPAInferencePolicy(self.actor, self.perception)
        self.perception_update_count = 0
        self.last_perception_receipt: dict = {}

    def act(self, obs: TensorDict) -> torch.Tensor:
        if "range_obs" not in obs:
            raise KeyError("Training act requires raw range_obs; only PPO minibatches consume cached features.")
        if not self.profile_online_breakdown:
            return super().act(self._live_policy.prepare(obs, record_indexed=self.perception_storage.add_indexed))
        compact = self._live_policy.prepare(obs, record_indexed=self.perception_storage.add_indexed, profile=self._online_profile)
        self._online_profile.mark("parent_act")
        try:
            return super().act(compact)
        finally:
            self._online_profile.mark()

    def process_env_step(self, obs, rewards, dones, extras):
        # obs is NEXT frame. Never run/ack its new capture here; the next act
        # consumes it exactly once. Scalar normalization remains original.
        compact = compact_range_observations(obs, self._live_policy.features.clone())
        super().process_env_step(compact, rewards, dones, extras)
        self._live_policy.reset_memory(dones)

    def compute_returns(self, obs):
        # Only the critic bootstraps; no perception advance on the final frame.
        return super().compute_returns(compact_range_observations(obs, self._live_policy.features.clone()))

    def _timing_barrier(self):
        if (self.profile_timing or self.profile_aux_breakdown) and torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

    def update(self):
        # Read online CUDA events once at the rollout boundary, before learning.
        self._timing_barrier()
        online_metrics = self._online_profile.finish(synchronize=not (self.profile_timing or self.profile_aux_breakdown))
        begin = time.perf_counter()
        loss = super().update()
        self._timing_barrier()
        parent_s = time.perf_counter() - begin
        begin = time.perf_counter()
        loss.update(self._update_perception())
        if not self.profile_aux_breakdown:  # breakdown.finish already synchronized once
            self._timing_barrier()
        aux_s = time.perf_counter() - begin
        self.last_perception_receipt.update(parent_AMPPPO_s=parent_s, aux_total_s=aux_s,
                                            timing_gpu_synchronized=(self.profile_timing or self.profile_aux_breakdown))
        if self.profile_aux_breakdown:
            prefix = f"range_profile/rank{self.gpu_global_rank}/"
            # Include profiler export/receipt overhead in the same measured
            # outer wall total, retaining exact additive top-level accounting.
            loss[prefix + "aux_wall_s"] = aux_s
            loss[prefix + "other_s"] = aux_s - sum(loss[prefix + name + "_s"] for name in RangeAuxProfiler.TOP_PHASES)
            self.last_perception_receipt["profile_aux_breakdown"] = {k[len(prefix):]: v for k, v in loss.items() if k.startswith(prefix)}
        loss.update({"range_jepa/parent_AMPPPO_s": parent_s, "range_jepa/aux_total_s": aux_s})
        if online_metrics:
            prefix = f"range_profile/rank{self.gpu_global_rank}/online/"
            loss.update({prefix+k: v for k, v in online_metrics.items()})
            self.last_perception_receipt["profile_online_breakdown"] = online_metrics
        return loss

    @torch.inference_mode(False)
    @torch.enable_grad()
    def _update_perception(self):
        profile = RangeAuxProfiler(self.profile_aux_breakdown, self.device)
        with profile.phase("pack_sequences"):
            batches = self.perception_storage.sequence_batches(self.perception_tbptt_steps, self.perception_sequence_batch_size)
        local_n = self.perception_storage.num_anchors
        count = torch.tensor(float(local_n), device=self.device, dtype=torch.float64)
        if self.is_multi_gpu:
            torch.distributed.all_reduce(count)
        global_n = float(count.item())
        params = list(self.perception.parameters())
        totals = torch.zeros(7, device=self.device, dtype=torch.float64)
        names = ("loss", "obs", "codec", "reg", "mem", "full", "range")
        steps = 0
        encode_host_s = 0.0
        encode_events = []
        cuda_timing = self.profile_timing and not self.profile_aux_breakdown and torch.device(self.device).type == "cuda"
        if global_n:
            for _ in range(self.perception_epochs):
                self.perception_optimizer.zero_grad(set_to_none=True)
                for batch in batches:
                    encode_start = time.perf_counter()
                    if cuda_timing:
                        event_start, event_stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        event_start.record()
                    with profile.phase("forward_loss"):
                        result = self.perception.auxiliary_loss(
                            batch.observed, batch.mapped, batch.simulated,
                            initial_hidden=batch.initial_hidden, reset=batch.reset, active=batch.active,
                            weights=self.perception_weights,
                            **({"profile": profile} if self.profile_aux_breakdown else {}),
                        )
                    # Per-anchor mean × number of anchors; SUM gradients across
                    # ranks / global anchors also handles zero-fresh local ranks.
                    with profile.phase("backward"):
                        (result["loss"] * (batch.anchor_count / global_n)).backward()
                    if cuda_timing:
                        event_stop.record()
                        encode_events.append((event_start, event_stop))
                    encode_host_s += time.perf_counter() - encode_start
                    for i, name in enumerate(names):
                        if name in result:
                            totals[i] += result[name].detach().to(torch.float64) * batch.anchor_count
                # Fixed full parameter layout on every rank, including unused
                # parameters and ranks with no local fresh captures.
                with profile.phase("gradient_sync_layout"):
                    flat = torch.cat([torch.zeros_like(p).flatten() if p.grad is None else p.grad.flatten() for p in params])
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(flat)
                    offset = 0
                    for p in params:
                        p.grad = flat[offset:offset + p.numel()].view_as(p).clone()
                        offset += p.numel()
                    if not torch.isfinite(flat).all():
                        raise FloatingPointError("Non-finite range-JEPA gradient; no auxiliary Adam step executed.")
                with profile.phase("clip_adam"):
                    nn.utils.clip_grad_norm_(params, self.perception_max_grad_norm)
                    self.perception_optimizer.step()
                steps += 1
            if self.is_multi_gpu:
                torch.distributed.all_reduce(totals)
            totals /= global_n * self.perception_epochs
        if cuda_timing:
            torch.cuda.synchronize(self.device)
        encode_s = sum(a.elapsed_time(b) / 1000.0 for a, b in encode_events) if cuda_timing else encode_host_s
        self.perception_update_count += steps
        self.last_perception_receipt = {
            "local_fresh_anchors": local_n, "global_fresh_anchors": int(global_n),
            "auxiliary_optimizer_steps": steps, "full_fresh_passes": steps,
            "anchors_per_local_pass": sum(b.anchor_count for b in batches),
            "tbptt_steps": self.perception_tbptt_steps, "sequence_batches": len(batches),
            "aux_encode_loss_s": encode_s, "aux_encode_loss_includes_backward": True,
            "aux_encode_loss_timing": "CUDA_events" if cuda_timing else "host_wall_unsynchronized",
            "execution_phase": "after_original_AMPPPO_update", "ppo_uses_cached_features_only": True,
        }
        self.perception_storage.clear()
        metrics = {**{f"range_jepa/{name}": value for name, value in zip(names, totals.tolist())},
                   "range_jepa/fresh_anchors": global_n, "range_jepa/optimizer_steps": float(steps)}
        if self.profile_aux_breakdown:
            # Every actual fresh capture is used once per full auxiliary pass;
            # count true rows from admitted Python metadata, not active.sum().item().
            profile.count("active_image_rows", local_n * 3 * steps)
            breakdown = profile.finish()
            metrics.update({f"range_profile/rank{self.gpu_global_rank}/{key}": value for key, value in breakdown.items()})
            self.last_perception_receipt["profile_aux_breakdown"] = breakdown
            self.last_perception_receipt["aux_encode_loss_s"] = breakdown["forward_loss_s"] + breakdown["backward_s"]
            self.last_perception_receipt["aux_encode_loss_timing"] = "CUDA_events" if profile.cuda else "host_wall"
        return metrics

    def train_mode(self):
        super().train_mode()
        self.perception.train()

    def eval_mode(self):
        super().eval_mode()
        self.perception.eval()

    def get_policy(self):
        return self._inference_policy

    @torch.no_grad()
    def broadcast_parameters(self):
        super().broadcast_parameters()
        if self.is_multi_gpu:
            for tensor in (*self.perception.parameters(), *self.perception.buffers()):
                # NCCL requires contiguous communication storage. Fixed angular
                # query buffers may be transposed views; keep their registered
                # object/storage/layout and copy the received scratch back.
                communication = tensor.detach()
                if not communication.is_contiguous():
                    communication = communication.contiguous()
                    torch.distributed.broadcast(communication, src=0)
                    tensor.copy_(communication)
                else:
                    torch.distributed.broadcast(communication, src=0)

    def save(self):
        return {**super().save(), "perception_state_dict": self.perception.state_dict(),
                "perception_optimizer_state_dict": self.perception_optimizer.state_dict(),
                "perception_cfg": deepcopy(self.perception_cfg),
                "perception_model_cfg": deepcopy(self.perception_model_cfg),
                "perception_update_count": self.perception_update_count}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        enabled = load_cfg is None or load_cfg.get("perception", load_cfg.get("actor", False))
        if enabled and strict and "perception_state_dict" not in loaded_dict:
            raise KeyError("A complete range-JEPA checkpoint requires perception_state_dict.")
        if enabled and strict:
            if "perception_model_cfg" not in loaded_dict:
                raise KeyError("Strict range-JEPA resume requires the effective perception_model_cfg contract.")
            if json.loads(json.dumps(loaded_dict["perception_model_cfg"])) != json.loads(json.dumps(self.perception_model_cfg)):
                raise ValueError("Perception model configuration differs from checkpoint (including FOV/normalization).")
        result = super().load(loaded_dict, load_cfg, strict)
        if enabled and "perception_state_dict" in loaded_dict:
            self.perception.load_state_dict(loaded_dict["perception_state_dict"], strict=strict)
            if load_cfg is None or load_cfg.get("optimizer", False):
                self.perception_optimizer.load_state_dict(loaded_dict["perception_optimizer_state_dict"])
            self.perception_update_count = loaded_dict.get("perception_update_count", 0)
        self._live_policy.clear_memory()
        self._inference_policy.clear_memory()
        self.perception_storage.clear()
        return result

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        spec = getattr(getattr(env, "unwrapped", env), "range_image_spec", None)
        if spec is not None:
            if not isinstance(spec, dict):
                raise TypeError("env.range_image_spec must be a plain metadata dict.")
            model_cfg = _model_configuration(cfg["algorithm"].get("perception_cfg"))
            for key in ("image_height", "image_width", "vertical_fov", "max_range", "input_normalized"):
                if key not in spec:
                    raise ValueError(f"env.range_image_spec is missing {key}.")
                if json.loads(json.dumps(spec[key])) != json.loads(json.dumps(model_cfg[key])):
                    raise ValueError(f"Environment range image {key} differs from perception configuration.")
        compact = compact_range_observations(obs, torch.zeros(env.num_envs, 512, device=device))
        return construct_multi_critic_algorithm(RangeJEPAAMPPPO, compact, env, cfg, device, include_amp_obs=True)
