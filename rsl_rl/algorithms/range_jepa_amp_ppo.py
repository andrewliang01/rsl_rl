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
import math
import torch
from torch import nn
from tensordict import TensorDict

from .amp_ppo import AMPPPO
from .ppo_builders import construct_multi_critic_algorithm
from rsl_rl.models.range_jepa import RangeJEPA
from rsl_rl.storage.perception_storage import PerceptionStorage
from rsl_rl.storage.range_history import RangeHistory
from rsl_rl.utils.range_aux_profiler import RangeAuxProfiler
from rsl_rl.utils.range_online_profiler import RangeOnlineProfiler

def _model_configuration(perception_cfg):
    cfg = deepcopy(perception_cfg or {})
    for key in ("num_learning_epochs", "tbptt_steps", "sequence_batch_size", "learning_rate", "max_grad_norm", "weights", "profile_timing", "profile_aux_breakdown", "profile_online_breakdown", "cudnn_benchmark", "history_encode_batch_size", "depth_huber_beta_m"):
        cfg.pop(key, None)
    model_cfg = cfg.pop("model", {})
    if set(model_cfg) & set(cfg):
        raise ValueError("Duplicate perception model configuration keys.")
    merged = {**cfg, **model_cfg}
    variant = merged.get("variant", "A0")
    if variant == "A0":
        merged.pop("variant", None)
        cls = RangeJEPA
    elif variant == "H":
        from rsl_rl.models.range_evidence import RangeEvidenceJEPA
        cls = RangeEvidenceJEPA
    elif variant == "CAMP":
        from rsl_rl.models.range_evidence import RangeEvidenceJEPA
        terrain = {key: merged.pop(key, default) for key, default in (
            ("bev_grid_size", 32), ("bev_extent_m", 2.4),
            ("source_priors", (1., .85, .35)), ("confidence_tau_m", .05))}
        merged["variant"] = "H"
        bound = inspect.signature(RangeEvidenceJEPA).bind(**merged)
        bound.apply_defaults()
        return {**dict(bound.arguments), **terrain, "variant": "CAMP"}
    elif variant == "G":
        from rsl_rl.models.range_predictive import RangePredictiveJEPA
        cls = RangePredictiveJEPA
    else:
        from rsl_rl.models.range_experiments import RangeExperiment
        cls = RangeExperiment
    bound = inspect.signature(cls).bind(**merged)
    bound.apply_defaults()
    return dict(bound.arguments)


def _canonical_perception_contract(configuration):
    """Legacy image semantics and the disabled H writer have explicit compatibility defaults."""
    if not isinstance(configuration, dict):
        raise TypeError("Perception model configuration must be a dictionary.")
    canonical = deepcopy(configuration)
    encoding = canonical.setdefault("image_encoding", "depth_valid")
    if encoding not in ("depth_valid", "depth_elevation", "depth_center"):
        raise ValueError("Unsupported perception image_encoding.")
    if canonical.get("variant") in ("H", "CAMP"):
        objective = canonical.pop("auxiliary_objective", "hybrid")
        if objective not in ("hybrid", "direct_geometry"):
            raise ValueError("Unsupported auxiliary_objective.")
        if objective != "hybrid":
            canonical["auxiliary_objective"] = objective
        directional = canonical.setdefault("directional_memory_write", False)
        if type(directional) is not bool:
            raise TypeError("directional_memory_write must be bool.")
    elif "directional_memory_write" in canonical:
        raise ValueError("Directional memory writing is only supported by the H model.")
    return json.loads(json.dumps(canonical))


RAW_RANGE_KEYS = frozenset(("range_obs", "range_map", "range_sim", "range_capture_id", "range_episode_id", "camp_terrain_targets"))


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
        self.history = None
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
        self.history = None

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
            if self.history is not None:
                self.history.reset(mask)
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
            self.features = image.new_zeros(b, getattr(self.perception, "feature_size", 512))
            offsets = getattr(self.perception, "history_offsets", ())
            if offsets:
                self.history = RangeHistory(b, offsets, self.perception.capture_dt, image.device)
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
        if self.history is not None:
            self.history.reset(changed)
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
            history_kwargs = {}
            body_observations = None
            if getattr(self.perception, "requires_body_features", False):
                body_observations = self.actor.proprioceptive_observations(obs).index_select(0, indices)
                history_kwargs["body_features"] = self.actor.encode_proprioception(body_observations).detach()
                if record_indexed is not None:
                    saved["body_observations"] = body_observations
                elif record is not None:
                    raise ValueError("F2 training requires indexed paired capture storage.")
            if self.history is not None:
                history_ids, history_age, context = self.history.select(indices, capture[indices])
                history_kwargs.update(context or {})
                if record_indexed is not None:
                    saved["history_ids"] = history_ids
                    saved["history_age"] = history_age
                elif record is not None:
                    raise ValueError("Explicit history replay requires indexed capture recording.")
            if profile is None:
                prediction = self.perception.step(fresh_image, pre_hidden, **history_kwargs)
            else:
                profile.mark("online_cnn")
                encoded = self.perception.encode(fresh_image)
                if body_observations is not None:
                    encoded = self.perception.condition_context(encoded, history_kwargs.pop("body_features"))
                profile.mark("memory_attention")
                prediction = self.perception._step_encoded(encoded, pre_hidden, **history_kwargs)
                if self.history is not None:
                    prediction["history_encoded"] = self.perception.history_from_encoded(encoded)
            if profile is not None:
                profile.mark("publish")
            feature_size = getattr(self.perception, "feature_size", 512)
            if prediction["features"].shape != (indices.numel(), feature_size):
                raise ValueError(f"Range perception must produce [fresh,{feature_size}] cached features.")
            if self.history is not None:
                self.history.append(indices, capture[indices], fresh_image, prediction["history_encoded"],
                                    body_observations=body_observations)
                if record_indexed is None and self.history.next_chunk % self.history.capacity == 0:
                    self.history.prune()
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
        # ManagerBasedEnv.seed() can override train.py's initial cuDNN flag.
        # Apply an explicit task-local request after env construction; None
        # preserves existing behavior for every other configuration.
        self.cudnn_benchmark_requested = cfg.pop("cudnn_benchmark", None)
        if self.cudnn_benchmark_requested is not None and type(self.cudnn_benchmark_requested) is not bool:
            raise ValueError("perception_cfg.cudnn_benchmark must be bool or None.")
        if self.cudnn_benchmark_requested is not None:
            previous = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = self.cudnn_benchmark_requested
            print(f"RangeJEPA cuDNN benchmark: requested={self.cudnn_benchmark_requested}, "
                  f"previous={previous}, actual={torch.backends.cudnn.benchmark}")
        self.perception_epochs = cfg.pop("num_learning_epochs", 5)
        self.perception_tbptt_steps = cfg.pop("tbptt_steps", 8)
        self.perception_sequence_batch_size = cfg.pop("sequence_batch_size", 128)
        for name in ("perception_epochs", "perception_tbptt_steps", "perception_sequence_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.requested_perception_learning_rate = cfg.pop("learning_rate", None)
        lr = 1e-3 if self.requested_perception_learning_rate is None else self.requested_perception_learning_rate
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError("Perception learning rate must be finite and positive, or None to inherit on resume.")
        self.perception_max_grad_norm = cfg.pop("max_grad_norm", 1.0)
        if self.perception_max_grad_norm <= 0:
            raise ValueError("Perception learning rate and max_grad_norm must be positive.")
        self.profile_timing = bool(cfg.pop("profile_timing", False))
        self.profile_aux_breakdown = bool(cfg.pop("profile_aux_breakdown", False))
        self.profile_online_breakdown = bool(cfg.pop("profile_online_breakdown", False))
        self._online_profile = RangeOnlineProfiler(self.profile_online_breakdown, self.device)
        self.perception_weights = cfg.pop("weights", None)
        self.requested_depth_huber_beta_m = cfg.pop("depth_huber_beta_m", None)
        self.depth_loss_config_source = "legacy_default" if self.requested_depth_huber_beta_m is None else "explicit_launch"
        self.history_encode_batch_size = cfg.pop("history_encode_batch_size", 256)
        if type(self.history_encode_batch_size) is not int or self.history_encode_batch_size < 1:
            raise ValueError("history_encode_batch_size must be a positive integer.")
        model_cfg = cfg.pop("model", {})
        if set(model_cfg) & set(cfg):
            raise ValueError("Duplicate perception model configuration keys.")
        self.perception_model_cfg = _model_configuration(perception_cfg)
        if self.perception_model_cfg.get("variant") == "F2":
            from rsl_rl.models.range_body import RangeBodyConditionedBEV
            self.perception = RangeBodyConditionedBEV(**self.perception_model_cfg).to(self.device)
            if not all(hasattr(self.actor, name) for name in ("proprioceptive_observations", "encode_proprioception")):
                raise ValueError("F2 requires the PPO-owned RangeBEVActor proprioceptive encoder.")
        elif self.perception_model_cfg.get("variant") == "CAMP":
            from rsl_rl.models.camp import CampTerrainMemory
            self.perception = CampTerrainMemory(**self.perception_model_cfg).to(self.device)
        elif self.perception_model_cfg.get("variant") == "H":
            from rsl_rl.models.range_evidence import RangeEvidenceJEPA
            self.perception = RangeEvidenceJEPA(**self.perception_model_cfg).to(self.device)
        elif self.perception_model_cfg.get("variant") == "G":
            from rsl_rl.models.range_predictive import RangePredictiveJEPA
            self.perception = RangePredictiveJEPA(**self.perception_model_cfg).to(self.device)
        elif "variant" in self.perception_model_cfg:
            from rsl_rl.models.range_experiments import RangeExperiment
            self.perception = RangeExperiment(**self.perception_model_cfg).to(self.device)
        else:
            self.perception = RangeJEPA(**self.perception_model_cfg).to(self.device)
        self._configure_depth_loss(self.requested_depth_huber_beta_m)
        self.perception_optimizer = torch.optim.Adam(self.perception.parameters(), lr=lr)
        ppo_params = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        if ppo_params & {id(p) for p in self.perception.parameters()}:
            raise RuntimeError("Perception parameters must be disjoint from original PPO/AMP optimizers.")
        self.perception_storage = PerceptionStorage()
        self._live_policy = RangeJEPAInferencePolicy(self.actor, self.perception)
        self._inference_policy = RangeJEPAInferencePolicy(self.actor, self.perception)
        self.perception_update_count = 0
        self.last_perception_receipt: dict = {"image_encoding": getattr(self.perception, "image_encoding", "depth_valid"),
                                             "cudnn_benchmark_requested": self.cudnn_benchmark_requested,
                                             "cudnn_benchmark_actual": torch.backends.cudnn.benchmark,
                                             **self._depth_loss_receipt()}

    def _configure_depth_loss(self, beta_m):
        if type(self.perception) is not RangeJEPA:
            if beta_m is not None:
                raise ValueError("depth_huber_beta_m currently supports only the A0 RangeJEPA criterion.")
            return
        self.perception.set_depth_huber_beta_m(beta_m)
        self.perception_cfg["depth_huber_beta_m"] = self.perception.depth_huber_beta_m

    def _depth_loss_receipt(self):
        if type(self.perception) is not RangeJEPA:
            return {}
        return {"depth_huber_beta_m": self.perception.depth_huber_beta_m,
                "depth_huber_beta_normalized": self.perception.depth_huber_beta_normalized,
                "depth_loss_config_source": self.depth_loss_config_source}

    def _record_perception_capture(self, *args):
        return self.perception_storage.add_indexed(*args)

    def _perception_auxiliary_kwargs(self, batch):
        return {}

    def act(self, obs: TensorDict) -> torch.Tensor:
        if "range_obs" not in obs:
            raise KeyError("Training act requires raw range_obs; only PPO minibatches consume cached features.")
        if not self.profile_online_breakdown:
            return super().act(self._live_policy.prepare(obs, record_indexed=self._record_perception_capture))
        compact = self._live_policy.prepare(obs, record_indexed=self._record_perception_capture, profile=self._online_profile)
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
        self.last_perception_receipt.update(cudnn_benchmark_requested=self.cudnn_benchmark_requested,
                                            cudnn_benchmark_actual=torch.backends.cudnn.benchmark,
                                            parent_AMPPPO_s=parent_s, aux_total_s=aux_s,
                                            timing_gpu_synchronized=(self.profile_timing or self.profile_aux_breakdown))
        if self.profile_aux_breakdown:
            prefix = f"range_profile/rank{self.gpu_global_rank}/"
            # Include profiler export/receipt overhead in the same measured
            # outer wall total, retaining exact additive top-level accounting.
            loss[prefix + "aux_wall_s"] = aux_s
            loss[prefix + "other_s"] = aux_s - sum(loss[prefix + name + "_s"] for name in RangeAuxProfiler.TOP_PHASES)
            self.last_perception_receipt["profile_aux_breakdown"] = {k[len(prefix):]: v for k, v in loss.items() if k.startswith(prefix)}
        loss.update({"range_jepa/parent_AMPPPO_s": parent_s, "range_jepa/aux_total_s": aux_s,
                     "range_jepa/cudnn_benchmark_actual": float(torch.backends.cudnn.benchmark)})
        if online_metrics:
            prefix = f"range_profile/rank{self.gpu_global_rank}/online/"
            loss.update({prefix+k: v for k, v in online_metrics.items()})
            self.last_perception_receipt["profile_online_breakdown"] = online_metrics
        return loss

    def _zero_perception_grad(self):
        self.perception_optimizer.zero_grad(set_to_none=True)

    def _step_perception_optimizer(self):
        self.perception_optimizer.step()

    @torch.inference_mode(False)
    @torch.enable_grad()
    def _update_perception(self):
        profile = RangeAuxProfiler(self.profile_aux_breakdown, self.device)
        with profile.phase("pack_sequences"):
            batches = self.perception_storage.sequence_plan(self.perception_tbptt_steps, self.perception_sequence_batch_size)
        local_n = self.perception_storage.num_anchors
        count = torch.tensor(float(local_n), device=self.device, dtype=torch.float64)
        if self.is_multi_gpu:
            torch.distributed.all_reduce(count)
        global_n = float(count.item())
        params = list(self.perception.parameters())
        names = ("loss", "obs", "codec", "reg", "mem", "full", "range") + tuple(getattr(self.perception, "metric_names", ())) + tuple(getattr(self, "perception_extra_metric_names", ()))
        # Model-declared names keep collective layout identical on empty ranks.
        # Geometric metrics are diagnostics only, never extra loss weights.
        totals = torch.zeros(len(names), device=self.device, dtype=torch.float64)
        steps = 0
        encode_host_s = 0.0
        encode_events = []
        cuda_timing = self.profile_timing and not self.profile_aux_breakdown and torch.device(self.device).type == "cuda"
        if global_n:
            for _ in range(self.perception_epochs):
                if self._live_policy.history is not None:
                    self._live_policy.history.begin_replay_pass()
                self._zero_perception_grad()
                for batch in batches:
                    encode_start = time.perf_counter()
                    if cuda_timing:
                        event_start, event_stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        event_start.record()
                    with profile.phase("forward_loss"):
                        history_kwargs = {}
                        if getattr(self.perception, "uses_temporal_prediction", False):
                            pair = torch.zeros_like(batch.active)
                            pair[1:] = (batch.active[1:] & batch.active[:-1] & ~batch.reset[1:]
                                        & (batch.capture_ids[1:] == batch.capture_ids[:-1] + 1)
                                        & (batch.episode_ids[1:] == batch.episode_ids[:-1])
                                        & (batch.env_ids[1:] == batch.env_ids[:-1]))
                            history_kwargs["temporal_valid"] = pair
                        if batch.history_ids is not None:
                            history_kwargs = self._live_policy.history.replay(
                                batch.history_ids, batch.history_age, self.perception, self.history_encode_batch_size,
                                body_encoder=(self.actor.encode_proprioception
                                              if getattr(self.perception, "requires_body_features", False) else None))
                        if getattr(self.perception, "requires_body_features", False):
                            if batch.body_observations is None:
                                raise ValueError("F2 auxiliary captures lack paired body observations.")
                            with torch.no_grad():
                                history_kwargs["body_features"] = self.actor.encode_proprioception(
                                    batch.body_observations).detach()
                        result = self.perception.auxiliary_loss(
                            batch.observed, batch.mapped, batch.simulated,
                            initial_hidden=batch.initial_hidden, reset=batch.reset, active=batch.active,
                            weights=self.perception_weights,
                            **history_kwargs,
                            **self._perception_auxiliary_kwargs(batch),
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
                if self._live_policy.history is not None:
                    # Cached history is valid only while this pass uses unchanged weights.
                    self._live_policy.history.end_replay_pass()
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
                    self._step_perception_optimizer()
                steps += 1
            if self.is_multi_gpu:
                torch.distributed.all_reduce(totals)
            totals /= global_n * self.perception_epochs
        if cuda_timing:
            torch.cuda.synchronize(self.device)
        encode_s = sum(a.elapsed_time(b) / 1000.0 for a, b in encode_events) if cuda_timing else encode_host_s
        self.perception_update_count += steps
        self.last_perception_receipt = {
            "image_encoding": getattr(self.perception, "image_encoding", "depth_valid"),
            "local_fresh_anchors": local_n, "global_fresh_anchors": int(global_n),
            "auxiliary_optimizer_steps": steps, "full_fresh_passes": steps,
            "anchors_per_local_pass": batches.anchor_count,
            "tbptt_steps": self.perception_tbptt_steps, "sequence_batches": len(batches),
            "aux_encode_loss_s": encode_s, "aux_encode_loss_includes_backward": True,
            "aux_encode_loss_timing": "CUDA_events" if cuda_timing else "host_wall_unsynchronized",
            "execution_phase": "after_original_AMPPPO_update", "ppo_uses_cached_features_only": True,
            **self._depth_loss_receipt(),
        }
        self.perception_storage.clear()
        if self._live_policy.history is not None:
            self._live_policy.history.prune()
            self.last_perception_receipt["history_training"] = "current-encoder raw replay, truncated past CNN gradients"
            self.last_perception_receipt["history_offsets"] = list(self._live_policy.history.offsets)
        metrics = {**{f"range_jepa/{name}": value for name, value in zip(names, totals.tolist())},
                   "range_jepa/fresh_anchors": global_n, "range_jepa/optimizer_steps": float(steps)}
        if self.profile_aux_breakdown:
            # Every actual fresh capture is used once per full auxiliary pass;
            # count true rows from admitted Python metadata, not active.sum().item().
            encoded_views = 1 if getattr(self.perception, "auxiliary_objective", None) == "direct_geometry" else 3
            profile.count("active_image_rows", local_n * encoded_views * steps)
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
        saved = {**super().save(), "perception_state_dict": self.perception.state_dict(),
                "perception_optimizer_state_dict": self.perception_optimizer.state_dict(),
                "perception_cfg": deepcopy(self.perception_cfg),
                "perception_model_cfg": deepcopy(self.perception_model_cfg),
                "perception_update_count": self.perception_update_count}
        if type(self.perception) is RangeJEPA:
            saved["perception_loss_cfg"] = {"version": 1, "depth_huber_beta_m": self.perception.depth_huber_beta_m}
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        enabled = load_cfg is None or load_cfg.get("perception", load_cfg.get("actor", False))
        if enabled and strict and "perception_state_dict" not in loaded_dict:
            raise KeyError("A complete range-JEPA checkpoint requires perception_state_dict.")
        if enabled:
            if strict and "perception_model_cfg" not in loaded_dict:
                raise KeyError("Strict range-JEPA resume requires the effective perception_model_cfg contract.")
            saved_contract = _canonical_perception_contract(loaded_dict.get("perception_model_cfg", {}))
            current_contract = _canonical_perception_contract(self.perception_model_cfg)
            # Tensor shapes are equal across encodings; strict=False must not
            # silently reinterpret a checkpoint's second channel.
            if saved_contract["image_encoding"] != current_contract["image_encoding"]:
                raise ValueError("Perception image_encoding differs from checkpoint; explicit cross-encoding transfer is not supported.")
            if saved_contract.get("directional_memory_write", False) != current_contract.get("directional_memory_write", False):
                raise ValueError("Perception directional_memory_write differs from checkpoint; explicit writer transfer is not supported.")
            if strict and saved_contract != current_contract:
                raise ValueError("Perception model configuration differs from checkpoint (including FOV/normalization).")
        if enabled and type(self.perception) is RangeJEPA:
            loss_cfg = loaded_dict.get("perception_loss_cfg")
            if loss_cfg is not None:
                if not isinstance(loss_cfg, dict) or type(loss_cfg.get("version")) is not int or loss_cfg["version"] != 1 or "depth_huber_beta_m" not in loss_cfg:
                    raise ValueError("Unsupported or incomplete perception_loss_cfg.")
                saved_beta_m = loss_cfg["depth_huber_beta_m"]
            else:
                saved_beta_m = loaded_dict.get("perception_cfg", {}).get("depth_huber_beta_m")
            # None means inherit the saved training criterion, or legacy for an
            # old checkpoint. A positive launch value is an explicit experiment
            # override. Strict model/FOV checking above is never relaxed.
            beta_m = saved_beta_m if self.requested_depth_huber_beta_m is None else self.requested_depth_huber_beta_m
            self._configure_depth_loss(beta_m)
            self.depth_loss_config_source = "checkpoint_inherited" if self.requested_depth_huber_beta_m is None else "explicit_launch_override"
            self.last_perception_receipt.update(self._depth_loss_receipt())
        result = super().load(loaded_dict, load_cfg, strict)
        if enabled and "perception_state_dict" in loaded_dict:
            self.perception.load_state_dict(loaded_dict["perception_state_dict"], strict=strict)
            if load_cfg is None or load_cfg.get("optimizer", False):
                self.perception_optimizer.load_state_dict(loaded_dict["perception_optimizer_state_dict"])
            self.perception_update_count = loaded_dict.get("perception_update_count", 0)
        if enabled and self.requested_perception_learning_rate is not None:
            # Loading Adam restores its saved LR. Honor an explicit experiment
            # override afterwards, while retaining moments and step counters.
            for group in self.perception_optimizer.param_groups:
                group["lr"] = float(self.requested_perception_learning_rate)
            self.last_perception_receipt["perception_optimizer_learning_rate"] = float(self.requested_perception_learning_rate)
            self.last_perception_receipt["perception_learning_rate_source"] = "explicit_launch_override_after_optimizer_load"
            print(f"RangeJEPA perception Adam LR override after resume: {self.requested_perception_learning_rate:g}")
        self._live_policy.clear_memory()
        self._inference_policy.clear_memory()
        self.perception_storage.clear()
        return result

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        spec = getattr(getattr(env, "unwrapped", env), "range_image_spec", None)
        if spec is not None:
            if not isinstance(spec, dict):
                raise TypeError("env.range_image_spec must be a plain metadata dict.")
            model_cfg = _model_configuration(cfg["algorithm"].get("perception_cfg"))
            environment_encoding = spec.get("image_encoding", "depth_valid")
            if environment_encoding != _canonical_perception_contract(model_cfg)["image_encoding"]:
                raise ValueError("Environment range image image_encoding differs from perception configuration.")
            for key in ("image_height", "image_width", "vertical_fov", "max_range", "input_normalized"):
                if key not in spec:
                    raise ValueError(f"env.range_image_spec is missing {key}.")
                if json.loads(json.dumps(spec[key])) != json.loads(json.dumps(model_cfg[key])):
                    raise ValueError(f"Environment range image {key} differs from perception configuration.")
        effective = _model_configuration(cfg["algorithm"].get("perception_cfg"))
        variant = effective.get("variant", "A0").upper()
        feature_size = 512
        if variant == "E":
            feature_size = len(effective["bev_extents"]) * 2 * effective["bev_grid_size"] ** 2
        elif variant in {"F", "F2"}:
            feature_size = len(effective["bev_extents"]) * effective["channels"] * 8 ** 2
        elif variant in {"H", "CAMP"}:
            import math
            feature_size = effective["hidden_size"] + math.prod(effective["context_grid"]) * (effective["channels"] + 1)
            cfg = deepcopy(cfg)
            cfg["actor"]["reader_cfg"] = {k: effective[k] for k in (
                "image_height", "image_width", "channels", "hidden_size", "bank_size",
                "context_grid", "vertical_fov", "attention_heads")}
            if variant == "CAMP":
                cfg["actor"]["terrain_cfg"] = {k: effective[k] for k in (
                    "bev_grid_size", "bev_extent_m", "source_priors")}
        compact = compact_range_observations(obs, torch.zeros(env.num_envs, feature_size, device=device))
        return construct_multi_critic_algorithm(cls, compact, env, cfg, device, include_amp_obs=True)
