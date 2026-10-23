"""Bounded per-environment scan context with immutable raw replay references.

A capture is admitted once. Selection is by capture-clock lag, not control ticks.
Raw chunks survive PPO rollout boundaries and resets until auxiliary replay ends;
identity selection prevents either episode or future-observation leakage.
"""

from __future__ import annotations

import torch


class RangeHistory:
    def __init__(self, batch_size, offsets, capture_dt, device):
        self.batch_size = batch_size
        self.offsets = tuple(offsets)
        if not self.offsets or min(self.offsets) < 1 or len(set(self.offsets)) != len(self.offsets):
            raise ValueError("History offsets must be distinct positive capture-clock lags.")
        self.capture_dt = float(capture_dt)
        self.capacity = max(self.offsets) + 1
        self.device = device
        self.ids = torch.full((batch_size, self.capacity), -1, dtype=torch.long, device=device)
        self.captures = torch.full_like(self.ids, -1)
        self.tokens = self.valid = None
        self.chunks = {}
        self.next_chunk = 0
        self.replay_cache = None
        self.replay_cache_bytes = 0
        self.replay_cache_budget = 0

    def reset(self, mask):
        self.ids[mask] = -1
        self.captures[mask] = -1

    def select(self, env_ids, capture_ids):
        requested = capture_ids[:, None] - torch.tensor(self.offsets, device=self.device)[None]
        slots = requested.remainder(self.capacity)
        rows = env_ids[:, None].expand_as(slots)
        found = (self.captures[rows, slots] == requested) & (self.ids[rows, slots] >= 0)
        ids = self.ids[rows, slots].masked_fill(~found, -1)
        age = (capture_ids[:, None] - self.captures[rows, slots]).to(torch.float32) * self.capture_dt
        age = age.masked_fill(~found, 0)
        if self.tokens is None:
            return ids, age, None
        tokens = self.tokens[rows, slots].to(torch.float32)
        valid = self.valid[rows, slots] & found[..., None]
        tokens.masked_fill_(~valid[..., None], 0)
        return ids, age, {"history_tokens": tokens, "history_valid": valid, "history_age": age}

    def append(self, env_ids, capture_ids, images, encoded):
        tokens, valid = encoded["tokens"].detach(), encoded["valid"].detach().bool()
        if self.tokens is None:
            self.tokens = torch.zeros(
                (self.batch_size, self.capacity, *tokens.shape[1:]), device=self.device, dtype=torch.float16
            )
            self.valid = torch.zeros(
                (self.batch_size, self.capacity, *valid.shape[1:]), device=self.device, dtype=torch.bool
            )
        # Global capture IDs may jump; a modulo ring plus exact ID checks makes
        # absent observations invalid instead of silently changing their age.
        slots = capture_ids.remainder(self.capacity)
        chunk = self.next_chunk
        self.next_chunk += 1
        self.chunks[chunk] = (env_ids.detach().clone(), images.detach().to(torch.float16).clone())
        self.ids[env_ids, slots] = chunk * self.batch_size + env_ids
        self.captures[env_ids, slots] = capture_ids
        self.tokens[env_ids, slots] = tokens.to(torch.float16)
        self.valid[env_ids, slots] = valid

    def prune(self):
        # Called only when no auxiliary anchors remain. One small ID transfer;
        # raw tensors are never copied back to the CPU.
        keep = set(torch.div(self.ids[self.ids >= 0], self.batch_size, rounding_mode="floor").unique().cpu().tolist())
        self.chunks = {k: v for k, v in self.chunks.items() if k in keep}

    def begin_replay_pass(self, cache_budget_bytes=256 * 1024**2):
        """A cache lives only while encoder parameters are unchanged in one pass."""
        self.replay_cache = {}
        self.replay_cache_bytes = 0
        self.replay_cache_budget = int(cache_budget_bytes)

    def end_replay_pass(self):
        self.replay_cache = None
        self.replay_cache_bytes = 0

    @torch.no_grad()
    def replay(self, ids, age, model, batch_size=256):
        """Encode each unique selected raw scan once using CURRENT parameters.

        History encoding is explicitly truncated (no history-CNN gradient); the
        current image and predictor retain gradients. No stale cached online
        token is used as a differentiable training context.
        """
        shape = ids.shape
        flat = ids.reshape(-1)
        unique, inverse = torch.unique(flat, return_inverse=True)
        prototype = self.tokens
        encoded = torch.zeros((unique.numel(), *prototype.shape[2:]), device=self.device, dtype=torch.float32)
        valid = torch.zeros((unique.numel(), *self.valid.shape[2:]), device=self.device, dtype=torch.bool)
        positive = unique >= 0
        chunk_ids = torch.div(unique[positive], self.batch_size, rounding_mode="floor").unique().cpu().tolist()
        for chunk in chunk_ids:
            where = (
                ((torch.div(unique, self.batch_size, rounding_mode="floor") == chunk) & positive).nonzero().flatten()
            )
            envs, raw = self.chunks[chunk]
            requested_envs = unique[where].remainder(self.batch_size)
            # Fresh env_ids come from nonzero(), hence are sorted.
            positions = torch.searchsorted(envs, requested_envs)
            cache = None if self.replay_cache is None else self.replay_cache.get(chunk)
            if cache is None and self.replay_cache is not None:
                required = raw.shape[0] * (encoded[0].numel() * encoded.element_size() + valid[0].numel() + 1)
                if self.replay_cache_bytes + required <= self.replay_cache_budget:
                    cache = (
                        encoded.new_zeros((raw.shape[0], *encoded.shape[1:])),
                        valid.new_zeros((raw.shape[0], *valid.shape[1:])),
                        torch.zeros(raw.shape[0], device=self.device, dtype=torch.bool),
                    )
                    self.replay_cache[chunk] = cache
                    self.replay_cache_bytes += required
            if cache is None:
                missing = torch.arange(where.numel(), device=self.device)
            else:
                missing = (~cache[2][positions]).nonzero().flatten()
            for start in range(0, missing.numel(), batch_size):
                request = missing[start : start + batch_size]
                selected = where[request]
                raw_rows = positions[request]
                images = raw[raw_rows].to(torch.float32)
                value = model.encode_history(images)
                encoded[selected] = value["tokens"].detach()
                valid[selected] = value["valid"].bool()
                if cache is not None:
                    cache[0][raw_rows] = encoded[selected]
                    cache[1][raw_rows] = valid[selected]
                    cache[2][raw_rows] = True
            if cache is not None:
                encoded[where] = cache[0][positions]
                valid[where] = cache[1][positions]
        return {
            "history_tokens": encoded[inverse].reshape(*shape, *encoded.shape[1:]),
            "history_valid": valid[inverse].reshape(*shape, *valid.shape[1:]),
            "history_age": age,
        }
