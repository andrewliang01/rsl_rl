# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DeltaNet recurrent building blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization using exporter-friendly primitives."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("RMSNorm dimension must be positive.")
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1.0e-6) * self.weight


def delta_rule_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    chunk_size: int = 32,
    reset_before: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the inclusive DeltaNet recurrence in chunks.

    Inputs ``q``, ``k`` and ``v`` use ``[B, H, T, D]`` layout, ``beta`` uses
    ``[B, H, T]``, and the recurrent matrix uses ``[B, H, V, K]``.
    """

    if chunk_size <= 0:
        raise ValueError("DeltaNet chunk_size must be positive.")
    outputs = []
    for start in range(0, q.shape[2], chunk_size):
        stop = start + chunk_size
        q_chunk = q[:, :, start:stop]
        k_chunk = k[:, :, start:stop]
        v_chunk = v[:, :, start:stop]
        beta_chunk = beta[:, :, start:stop].unsqueeze(-1)
        size = q_chunk.shape[2]
        same_episode = 1.0
        initial_active = 1.0
        final_active = 1.0
        if reset_before is not None:
            # [B,T]: reset happens BEFORE consuming the current observation.
            episode = reset_before[:, start:stop].long().cumsum(-1)
            same_episode = (episode[:, :, None] == episode[:, None, :])[:, None]
            initial_active = (episode == 0)[:, None, :, None]
            final_active = (episode == episode[:, -1:])[:, None, :, None]
        system = torch.eye(size, device=q.device, dtype=q.dtype)
        system = system + torch.tril(
            beta_chunk * (k_chunk @ k_chunk.transpose(-1, -2)) * same_episode,
            diagonal=-1,
        )
        errors = torch.linalg.solve_triangular(
            system,
            beta_chunk * (v_chunk - (k_chunk @ state.transpose(-1, -2)) * initial_active),
            upper=False,
            unitriangular=True,
        )
        outputs.append(
            (q_chunk @ state.transpose(-1, -2)) * initial_active
            + (torch.tril(q_chunk @ k_chunk.transpose(-1, -2)) * same_episode) @ errors
        )
        if reset_before is not None:
            state = state * (episode[:, -1] == 0)[:, None, None, None]
        state = state + (errors * final_active).transpose(-1, -2) @ k_chunk
    return torch.cat(outputs, dim=2), state


class DeltaNetBlock(nn.Module):
    """Pre-normalized DeltaNet mixer followed by a SwiGLU feed-forward block."""

    def __init__(
        self,
        dim: int = 256,
        heads: int = 4,
        head_dim: int = 32,
        ffn_dim: int = 512,
        conv_size: int = 4,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if min(dim, heads, head_dim, ffn_dim, conv_size, chunk_size) <= 0:
            raise ValueError("DeltaNet dimensions must be positive.")
        self.heads = heads
        self.head_dim = head_dim
        self.conv_size = conv_size
        self.chunk_size = chunk_size
        self.inner_dim = heads * head_dim
        self.matrix_size = heads * head_dim * head_dim
        self.state_size = self.matrix_size + 3 * self.inner_dim * (conv_size - 1)
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * self.inner_dim, bias=False)
        self.conv = nn.Conv1d(
            3 * self.inner_dim,
            3 * self.inner_dim,
            conv_size,
            groups=3 * self.inner_dim,
            bias=False,
        )
        self.beta = nn.Linear(dim, heads, bias=False)
        self.out_norm = RMSNorm(head_dim)
        self.out = nn.Linear(self.inner_dim, dim, bias=False)
        self.ffn_norm = RMSNorm(dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_up = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        packed: torch.Tensor,
        reset_before: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length, batch = x.shape[:2]
        state = packed[:, : self.matrix_size].reshape(
            batch,
            self.heads,
            self.head_dim,
            self.head_dim,
        )
        cache = packed[:, self.matrix_size :].reshape(
            batch,
            3 * self.inner_dim,
            self.conv_size - 1,
        )
        normalized = self.norm(x)
        projected = self.qkv(normalized).permute(1, 2, 0)
        history = torch.cat((cache, projected), dim=-1)
        if reset_before is None:
            filtered = self.conv(history).transpose(1, 2)
        else:
            if reset_before.shape != (length, batch):
                raise ValueError("DeltaNet reset_before must have shape [T,B].")
            episode = reset_before.transpose(0, 1).long().cumsum(-1)
            history_episode = F.pad(episode, (self.conv_size - 1, 0))
            # Parallel across time and batch; only loop over the short conv taps.
            filtered = torch.zeros_like(projected)
            for tap in range(self.conv_size):
                valid = history_episode[:, tap : tap + length] == episode
                filtered = filtered + (
                    history[:, :, tap : tap + length] * self.conv.weight[:, 0, tap][None, :, None] * valid[:, None]
                )
            filtered = filtered.transpose(1, 2)
        filtered = filtered.reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = filtered.unbind(dim=2)
        q = F.normalize(F.silu(q), dim=-1).transpose(1, 2)
        k = F.normalize(F.silu(k), dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        beta = self.beta(normalized).sigmoid().permute(1, 2, 0)

        if length == 1:
            if reset_before is not None:
                state = state * (~reset_before[0].bool())[:, None, None, None]
            key = k[:, :, 0]
            error = v[:, :, 0] - (state @ key.unsqueeze(-1)).squeeze(-1)
            state = state + beta[:, :, 0, None, None] * error.unsqueeze(-1) * key.unsqueeze(-2)
            result = (state @ q[:, :, 0].unsqueeze(-1)).squeeze(-1).unsqueeze(2)
        else:
            result, state = delta_rule_chunkwise(
                q,
                k,
                v,
                beta,
                state,
                self.chunk_size,
                None if reset_before is None else reset_before.transpose(0, 1),
            )

        result = self.out_norm(result).permute(2, 0, 1, 3)
        result = result.reshape(length, batch, self.inner_dim)
        x = x + self.out(result)
        normalized = self.ffn_norm(x)
        x = x + self.ffn_down(F.silu(self.ffn_gate(normalized)) * self.ffn_up(normalized))
        if self.conv_size > 1:
            new_cache = history[:, :, -(self.conv_size - 1) :]
            if reset_before is not None:
                valid = history_episode[:, -(self.conv_size - 1) :] == episode[:, -1:]
                new_cache = new_cache * valid[:, None]
        else:
            new_cache = history[:, :, :0]
        return x, torch.cat((state.flatten(1), new_cache.flatten(1)), dim=-1)


__all__ = ["DeltaNetBlock", "RMSNorm", "delta_rule_chunkwise"]
