"""Fresh-capture, episode-aware TBPTT storage, independent of PPO storage."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from bisect import bisect_right


@dataclass(frozen=True)
class PerceptionSequenceBatch:
    observed: torch.Tensor
    mapped: torch.Tensor
    simulated: torch.Tensor
    initial_hidden: torch.Tensor
    active: torch.Tensor
    reset: torch.Tensor
    env_ids: torch.Tensor
    episode_ids: torch.Tensor
    capture_ids: torch.Tensor
    anchor_count: int
    history_ids: torch.Tensor | None = None
    history_age: torch.Tensor | None = None


class PerceptionStorage:
    """Own only captures actually consumed by the online perception network.

    Held control frames never enter this buffer. Each TBPTT segment starts from
    the corresponding detached online PRE hidden state. Segments never cross an
    environment/episode boundary, and padding has active=False. No sampling or
    dropping is performed; every stored anchor occurs exactly once per epoch.
    """
    def __init__(self) -> None:
        self._batches: list[dict[str, torch.Tensor]] = []
        self.num_anchors = 0

    @torch.inference_mode(False)
    @torch.no_grad()
    def add(self, observed, mapped, simulated, pre_hidden, env_ids, episode_ids, capture_ids) -> None:
        n = env_ids.numel()
        if not n:
            return
        if observed.ndim != 4 or observed.shape[1] != 2 or mapped.shape != observed.shape or simulated.shape != observed.shape:
            raise ValueError("Perception targets must share [B,2,H,W] depth/mask shape.")
        if pre_hidden.ndim != 2 or pre_hidden.shape[0] != n or observed.shape[0] != n:
            raise ValueError("Fresh capture and PRE hidden batch sizes disagree.")
        for name, ids in (("env", env_ids), ("episode", episode_ids), ("capture", capture_ids)):
            if ids.dtype != torch.long or ids.numel() != n:
                raise ValueError(f"{name} IDs must be nonnegative int64 per capture.")
        if torch.any(torch.stack((env_ids.reshape(-1), episode_ids.reshape(-1), capture_ids.reshape(-1))) < 0):
            raise ValueError("Source env/episode/capture IDs must be nonnegative.")
        self._batches.append({name: tensor.detach().clone() for name, tensor in {
            "observed": observed, "mapped": mapped, "simulated": simulated,
            "pre_hidden": pre_hidden, "env_ids": env_ids.reshape(-1),
            "episode_ids": episode_ids.reshape(-1), "capture_ids": capture_ids.reshape(-1),
        }.items()})
        self.num_anchors += n

    @torch.inference_mode(False)
    @torch.no_grad()
    def add_indexed(self, observed, mapped, simulated, pre_hidden, env_ids, episode_ids, capture_ids):
        """Select fresh rows once, own their snapshots, and return them for online reuse.

        Images, PRE hidden and episode/capture IDs are full environment batches;
        env_ids selects the fresh rows. Unlike add(), index_select already creates
        independent storage, so an extra clone would only copy the same images.
        The returned dictionary is read-only to its caller: the storage owns it.
        """
        if env_ids.dtype != torch.long or env_ids.ndim != 1:
            raise ValueError("Fresh env IDs must be a one-dimensional int64 index.")
        n = env_ids.numel()
        if not n:
            return None
        if observed.ndim != 4 or observed.shape[1] != 2 or mapped.shape != observed.shape or simulated.shape != observed.shape:
            raise ValueError("Perception targets must share [B,2,H,W] depth/mask shape.")
        b = observed.shape[0]
        if pre_hidden.ndim != 2 or pre_hidden.shape[0] != b:
            raise ValueError("Fresh capture and PRE hidden batch sizes disagree.")
        for name, ids in (("episode", episode_ids), ("capture", capture_ids)):
            if ids.dtype != torch.long or ids.numel() != b:
                raise ValueError(f"{name} IDs must be int64 per source environment.")
        owned = {name: tensor.detach().index_select(0, env_ids) for name, tensor in {
            "observed": observed, "mapped": mapped, "simulated": simulated,
            "pre_hidden": pre_hidden, "episode_ids": episode_ids.reshape(-1),
            "capture_ids": capture_ids.reshape(-1),
        }.items()}
        owned["env_ids"] = env_ids.detach().clone()
        if torch.any(torch.stack([owned[k] for k in ("env_ids", "episode_ids", "capture_ids")]) < 0):
            raise ValueError("Source env/episode/capture IDs must be nonnegative.")
        self._batches.append(owned)
        self.num_anchors += n
        return owned

    def clear(self) -> None:
        self._batches.clear()
        self.num_anchors = 0

    def sequence_plan(self, tbptt_steps: int, sequence_batch_size: int):
        """Replayable plan: materialize one batch, not every image sequence.

        Tensor snapshots remain owned until the pass ends. Only small identity
        metadata is concatenated; image gathers read directly from snapshots.
        """
        return PerceptionSequencePlan(self._batches, tbptt_steps, sequence_batch_size)

    @torch.inference_mode(False)
    @torch.no_grad()
    def sequence_batches(self, tbptt_steps: int, sequence_batch_size: int) -> list[PerceptionSequenceBatch]:
        if tbptt_steps < 1 or sequence_batch_size < 1:
            raise ValueError("TBPTT length and sequence batch size must be positive.")
        if not self.num_anchors:
            return []
        data = {key: torch.cat([b[key] for b in self._batches]) for key in self._batches[0]}
        # One compact metadata transfer; no per-image CPU copies or per-env GPU kernels.
        rows = torch.stack([data[k] for k in ("env_ids", "episode_ids", "capture_ids")], 1).cpu().tolist()
        groups: dict[tuple[int, int], list[int]] = {}
        for i, (env, episode, capture) in enumerate(rows):
            group = groups.setdefault((env, episode), [])
            if group and capture <= rows[group[-1]][2]:
                raise ValueError("Capture IDs must increase strictly within an environment episode.")
            group.append(i)
        sequences = [group[i:i + tbptt_steps] for group in groups.values() for i in range(0, len(group), tbptt_steps)]
        result = []
        for start in range(0, len(sequences), sequence_batch_size):
            seqs = sequences[start:start + sequence_batch_size]
            length = max(map(len, seqs))
            index = torch.tensor([s + [s[-1]] * (length - len(s)) for s in seqs], device=data["observed"].device).T
            active = torch.arange(length, device=index.device)[:, None] < torch.tensor([len(s) for s in seqs], device=index.device)[None]
            result.append(PerceptionSequenceBatch(
                **{key: data[key][index] for key in ("observed", "mapped", "simulated", "env_ids", "episode_ids", "capture_ids")},
                initial_hidden=data["pre_hidden"][index[0]], active=active,
                # PRE hidden already equals zero at reset. Never reset at arbitrary TBPTT boundaries.
                reset=torch.zeros_like(active), anchor_count=sum(map(len, seqs)),
                **{key: data[key][index] for key in ("history_ids", "history_age") if key in data},
            ))
        if sum(batch.anchor_count for batch in result) != self.num_anchors:
            raise RuntimeError("TBPTT packing lost fresh captures.")
        return result


class PerceptionSequencePlan:
    """Same ordering/padding as sequence_batches, with bounded image scratch."""

    def __init__(self, snapshots, tbptt_steps, sequence_batch_size):
        if tbptt_steps < 1 or sequence_batch_size < 1:
            raise ValueError("TBPTT length and sequence batch size must be positive.")
        self.snapshots = tuple(snapshots)
        self.groups = []
        self.offsets = [0]
        self.anchor_count = 0
        if not snapshots:
            return
        for snapshot in snapshots:
            self.offsets.append(self.offsets[-1] + snapshot["env_ids"].numel())
        metadata = torch.cat([torch.stack([s[k] for k in ("env_ids", "episode_ids", "capture_ids")], 1)
                              for s in snapshots]).cpu().tolist()
        groups = {}
        for index, (env, episode, capture) in enumerate(metadata):
            group = groups.setdefault((env, episode), [])
            if group and capture <= metadata[group[-1]][2]:
                raise ValueError("Capture IDs must increase strictly within an environment episode.")
            group.append(index)
        sequences = [g[i:i + tbptt_steps] for g in groups.values() for i in range(0, len(g), tbptt_steps)]
        for start in range(0, len(sequences), sequence_batch_size):
            self.groups.append(sequences[start:start + sequence_batch_size])
        self.anchor_count = sum(len(s) for group in self.groups for s in group)
        if self.anchor_count != self.offsets[-1]:
            raise RuntimeError("TBPTT planning lost fresh captures.")

    def __len__(self):
        return len(self.groups)

    def _selectors(self, indices, device):
        buckets = {}
        for destination, index in enumerate(indices):
            source = bisect_right(self.offsets, index) - 1
            dest, rows = buckets.setdefault(source, ([], []))
            dest.append(destination)
            rows.append(index - self.offsets[source])
        return [(source, torch.tensor(dest, device=device), torch.tensor(rows, device=device))
                for source, (dest, rows) in buckets.items()]

    def _gather(self, key, selectors, shape):
        reference = self.snapshots[0][key]
        count = 1
        for size in shape:
            count *= size
        result = reference.new_empty((count, *reference.shape[1:]))
        for source, destinations, rows in selectors:
            result.index_copy_(0, destinations, self.snapshots[source][key].index_select(0, rows))
        return result.reshape(*shape, *reference.shape[1:])

    @torch.inference_mode(False)
    @torch.no_grad()
    def __iter__(self):
        for sequences in self.groups:
            length, width = max(map(len, sequences)), len(sequences)
            device = self.snapshots[0]["observed"].device
            indices = [s[min(t, len(s)-1)] for t in range(length) for s in sequences]
            selectors = self._selectors(indices, device)
            initial = self._selectors([s[0] for s in sequences], device)
            active = torch.arange(length, device=device)[:, None] < torch.tensor([len(s) for s in sequences], device=device)[None]
            yield PerceptionSequenceBatch(
                **{k: self._gather(k, selectors, (length, width))
                   for k in ("observed", "mapped", "simulated", "env_ids", "episode_ids", "capture_ids")},
                initial_hidden=self._gather("pre_hidden", initial, (width,)),
                active=active, reset=torch.zeros_like(active), anchor_count=sum(map(len, sequences)),
                **{k: self._gather(k, selectors, (length, width))
                   for k in ("history_ids", "history_age") if k in self.snapshots[0]},
            )
