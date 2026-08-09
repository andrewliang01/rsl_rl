# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-first uncertainty-adaptive budget allocation (UABA) primitive.

UABA consumes *already causal* non-negative demand values for the four fixed
support roles.  It does not inspect future contact labels and it deliberately
does not contain a learned budget network, actor, Gym, simulator, or CUDA
integration.  Upstream code may combine causal hazard entropy, censor mass,
evidence staleness, and previous allocation shortfall into the supplied demand.

The allocator has two separate contracts:

1. A bounded largest-remainder apportionment always produces integer role
   targets whose sum is exactly ``M_total`` (``0 <= M_total <= 32``).
2. Candidate availability can only reduce the realized count.  Stable event
   ids are globally deduplicated with deterministic maximum-cardinality
   matching; scarcity never silently moves or shrinks a target budget.

Every sample receives a canonical JSON receipt and SHA-256 digest.  The digest
contains semantic values only (not NumPy strides, byte order, memory addresses,
or device metadata), which makes it suitable for CPU/GPU integration audits.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Any, Final, Mapping, Sequence

import numpy as np


UABA_ROLE_NAMES: Final[tuple[str, ...]] = (
    "left_current_support",
    "left_landing_support",
    "right_current_support",
    "right_landing_support",
)
UABA_NUM_ROLES: Final[int] = len(UABA_ROLE_NAMES)
UABA_MAX_TOTAL_BUDGET: Final[int] = 32
UABA_RECEIPT_SCHEMA: Final[str] = "uaba-causal-budget-receipt-v1"


class UabaContractError(ValueError):
    """Raised when an allocation request violates the UABA contract."""


def _readonly_int_array(value: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


def _readonly_float_array(value: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _strict_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise UabaContractError(f"{name} must be an integer, got {value!r}.")
    normalized = int(value)
    if normalized < minimum or normalized > maximum:
        raise UabaContractError(
            f"{name} must be in [{minimum}, {maximum}], got {normalized}."
        )
    return normalized


def _normalize_demand(causal_demand: Any) -> np.ndarray:
    demand = np.asarray(causal_demand)
    if demand.ndim == 1:
        demand = demand[None, :]
    if demand.ndim != 2 or demand.shape[1] != UABA_NUM_ROLES:
        raise UabaContractError(
            "causal_demand must have shape [B,4] in UABA_ROLE_NAMES order."
        )
    if demand.dtype.kind not in "fiu":
        raise UabaContractError("causal_demand must be a real numeric array.")
    demand = np.asarray(demand, dtype=np.float64)
    if not np.isfinite(demand).all() or np.any(demand < 0.0):
        raise UabaContractError(
            "causal_demand must contain finite non-negative causal values."
        )
    return np.ascontiguousarray(demand)


def _normalize_totals(total_budget: Any, batch_size: int) -> np.ndarray:
    if isinstance(total_budget, Integral) and not isinstance(total_budget, bool):
        value = _strict_int(
            total_budget,
            "total_budget",
            minimum=0,
            maximum=UABA_MAX_TOTAL_BUDGET,
        )
        return np.full(batch_size, value, dtype=np.int64)
    values = np.asarray(total_budget, dtype=object)
    if values.shape != (batch_size,):
        raise UabaContractError("total_budget must be an integer or shape [B].")
    return np.asarray(
        [
            _strict_int(
                value,
                f"total_budget[{index}]",
                minimum=0,
                maximum=UABA_MAX_TOTAL_BUDGET,
            )
            for index, value in enumerate(values.tolist())
        ],
        dtype=np.int64,
    )


def _normalize_role_bounds(
    value: Any,
    name: str,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(value, dtype=object)
    if values.ndim == 0:
        values = np.full((batch_size, UABA_NUM_ROLES), values.item(), dtype=object)
    elif values.shape == (UABA_NUM_ROLES,):
        values = np.broadcast_to(values[None, :], (batch_size, UABA_NUM_ROLES))
    elif values.shape != (batch_size, UABA_NUM_ROLES):
        raise UabaContractError(
            f"{name} must be scalar, shape [4], or shape [B,4]."
        )
    normalized = np.empty((batch_size, UABA_NUM_ROLES), dtype=np.int64)
    for batch_index in range(batch_size):
        for role_index in range(UABA_NUM_ROLES):
            normalized[batch_index, role_index] = _strict_int(
                values[batch_index, role_index],
                f"{name}[{batch_index},{role_index}]",
                minimum=0,
                maximum=UABA_MAX_TOTAL_BUDGET,
            )
    return normalized


def _normalize_candidates(
    candidate_event_ids: Any,
    batch_size: int,
) -> tuple[
    tuple[tuple[tuple[int, ...], ...], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if not isinstance(candidate_event_ids, Sequence) or isinstance(
        candidate_event_ids, (str, bytes)
    ):
        raise UabaContractError(
            "candidate_event_ids must be nested as [B][4][candidate]."
        )
    if len(candidate_event_ids) != batch_size:
        raise UabaContractError(
            "candidate_event_ids outer length must equal demand batch size."
        )
    batches: list[tuple[tuple[int, ...], ...]] = []
    duplicate_count = np.zeros(
        (batch_size, UABA_NUM_ROLES), dtype=np.int64
    )
    candidate_count = np.zeros(
        (batch_size, UABA_NUM_ROLES), dtype=np.int64
    )
    overlap_count = np.zeros(batch_size, dtype=np.int64)
    global_unique_count = np.zeros(batch_size, dtype=np.int64)
    for batch_index, batch_candidates in enumerate(candidate_event_ids):
        if not isinstance(batch_candidates, Sequence) or len(batch_candidates) != UABA_NUM_ROLES:
            raise UabaContractError(
                f"candidate_event_ids[{batch_index}] must contain four role lists."
            )
        roles: list[tuple[int, ...]] = []
        union: set[int] = set()
        role_unique_sum = 0
        for role_index, role_candidates in enumerate(batch_candidates):
            if isinstance(role_candidates, np.ndarray):
                if role_candidates.ndim != 1:
                    raise UabaContractError(
                        "Each candidate role list must be one-dimensional."
                    )
                role_candidates = role_candidates.tolist()
            if not isinstance(role_candidates, Sequence) or isinstance(
                role_candidates, (str, bytes)
            ):
                raise UabaContractError("Each role's candidates must be a sequence.")
            seen: set[int] = set()
            unique: list[int] = []
            for candidate_index, event_id in enumerate(role_candidates):
                if isinstance(event_id, bool) or not isinstance(event_id, Integral):
                    raise UabaContractError(
                        "candidate event ids must be non-negative integers; "
                        f"got {event_id!r} at [{batch_index},{role_index},{candidate_index}]."
                    )
                normalized = int(event_id)
                if normalized < 0 or normalized > np.iinfo(np.int64).max:
                    raise UabaContractError(
                        "candidate event ids must fit non-negative int64."
                    )
                if normalized in seen:
                    duplicate_count[batch_index, role_index] += 1
                    continue
                seen.add(normalized)
                unique.append(normalized)
            roles.append(tuple(unique))
            candidate_count[batch_index, role_index] = len(unique)
            role_unique_sum += len(unique)
            union.update(unique)
        overlap_count[batch_index] = role_unique_sum - len(union)
        global_unique_count[batch_index] = len(union)
        batches.append(tuple(roles))
    return (
        tuple(batches),
        candidate_count,
        duplicate_count,
        overlap_count,
        global_unique_count,
    )


def _bounded_largest_remainder(
    demand: np.ndarray,
    total: int,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:
    """Allocate one sample with stable role-index tie-breaking."""
    target = minimum.astype(np.int64, copy=True)
    remaining = int(total - int(target.sum()))
    while remaining > 0:
        capacity = maximum - target
        active = capacity > 0
        if not np.any(active):  # Feasibility is checked before this function.
            raise AssertionError("UABA feasible capacity unexpectedly exhausted.")
        scaled = np.zeros(UABA_NUM_ROLES, dtype=np.float64)
        active_demand = demand[active]
        largest = float(active_demand.max(initial=0.0))
        if largest > 0.0:
            scaled[active] = active_demand / largest
        if float(scaled.sum()) == 0.0:
            scaled[active] = 1.0
        quota = remaining * scaled / float(scaled.sum())
        floor_quota = np.floor(quota).astype(np.int64)
        grant = np.minimum(floor_quota, capacity)
        if int(grant.sum()) > 0:
            target += grant
            remaining -= int(grant.sum())
            if remaining == 0:
                break
            capacity = maximum - target
        fractional = quota - floor_quota
        order = sorted(
            (role for role in range(UABA_NUM_ROLES) if capacity[role] > 0),
            key=lambda role: (-float(fractional[role]), role),
        )
        progressed = False
        for role in order:
            if remaining == 0:
                break
            if target[role] < maximum[role]:
                target[role] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise AssertionError("UABA largest-remainder allocation made no progress.")
    return target


def _maximum_cardinality_realization(
    candidates: tuple[tuple[int, ...], ...],
    target: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    """Fill role slots with globally unique ids using deterministic matching."""
    slots = tuple(
        (role, slot)
        for role in range(UABA_NUM_ROLES)
        for slot in range(int(target[role]))
    )
    event_to_slot: dict[int, tuple[int, int]] = {}
    slot_to_event: dict[tuple[int, int], int] = {}

    def augment(slot: tuple[int, int], visited: set[int]) -> bool:
        role = slot[0]
        for event_id in candidates[role]:
            if event_id in visited:
                continue
            visited.add(event_id)
            owner = event_to_slot.get(event_id)
            if owner is None or augment(owner, visited):
                event_to_slot[event_id] = slot
                slot_to_event[slot] = event_id
                return True
        return False

    for slot in slots:
        augment(slot, set())

    selected: list[tuple[int, ...]] = []
    for role in range(UABA_NUM_ROLES):
        matched = {
            event_id
            for slot, event_id in slot_to_event.items()
            if slot[0] == role
        }
        selected.append(
            tuple(event_id for event_id in candidates[role] if event_id in matched)
        )
    return tuple(selected)


def _shortfall_reason(
    target: int,
    selected: tuple[int, ...],
    candidates: tuple[int, ...],
) -> str:
    realized = len(selected)
    if target == 0:
        return "target_zero"
    if realized == target:
        return "complete"
    if not candidates:
        return "no_candidates"
    if len(candidates) < target:
        return "candidate_count_below_target"
    return "global_duplicate_conflict"


def _canonical_float(value: float) -> str:
    # Exact binary64 hexadecimal text is independent of array byte order,
    # strides, memory address, and JSON implementation float formatting.
    return float(value).hex()


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class UabaAllocationBatch:
    """Immutable batched allocation result and auditable receipts."""

    demand: np.ndarray
    total_budget: np.ndarray
    per_role_min: np.ndarray
    per_role_max: np.ndarray
    target: np.ndarray
    realized: np.ndarray
    shortfall: np.ndarray
    selected_event_ids: tuple[tuple[tuple[int, ...], ...], ...]
    shortfall_reason: tuple[tuple[str, ...], ...]
    candidate_count: np.ndarray
    global_unique_candidate_count: np.ndarray
    within_role_duplicate_count: np.ndarray
    cross_role_overlap_count: np.ndarray
    receipt_sha256: tuple[str, ...]
    aggregate_sha256: str
    schema: str = UABA_RECEIPT_SCHEMA
    role_names: tuple[str, ...] = UABA_ROLE_NAMES
    causal_only: bool = True
    training_ready: bool = False
    actor_integrated: bool = False
    gpu_required: bool = False

    def receipt(self, batch_index: int) -> Mapping[str, Any]:
        """Reconstruct the canonical semantic receipt for one sample."""
        if isinstance(batch_index, bool) or not isinstance(batch_index, Integral):
            raise IndexError("batch_index must be an integer.")
        index = int(batch_index)
        if index < 0 or index >= self.demand.shape[0]:
            raise IndexError("batch_index out of range.")
        return {
            "schema": self.schema,
            "role_names": list(self.role_names),
            "causal_only": self.causal_only,
            "training_ready": self.training_ready,
            "total_budget": int(self.total_budget[index]),
            "per_role_min": self.per_role_min[index].tolist(),
            "per_role_max": self.per_role_max[index].tolist(),
            "demand": self.demand[index].tolist(),
            "demand_binary64": [
                _canonical_float(value) for value in self.demand[index]
            ],
            "target": self.target[index].tolist(),
            "realized": self.realized[index].tolist(),
            "shortfall": self.shortfall[index].tolist(),
            "selected_event_ids": [
                list(ids) for ids in self.selected_event_ids[index]
            ],
            "shortfall_reason": list(self.shortfall_reason[index]),
            "candidate_count": self.candidate_count[index].tolist(),
            "global_unique_candidate_count": int(
                self.global_unique_candidate_count[index]
            ),
            "within_role_duplicate_count": self.within_role_duplicate_count[
                index
            ].tolist(),
            "cross_role_overlap_count": int(
                self.cross_role_overlap_count[index]
            ),
        }


def allocate_uaba_role_budgets(
    causal_demand: Any,
    candidate_event_ids: Any,
    *,
    total_budget: Any,
    per_role_min: Any = 0,
    per_role_max: Any | None = None,
) -> UabaAllocationBatch:
    """Allocate strict UABA targets and realize them from stable event ids.

    Args:
        causal_demand: ``[B,4]`` finite non-negative values.  UABA treats
            these as precomputed causal quantities and does not infer them.
        candidate_event_ids: Nested ``[B][4][candidate]`` stable integer ids,
            ordered by upstream relevance within each role.
        total_budget: Integer scalar or ``[B]`` values, each at most 32.
        per_role_min: Scalar, ``[4]``, or ``[B,4]`` anti-starvation floors.
        per_role_max: Matching upper bounds.  ``None`` means ``M_total`` for
            every role in each sample.
    """
    demand = _normalize_demand(causal_demand)
    batch_size = demand.shape[0]
    totals = _normalize_totals(total_budget, batch_size)
    minimum = _normalize_role_bounds(per_role_min, "per_role_min", batch_size)
    maximum = (
        np.broadcast_to(totals[:, None], (batch_size, UABA_NUM_ROLES)).copy()
        if per_role_max is None
        else _normalize_role_bounds(
            per_role_max, "per_role_max", batch_size
        )
    )
    if np.any(minimum > maximum):
        raise UabaContractError("per_role_min cannot exceed per_role_max.")
    if np.any(minimum.sum(axis=1) > totals):
        raise UabaContractError(
            "Sum of per-role minimums cannot exceed total_budget."
        )
    if np.any(maximum.sum(axis=1) < totals):
        raise UabaContractError(
            "Sum of per-role maximums must cover total_budget."
        )
    (
        candidates,
        candidate_count,
        within_duplicates,
        cross_overlap,
        global_unique_count,
    ) = _normalize_candidates(candidate_event_ids, batch_size)

    target = np.empty((batch_size, UABA_NUM_ROLES), dtype=np.int64)
    selected_batches: list[tuple[tuple[int, ...], ...]] = []
    reasons: list[tuple[str, ...]] = []
    for batch_index in range(batch_size):
        target[batch_index] = _bounded_largest_remainder(
            demand[batch_index],
            int(totals[batch_index]),
            minimum[batch_index],
            maximum[batch_index],
        )
        selected = _maximum_cardinality_realization(
            candidates[batch_index], target[batch_index]
        )
        selected_batches.append(selected)
        reasons.append(
            tuple(
                _shortfall_reason(
                    int(target[batch_index, role]),
                    selected[role],
                    candidates[batch_index][role],
                )
                for role in range(UABA_NUM_ROLES)
            )
        )
    selected_tuple = tuple(selected_batches)
    reason_tuple = tuple(reasons)
    realized = np.asarray(
        [
            [len(selected_tuple[b][role]) for role in range(UABA_NUM_ROLES)]
            for b in range(batch_size)
        ],
        dtype=np.int64,
    )
    shortfall = target - realized

    provisional = UabaAllocationBatch(
        demand=_readonly_float_array(demand),
        total_budget=_readonly_int_array(totals),
        per_role_min=_readonly_int_array(minimum),
        per_role_max=_readonly_int_array(maximum),
        target=_readonly_int_array(target),
        realized=_readonly_int_array(realized),
        shortfall=_readonly_int_array(shortfall),
        selected_event_ids=selected_tuple,
        shortfall_reason=reason_tuple,
        candidate_count=_readonly_int_array(candidate_count),
        global_unique_candidate_count=_readonly_int_array(global_unique_count),
        within_role_duplicate_count=_readonly_int_array(within_duplicates),
        cross_role_overlap_count=_readonly_int_array(cross_overlap),
        receipt_sha256=(),
        aggregate_sha256="",
    )
    receipt_hashes = tuple(
        _receipt_digest(provisional.receipt(index))
        for index in range(batch_size)
    )
    aggregate_hash = hashlib.sha256(
        json.dumps(
            {
                "schema": UABA_RECEIPT_SCHEMA,
                "receipt_sha256": list(receipt_hashes),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return UabaAllocationBatch(
        demand=provisional.demand,
        total_budget=provisional.total_budget,
        per_role_min=provisional.per_role_min,
        per_role_max=provisional.per_role_max,
        target=provisional.target,
        realized=provisional.realized,
        shortfall=provisional.shortfall,
        selected_event_ids=provisional.selected_event_ids,
        shortfall_reason=provisional.shortfall_reason,
        candidate_count=provisional.candidate_count,
        global_unique_candidate_count=provisional.global_unique_candidate_count,
        within_role_duplicate_count=provisional.within_role_duplicate_count,
        cross_role_overlap_count=provisional.cross_role_overlap_count,
        receipt_sha256=receipt_hashes,
        aggregate_sha256=aggregate_hash,
    )


__all__ = [
    "UABA_MAX_TOTAL_BUDGET",
    "UABA_NUM_ROLES",
    "UABA_RECEIPT_SCHEMA",
    "UABA_ROLE_NAMES",
    "UabaAllocationBatch",
    "UabaContractError",
    "allocate_uaba_role_budgets",
]
