"""Opt-in disjoint online stream intervals, resolved once per rollout.

CUDA events measure stream elapsed intervals, including dispatch gaps inside a
scope; they are not sums of kernel execution times. Host submission intervals
are supplemental and must never be added to the GPU intervals. No reductions
across ranks occur here.
"""
from __future__ import annotations
import time
import torch


class RangeOnlineProfiler:
    PHASES = ("validate_select", "cache_enqueue", "online_cnn", "memory_attention", "publish", "parent_act")

    def __init__(self, enabled: bool, device):
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self.cuda = self.enabled and self.device.type == "cuda"
        self.records = []
        self.active = None

    def mark(self, name=None):
        """Close the last scope and optionally start the next; never wait."""
        if not self.enabled:
            return
        if name is not None and name not in self.PHASES:
            raise ValueError("Unknown online profiling phase.")
        event = None
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
        now = time.perf_counter()
        if self.active is not None:
            previous, host_start, start_event = self.active
            self.records.append((previous, now-host_start, start_event, event))
        self.active = (name, now, event) if name is not None else None

    def finish(self, *, synchronize=True):
        """Resolve all scopes once, then clear for the next rollout."""
        if not self.enabled:
            return {}
        if self.active is not None:
            raise ValueError("Online timing scope still open.")
        if self.cuda and self.records and synchronize:
            self.records[-1][3].synchronize()
        result = {name+"_s": 0.0 for name in self.PHASES}
        result.update({"host/"+name+"_s": 0.0 for name in self.PHASES})
        result.update({name+"_calls": 0 for name in self.PHASES})
        for name, host, start, stop in self.records:
            result[name+"_s"] += start.elapsed_time(stop)/1000.0 if self.cuda else host
            result["host/"+name+"_s"] += host
            result[name+"_calls"] += 1
        result["scoped_stream_s"] = sum(result[n+"_s"] for n in self.PHASES)
        result["host/scoped_s"] = sum(result["host/"+n+"_s"] for n in self.PHASES)
        result["cuda_event_timing"] = float(self.cuda)
        result["same_rank_additive"] = 1.0
        self.records.clear()
        return result
