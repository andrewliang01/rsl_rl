"""Opt-in local-rank auxiliary timing; CUDA events never synchronize per phase."""
from __future__ import annotations
from contextlib import contextmanager
import time
import torch


class RangeAuxProfiler:
    TOP_PHASES = ("pack_sequences", "forward_loss", "backward", "gradient_sync_layout", "clip_adam")
    FORWARD_PHASES = ("shared_cnn", "memory_readout_completion", "decoder", "loss_regularizer_mask")

    def __init__(self, enabled: bool, device):
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self.cuda = self.enabled and self.device.type == "cuda"
        self.records = []
        self.counters: dict[str, int] = {}
        self.active_top = None
        self.active_forward = None
        self.host_start = time.perf_counter()
        self.start = self.end = None
        self._finished = None
        if self.cuda:
            self.start, self.end = self._events()
            self.start.record(torch.cuda.current_stream(self.device))

    @staticmethod
    def _events():
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def count(self, name: str, amount: int = 1):
        """CPU-only exact shape/accounting counts; never inspect a CUDA tensor."""
        if not self.enabled:
            return
        if type(amount) is not int or amount < 0:
            raise ValueError("Profile counts require nonnegative Python integers.")
        self.counters[name] = self.counters.get(name, 0) + amount

    @contextmanager
    def phase(self, name: str, domain: str = "top"):
        if not self.enabled:
            yield
            return
        if domain == "top":
            if name not in self.TOP_PHASES or self.active_top is not None:
                raise ValueError("Top-level auxiliary phases must be known and non-overlapping.")
            self.active_top = name
        elif domain == "forward":
            if name not in self.FORWARD_PHASES or self.active_top != "forward_loss" or self.active_forward is not None:
                raise ValueError("Forward subphases must be non-overlapping children of forward_loss.")
            self.active_forward = name
        else:
            raise ValueError("Unknown profiling domain.")
        before = time.perf_counter()
        start = stop = None
        if self.cuda:
            start, stop = self._events()
            start.record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            if self.cuda:
                stop.record(torch.cuda.current_stream(self.device))
            host = time.perf_counter() - before
            self.records.append((domain, name, host, start, stop))
            if domain == "top":
                self.active_top = None
            else:
                self.active_forward = None

    def finish(self):
        """One device synchronization, then resolve every recorded CUDA event.

        Top stream intervals are disjoint. The remaining measured wall time is
        explicitly 'other', so the reported local-rank top terms add to wall.
        Forward children describe their parent and must NOT be added again.
        Host submission times are supplemental and can overlap device work.
        """
        if self._finished is not None:
            return dict(self._finished)
        if not self.enabled:
            self._finished = {}
            return {}
        if self.active_top is not None or self.active_forward is not None:
            raise ValueError("Cannot finish while a timed phase remains open.")
        if self.cuda:
            self.end.record(torch.cuda.current_stream(self.device))
            torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - self.host_start
        metrics = {name + "_s": 0.0 for name in self.TOP_PHASES}
        metrics.update({"forward/" + name + "_s": 0.0 for name in self.FORWARD_PHASES})
        metrics.update({"host/" + name + "_s": 0.0 for name in self.TOP_PHASES})
        metrics.update({name + "_calls": 0 for name in self.TOP_PHASES})
        metrics.update({"forward/" + name + "_calls": 0 for name in self.FORWARD_PHASES})
        for domain, name, host, start, stop in self.records:
            seconds = start.elapsed_time(stop) / 1000.0 if self.cuda else host
            key = name + "_s" if domain == "top" else "forward/" + name + "_s"
            metrics[key] += seconds
            metrics[key[:-2] + "_calls"] += 1
            if domain == "top":
                metrics["host/" + name + "_s"] += host
        top_sum = sum(metrics[name + "_s"] for name in self.TOP_PHASES)
        metrics["aux_wall_s"] = wall
        metrics["other_s"] = wall - top_sum
        metrics["forward/other_s"] = metrics["forward_loss_s"] - sum(metrics["forward/" + name + "_s"] for name in self.FORWARD_PHASES)
        metrics["cuda_event_timing"] = float(self.cuda)
        metrics["same_rank_additive"] = 1.0
        if self.cuda:
            metrics["cuda_stream_span_s"] = self.start.elapsed_time(self.end) / 1000.0
            metrics["cuda_stream_other_s"] = metrics["cuda_stream_span_s"] - top_sum
        metrics.update(self.counters)
        if "active_image_rows" in self.counters:
            encoded = self.counters.get("encoded_image_rows", 0)
            active = self.counters["active_image_rows"]
            if active > encoded:
                raise ValueError("Active image rows exceed encoded rows; profiling accounting is inconsistent.")
            metrics["encoded_image_rows"] = encoded
            metrics["padding_image_rows"] = encoded - active
            metrics["padding_ratio"] = (encoded - active) / encoded if encoded else 0.0
        self._finished = metrics
        return dict(metrics)
