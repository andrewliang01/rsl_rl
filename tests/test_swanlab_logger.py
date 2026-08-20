from __future__ import annotations

from types import SimpleNamespace

import torch

from rsl_rl.utils.swanlab_utils import SwanLabSummaryWriter


class _FakeRun:
    def __init__(self) -> None:
        self.config = SimpleNamespace(update=lambda values: setattr(self, "stored_config", values))
        self.logged: list[tuple[dict, int | None]] = []
        self.saved: list[tuple[str, str | None, str]] = []
        self.finished = False

    def log(self, values: dict, step: int | None = None) -> None:
        self.logged.append((values, step))

    def save(self, path: str, base_path: str | None = None, policy: str = "live") -> None:
        self.saved.append((path, base_path, policy))

    def finish(self) -> None:
        self.finished = True


def test_swanlab_writer_keeps_tensorboard_and_mirrors_scalars(monkeypatch, tmp_path):
    fake_run = _FakeRun()
    init_kwargs = {}

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return fake_run

    monkeypatch.setattr("rsl_rl.utils.swanlab_utils.swanlab.init", fake_init)
    writer = SwanLabSummaryWriter(
        log_dir=str(tmp_path / "run_a"),
        flush_secs=1,
        cfg={"swanlab_project": "b2-piper-pos-orn"},
    )
    writer.add_scalar("Loss/test", torch.tensor(1.25), global_step=7)
    writer.store_config({"seed": 42}, {"logger": "swanlab"})
    provenance = tmp_path / "git.diff"
    provenance.write_text("clean")
    writer.save_file(str(provenance))
    writer.stop()

    assert init_kwargs["project"] == "b2-piper-pos-orn"
    assert init_kwargs["name"] == "run_a"
    assert fake_run.logged == [({"Loss/test": 1.25}, 7)]
    assert fake_run.stored_config == {"train_cfg": {"logger": "swanlab"}, "env_cfg": {"seed": 42}}
    assert fake_run.saved[0][2] == "now"
    assert fake_run.finished
    assert list((tmp_path / "run_a").glob("events.out.tfevents.*"))
