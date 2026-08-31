"""Contracts for the composed UniFP and AMP PPO algorithm."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

import rsl_rl.algorithms.unifp_amp_ppo as unifp_amp_module
from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.algorithms.unifp_adaptation_ppo import UniFPAdaptationMixin
from rsl_rl.algorithms.unifp_amp_ppo import UniFPAMPAdaptationPPO


def test_unifp_amp_algorithm_composes_both_feature_paths():
    assert issubclass(UniFPAMPAdaptationPPO, UniFPAdaptationMixin)
    assert issubclass(UniFPAMPAdaptationPPO, AMPPPO)
    assert UniFPAMPAdaptationPPO.__mro__[:3] == (
        UniFPAMPAdaptationPPO,
        UniFPAdaptationMixin,
        AMPPPO,
    )


def test_constructor_keeps_amp_observation_in_rollout(monkeypatch):
    captured = {}

    def fake_builder(algorithm_class, obs, env, cfg, device, **kwargs):
        captured.update(
            algorithm_class=algorithm_class,
            obs=obs,
            env=env,
            cfg=cfg,
            device=device,
            kwargs=kwargs,
        )
        return "composed"

    monkeypatch.setattr(unifp_amp_module, "construct_single_critic_algorithm", fake_builder)
    obs = object()
    env = object()
    cfg = {"algorithm": {"amp_cfg": {}}}
    result = UniFPAMPAdaptationPPO.construct_algorithm(obs, env, cfg, "cuda:0")

    assert result == "composed"
    assert captured["algorithm_class"] is UniFPAMPAdaptationPPO
    assert captured["obs"] is obs
    assert captured["env"] is env
    assert captured["cfg"] is cfg
    assert captured["device"] == "cuda:0"
    assert captured["kwargs"] == {"include_amp_obs": True}


def test_update_reports_amp_and_unifp_losses_together(monkeypatch):
    algorithm = UniFPAMPAdaptationPPO.__new__(UniFPAMPAdaptationPPO)
    algorithm._update_count = 0
    algorithm.freeze_adaptation_after_iter = None
    algorithm.adaptation_frozen = False
    algorithm.adaptation_part_names = ()
    algorithm.adaptation_loss_types = ()
    algorithm.reconstruction_obs_group = None
    algorithm.is_multi_gpu = False

    class _Storage:
        def mini_batch_generator(self, num_mini_batches, num_learning_epochs=8):
            del num_mini_batches, num_learning_epochs
            yield SimpleNamespace()

    algorithm.storage = _Storage()

    def fake_amp_update(self):
        for _ in self.storage.mini_batch_generator(1, 1):
            pass
        return {"amp": 2.0}

    def fake_adaptation_step(self, batch, logs, *, train):
        del self, batch
        assert train
        logs["adaptation"] += 3.0
        logs["_count"] += 1.0

    monkeypatch.setattr(AMPPPO, "update", fake_amp_update)
    algorithm._adaptation_step = MethodType(fake_adaptation_step, algorithm)
    losses = algorithm.update()

    assert losses["amp"] == 2.0
    assert losses["adaptation"] == 3.0
    assert losses["adaptation_frozen"] == 0.0
    assert algorithm._update_count == 1


def test_save_chains_amp_and_adaptation_state(monkeypatch):
    algorithm = UniFPAMPAdaptationPPO.__new__(UniFPAMPAdaptationPPO)
    parameter = torch.nn.Parameter(torch.ones(()))
    algorithm.adaptation_optimizer = torch.optim.Adam([parameter], lr=1.0e-4)
    algorithm._update_count = 17
    algorithm.adaptation_frozen = False

    monkeypatch.setattr(
        AMPPPO,
        "save",
        lambda self: {"amp_discriminator_state_dict": {"weight": torch.ones(1)}},
    )
    saved = algorithm.save()

    assert "amp_discriminator_state_dict" in saved
    assert "adaptation_optimizer_state_dict" in saved
    assert saved["adaptation_update_count"] == 17
    assert saved["adaptation_frozen"] is False
