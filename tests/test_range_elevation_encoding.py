"""CPU contracts for the opt-in two-channel elevation-position encoding."""
from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict
from torch.nn import functional as F

from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAInferencePolicy
from rsl_rl.models.range_evidence import RangeEvidenceActor, RangeEvidenceJEPA
from rsl_rl.models.range_jepa import RangeJEPA, RangeJEPAActor


GEOMETRY = dict(image_height=8, image_width=16, hidden_size=16, bank_size=4)
READER = {**GEOMETRY, "context_grid": (2, 4)}


def make_model(cls, encoding="depth_elevation", **kwargs):
    cfg = dict(GEOMETRY)
    if cls is RangeEvidenceJEPA:
        cfg["context_grid"] = (2, 4)
    return cls(**{**cfg, "image_encoding": encoding, **kwargs})


def image(channel=.5, batch=2):
    x = torch.empty(batch, 2, 8, 16)
    x[:, 0] = torch.linspace(.02, .3, batch * 8 * 16).reshape(batch, 8, 16)
    x[:, 1] = channel
    return x


def loss(model, observed, mapped, simulated, **kwargs):
    if isinstance(model, RangeEvidenceJEPA):
        shape = observed.shape[:2] if observed.ndim == 5 else observed.shape[:1]
        kwargs.setdefault("body_features", torch.zeros(*shape, 64))
    return model.auxiliary_loss(observed, mapped, simulated, **kwargs)


def make_actor(model):
    n = 2
    obs = TensorDict({"policy": torch.randn(n, 6),
                      "range_features": torch.zeros(n, getattr(model, "feature_size", 512))}, [n])
    groups = {"actor": ["policy", "range_features"]}
    cfg = dict(hidden_dims=(16,), obs_normalization=False)
    if isinstance(model, RangeEvidenceJEPA):
        actor = RangeEvidenceActor(obs, groups, "actor", 3, reader_cfg=READER,
                                   reader_amp=False, reader_chunk_size=1, **cfg)
        model.bind_reader(actor.reader)
    else:
        actor = RangeJEPAActor(obs, groups, "actor", 3, **cfg)
    return actor, obs


class RangeElevationEncodingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(837)
        torch.set_num_threads(1)

    def assert_same(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def assert_module_gradient(self, module):
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_opt_in_signature_and_legacy_positional_arguments(self):
        old_base = (8, 16, 32, 16, 4, 8, (52., -90.), 40., 4, True, .5, .01)
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                self.assertEqual(inspect.signature(cls).parameters["image_encoding"].default, "depth_valid")
                args = old_base if cls is RangeJEPA else (*old_base, "H", (2, 4), .1, .1, 3., "linear")
                self.assertEqual(cls(*args).image_encoding, "depth_valid")
                for bad in (None, "", "elevation", "depth_elevations"):
                    with self.assertRaisesRegex(ValueError, "image_encoding"):
                        make_model(cls, bad)

    def test_cnn_keeps_quarter_center_and_three_quarters_with_full_coverage(self):
        x = image(batch=3)
        x[:, 0] = x[0, 0].clone()
        x[:, 1] = torch.tensor([.25, .5, .75])[:, None, None]
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                model = make_model(cls)
                cnn = []
                hook = model.encoder.register_forward_pre_hook(lambda _, args: cnn.append(args[0].detach().clone()))
                encoded = model.encode(x)
                hook.remove()
                self.assert_same(cnn[0], x)
                self.assert_same(encoded["image"][:, 1], torch.ones_like(x[:, 1]))
                self.assert_same(encoded["coverage"], torch.ones_like(encoded["coverage"]))
                self.assertGreater((encoded["tokens"][0] - encoded["tokens"][1]).abs().max().item(), 1e-5)
                self.assertGreater((encoded["tokens"][1] - encoded["tokens"][2]).abs().max().item(), 1e-5)

    def test_empty_nan_depth_limits_and_positive_channel_excursions(self):
        x = torch.zeros(1, 2, 8, 16)
        x[0, :, 0] = torch.tensor([
            [.1, .1, .1, 0., float("nan"), float("inf"), -.1, 1.001,
             .1, .1, .1, .1, .1, 1., 0., .1],
            [.25, .5, .75, 0., .5, .5, .5, .5,
             float("nan"), float("inf"), 0., -.2, 1.2, .5, .5, -float("inf")],
        ])
        expected = torch.zeros_like(x)
        for col in (0, 1, 2, 12, 13):
            expected[0, 0, 0, col] = x[0, 0, 0, col]
            expected[0, 1, 0, col] = 1.
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                model = make_model(cls)
                cnn = []
                hook = model.encoder.register_forward_pre_hook(lambda _, args: cnn.append(args[0].detach().clone()))
                encoded = model.encode(x)
                hook.remove()
                self.assert_same(encoded["image"], expected)
                self.assertTrue(torch.isfinite(cnn[0]).all())
                self.assertEqual(cnn[0][0, 1, 0, 12].item(), x[0, 1, 0, 12].item())
                self.assertEqual(cnn[0][0, 1, 0, 3:].count_nonzero().item(), 2)
                coverage = F.adaptive_avg_pool2d(expected[:, 1:2], (2, 4)).flatten(2).transpose(1, 2)
                self.assert_same(encoded["coverage"], coverage)

    def test_metric_input_normalizes_depth_without_altering_channel(self):
        x = image(.25)
        x[:, 0] *= 40.
        x[0, 0, 0, 0] = 40.
        x[0, 0, 0, 1] = 40.01
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                model = make_model(cls, input_normalized=False)
                encoded = model.encode(x)
                self.assertEqual(encoded["image"][0, 0, 0, 0].item(), 1.)
                self.assertEqual(encoded["image"][0, 1, 0, 0].item(), 1.)
                self.assertEqual(encoded["image"][0, :, 0, 1].count_nonzero().item(), 0)
                self.assert_same(encoded["image"][:, 0, 1:], x[:, 0, 1:] / 40.)

    def test_default_threshold_behavior_is_unchanged(self):
        x = image(batch=4)
        x[:, 1] = torch.tensor([.25, .5, .75, float("inf")])[:, None, None]
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                clean = make_model(cls, "depth_valid")._clean(x)
                self.assertFalse(clean[:2].any())
                self.assert_same(clean[2:, 1], torch.ones_like(clean[2:, 1]))
                self.assert_same(clean[2:, 0], x[2:, 0])

    def test_reconstruction_uses_binary_masks_and_stops_channel_target_gradients(self):
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                model = make_model(cls)
                x = image()
                x[:, 1] = torch.linspace(.25, .75, 16)[None, None]
                x[:, :, 0, :4] = 0
                x.requires_grad_()
                clean = model._clean(x)
                binary_target = torch.cat((x[:, :1], (x[:, :1] > 0).to(x.dtype)), 1)
                prediction = torch.full_like(x, .1)
                actual = model._reconstruction(prediction, clean)
                self.assert_same(actual, model._reconstruction(prediction, binary_target))
                actual.sum().backward()
                self.assertFalse(x.grad[:, 1].any())
                self.assertGreater(x.grad[:, 0].abs().sum().item(), 0)

    def test_evidence_support_bce_and_metrics_use_binary_map_validity(self):
        model = make_model(RangeEvidenceJEPA)
        observed, mapped, simulated = image(.5), image(.25), image(.75)
        mapped[:, :, :2, :4] = 0
        support = torch.ones(2, 8, 16, dtype=torch.bool)
        support[:, :, 8:] = False
        valid = (mapped[:, :1] > 0) & support[:, None]
        expected = F.adaptive_avg_pool2d(valid.float(), (2, 4)).flatten(2).transpose(1, 2)
        with patch("rsl_rl.models.range_evidence.F.binary_cross_entropy_with_logits",
                   wraps=F.binary_cross_entropy_with_logits) as bce:
            result = loss(model, observed, mapped, simulated, support=support)
        support_targets = [call.args[1] for call in bce.call_args_list if call.args[1].ndim == 3]
        self.assertEqual(len(support_targets), 1)
        self.assert_same(support_targets[0], expected)
        for call in bce.call_args_list:
            if call.args[1].ndim == 4:
                self.assertTrue(((call.args[1] == 0) | (call.args[1] == 1)).all())
        self.assert_same(result["map_coverage"], valid.float().mean())
        support_call = next(call for call in bce.call_args_list if call.args[1].ndim == 3)
        expected_bce = F.binary_cross_entropy_with_logits(support_call.args[0], expected)
        torch.testing.assert_close(result["support"], expected_bce)

    def test_shared_codec_trains_continuous_channel_in_all_three_views(self):
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                model = make_model(cls)
                views = [image(c).requires_grad_() for c in (.25, .5, .75)]
                weights = {k: 0. for k in model.default_weights}
                weights.update(codec=1., reg=.1)
                result = loss(model, *views, weights=weights)
                self.assertTrue(torch.isfinite(result["loss"]))
                result["loss"].backward()
                for view in views:
                    self.assertTrue(torch.isfinite(view.grad).all())
                    self.assertGreater(view.grad[:, 1].abs().sum().item(), 0)
                self.assert_module_gradient(model.encoder)
                self.assert_module_gradient(model.decoder)

    def test_default_auxiliary_is_finite_and_recurrent_gradients_cross_timesteps(self):
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                model = make_model(cls)
                observed = torch.stack((image(.25), image(.5))).requires_grad_()
                mapped = torch.stack((image(.5), image(.25))).requires_grad_()
                simulated = torch.stack((image(.75), image(.75))).requires_grad_()
                result = loss(model, observed, mapped, simulated)
                self.assertTrue(all(torch.isfinite(v).all() for v in result.values()))
                recurrent = torch.autograd.grad(result["hidden"].sum(), observed, retain_graph=True)[0]
                self.assertGreater(recurrent[0, :, 1].abs().sum().item(), 0)
                result["loss"].backward()
                for module in (model.encoder, model.gru, model.decoder):
                    self.assert_module_gradient(module)
                if cls is RangeEvidenceJEPA:
                    self.assert_module_gradient(model.reader.readout)
                    self.assert_module_gradient(model.reader.support_head)

    def test_jepa_target_detach_and_body_detach_are_preserved(self):
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                model = make_model(cls)
                views = [image(c).requires_grad_() for c in (.25, .5, .75)]
                weights = {k: 0. for k in model.default_weights}
                weights.update(mem=1., full=1.)
                body = torch.randn(2, 64, requires_grad=True)
                kwargs = {"body_features": body} if cls is RangeEvidenceJEPA else {}
                loss(model, *views, weights=weights, **kwargs)["loss"].backward()
                self.assertGreater(views[0].grad[:, 1].abs().sum().item(), 0)
                self.assertFalse(views[1].grad.any())
                self.assertFalse(views[2].grad.any())
                self.assertIsNone(body.grad)
                self.assert_module_gradient(model.gru)

    def test_empty_frame_holds_memory_and_reset_matches_new_episode(self):
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                model = make_model(cls)
                hidden = torch.randn(2, model.hidden_size)
                empty = torch.zeros(2, 2, 8, 16)
                held = model.step(empty, hidden)
                self.assert_same(held["hidden"], hidden)
                reset = model.step(empty, hidden, reset=torch.tensor([True, False]))
                self.assertFalse(reset["hidden"][0].any())
                self.assert_same(reset["hidden"][1], hidden[1])
                a = model.step(image(.5), hidden, reset=torch.ones(2, dtype=torch.bool))
                b = model.step(image(.5), model.initial_hidden(2))
                for key in a:
                    self.assert_same(a[key], b[key])
                self.assertTrue(torch.isfinite(held["features"]).all())

    def test_real_capture_cache_holds_then_updates_and_resets_without_layout_change(self):
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            for profiled in (False, True):
                with self.subTest(model=cls.__name__, profiled=profiled):
                    from rsl_rl.utils.range_online_profiler import RangeOnlineProfiler
                    model = make_model(cls)
                    actor, obs = make_actor(model)
                    policy = RangeJEPAInferencePolicy(actor, model)
                    profile = RangeOnlineProfiler(True, "cpu") if profiled else None
                    raw = obs.clone()
                    raw.update({"range_obs": image(.25), "range_map": image(.5), "range_sim": image(.75),
                                "range_capture_id": torch.zeros(2, 1, dtype=torch.long),
                                "range_episode_id": torch.zeros(2, 1, dtype=torch.long)})
                    first = policy.prepare(raw, profile=profile)
                    snapshot, hidden = first["range_features"].clone(), policy.hidden.clone()
                    raw["range_obs"][:, 1] = .75
                    raw["range_map"].fill_(float("nan"))
                    raw["range_sim"].fill_(float("inf"))
                    held = policy.prepare(raw, profile=profile)
                    self.assert_same(held["range_features"], snapshot)
                    self.assert_same(policy.hidden, hidden)
                    self.assertFalse(held["range_features"].requires_grad)
                    self.assertFalse(any(key in held for key in ("range_obs", "range_map", "range_sim")))
                    raw["range_capture_id"].fill_(1)
                    updated = policy.prepare(raw, profile=profile)
                    self.assertGreater((updated["range_features"] - snapshot).abs().sum().item(), 0)
                    self.assert_same(first["range_features"], snapshot)
                    expected_size = model.hidden_size + 2 * 4 * 33 if cls is RangeEvidenceJEPA else 512
                    self.assertEqual(updated["range_features"].shape, (2, expected_size))
                    policy.reset(torch.tensor([True, False]))
                    self.assertFalse(policy.hidden[0].any())
                    self.assertFalse(policy.features[0].any())
                    self.assertIsNone(policy.history)

    def test_ppo_trains_reader_but_does_not_backpropagate_to_cnn_gru_or_cache(self):
        model = make_model(RangeEvidenceJEPA)
        actor, obs = make_actor(model)
        x = image(.25).requires_grad_()
        cache = model.step(x, model.initial_hidden(2))["features"]
        cache.retain_grad()
        obs["range_features"] = cache
        actor(obs).square().mean().backward()
        self.assertIs(model.reader, actor.reader)
        for module in (actor.reader.readout, actor.reader.completion_mlp,
                       actor.reader.body_query, actor.prop_mlp):
            self.assert_module_gradient(module)
        for module in (model.encoder, model.gru, model.decoder, actor.reader.support_head):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertIsNone(cache.grad)
        self.assertIsNone(x.grad)

    def test_fp16_storage_roundtrip_and_direct_fp16_encoder_input(self):
        x = image(batch=3)
        x[:, 1] = torch.tensor([.25, .5, .75])[:, None, None]
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                model = make_model(cls)
                stored = x.half()
                encoded = model.encode(stored.float())
                self.assert_same(encoded["image"][:, 1], torch.ones_like(x[:, 1]))
                self.assert_same(encoded["coverage"], torch.ones_like(encoded["coverage"]))
                self.assert_same(encoded["image"][:, 0], stored[:, 0].float())
                result = loss(model, stored.float(), image(.5, 3).half().float(), image(.75, 3).half().float())
                self.assertTrue(torch.isfinite(result["loss"]))
                model = model.half()
                encoded_half = model.encode(stored)
                self.assertEqual(encoded_half["tokens"].dtype, torch.float16)
                self.assertTrue(torch.isfinite(encoded_half["tokens"]).all())
                step = model.step(stored, model.initial_hidden(3))
                self.assertTrue(torch.isfinite(step["features"]).all())

    def test_default_and_explicit_legacy_state_dict_and_outputs_match_exactly(self):
        for cls in (RangeJEPA, RangeEvidenceJEPA):
            with self.subTest(model=cls.__name__):
                cfg = READER if cls is RangeEvidenceJEPA else GEOMETRY
                torch.manual_seed(27)
                implicit = cls(**cfg)
                torch.manual_seed(27)
                explicit = make_model(cls, "depth_valid")
                torch.manual_seed(27)
                candidate = make_model(cls)
                self.assertEqual(tuple(implicit.state_dict()), tuple(explicit.state_dict()))
                self.assertEqual(tuple(implicit.state_dict()), tuple(candidate.state_dict()))
                self.assertEqual(sum(p.numel() for p in implicit.parameters()),
                                 sum(p.numel() for p in candidate.parameters()))
                for key, value in implicit.state_dict().items():
                    self.assert_same(value, explicit.state_dict()[key])
                    self.assert_same(value, candidate.state_dict()[key])
                explicit.load_state_dict(implicit.state_dict(), strict=True)
                x = image(1.)
                h = torch.randn(2, implicit.hidden_size)
                for operation in (lambda m: m.encode(x), lambda m: m.step(x, h),
                                  lambda m: loss(m, x, x, x)):
                    a, b = operation(implicit), operation(explicit)
                    for key in a:
                        self.assert_same(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
