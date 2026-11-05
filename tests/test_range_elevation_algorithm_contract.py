"""Pure CPU checks for elevation encoding metadata and real H PPO/aux updates."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import torch

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.algorithms.range_evidence_amp_ppo import RangeEvidenceAMPPPO
from rsl_rl.algorithms import range_jepa_amp_ppo as range_algorithm
from rsl_rl.algorithms.range_jepa_amp_ppo import RAW_RANGE_KEYS, RangeJEPAAMPPPO, _model_configuration
from test_range_jepa_amp_ppo import TinyEnv, cfg


LEGACY = "depth_valid"
ELEVATION = "depth_elevation"
MISSING = "legacy_missing"
GEOMETRY = dict(image_height=8, image_width=16, hidden_size=16, bank_size=4)


class EncodingEnv(TinyEnv):
    def __init__(self, encoding=LEGACY, spec_encoding=MISSING):
        super().__init__(n=2)
        self.encoding = encoding
        self.range_image_spec = dict(image_height=8, image_width=16, vertical_fov=[52., -90.],
                                     max_range=40., input_normalized=True)
        if spec_encoding != MISSING:
            self.range_image_spec["image_encoding"] = spec_encoding

    def get_observations(self):
        obs = super().get_observations()
        for name, channel in (("range_obs", .25), ("range_map", .5), ("range_sim", .75)):
            valid = obs[name][:, 1:2] > .5
            obs[name][:, :1] *= valid
            if self.encoding == ELEVATION:
                obs[name][:, 1:2] = valid.to(obs[name].dtype) * channel
        return obs

    def step(self, action):
        obs, rewards, dones, extras = super().step(action)
        return obs, rewards[:, :1], dones, extras


def configuration(variant="H", encoding=LEGACY):
    result = cfg()
    result["num_steps_per_env"] = 4
    # No expert AMP data is constructed in this CPU fixture. Use one task
    # critic so AMPPPO does not replace task advantages with absent style rewards.
    result["algorithm"].update(num_learning_epochs=1, num_mini_batches=1, num_critics=1,
                               reward_group_names=["task"], reward_group_weights=[1.])
    perception = dict(**GEOMETRY, num_learning_epochs=1, sequence_batch_size=2, tbptt_steps=2)
    if encoding != MISSING:
        perception["image_encoding"] = encoding
    if variant == "H":
        result["actor"].update(class_name="rsl_rl.models.range_evidence:RangeEvidenceActor",
                                 reader_amp=False, reader_chunk_size=2)
        result["algorithm"].update(class_name="rsl_rl.algorithms.range_evidence_amp_ppo:RangeEvidenceAMPPPO")
        perception.update(variant="H", context_grid=(2, 4))
    result["algorithm"]["perception_cfg"] = perception
    return result


def construct(variant="H", encoding=LEGACY, spec_encoding=None):
    sensor_encoding = LEGACY if encoding == MISSING else encoding
    spec_encoding = sensor_encoding if spec_encoding is None else spec_encoding
    env = EncodingEnv(sensor_encoding, spec_encoding)
    cls = RangeEvidenceAMPPPO if variant == "H" else RangeJEPAAMPPPO
    config = configuration(variant, encoding)
    with redirect_stdout(io.StringIO()):
        algorithm = cls.construct_algorithm(env.get_observations(), env, config, "cpu")
    return algorithm, env


def collect(algorithm, env):
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(4):
            action = algorithm.act(obs)
            obs, rewards, dones, extras = env.step(action)
            algorithm.process_env_step(obs, rewards, dones, extras)
        algorithm.compute_returns(obs)


def old_metadata(saved):
    result = deepcopy(saved)
    result["perception_model_cfg"].pop("image_encoding", None)
    result["perception_cfg"].pop("image_encoding", None)
    result["perception_cfg"].get("model", {}).pop("image_encoding", None)
    return result


class ElevationAlgorithmContractTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(241)

    def assert_state_equal(self, a, b):
        self.assertEqual(tuple(a), tuple(b))
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def assert_finite_metrics(self, metrics):
        self.assertTrue(metrics)
        for name, value in metrics.items():
            self.assertTrue(torch.isfinite(torch.as_tensor(value)).all(), name)

    def assert_ownership(self, algorithm):
        actor = {id(p) for group in algorithm.optimizer.param_groups for p in group["params"]}
        aux = {id(p) for group in algorithm.perception_optimizer.param_groups for p in group["params"]}
        self.assertFalse(actor & aux)
        self.assertLessEqual({id(p) for p in algorithm.perception.parameters()}, actor | aux)
        self.assertLessEqual({id(p) for p in algorithm.actor.reader.parameters()}, actor)
        for module in (algorithm.perception.encoder, algorithm.perception.gru, algorithm.perception.decoder):
            self.assertLessEqual({id(p) for p in module.parameters()}, aux)
        self.assertIs(algorithm.perception.reader, algorithm.actor.reader)

    def assert_compact(self, algorithm):
        observations = algorithm.storage.observations
        self.assertEqual(set(observations.keys()), {"policy", "critic", "amp", "range_features"})
        self.assertFalse(set(observations.keys()) & RAW_RANGE_KEYS)
        self.assertTrue(all(value.is_floating_point() for value in observations.values()))
        self.assertEqual(observations["range_features"].shape, (4, 2, 16 + 2 * 4 * 33))

    def test_model_configuration_defaults_and_explicit_nested_elevation(self):
        for variant in ("A0", "H"):
            with self.subTest(variant=variant):
                model = {**GEOMETRY}
                if variant == "H":
                    model.update(variant="H", context_grid=(2, 4))
                self.assertEqual(_model_configuration(model)["image_encoding"], LEGACY)
                nested = {**model, "model": {"image_encoding": ELEVATION}}
                self.assertEqual(_model_configuration(nested)["image_encoding"], ELEVATION)

    def test_canonical_contract_only_fills_missing_image_encoding(self):
        canonical = range_algorithm._canonical_perception_contract
        original = {"vertical_fov": (52., -90.), "custom": {"retained": [1, 2]}, "hidden_size": 16}
        snapshot = deepcopy(original)
        expected = json.loads(json.dumps({**snapshot, "image_encoding": LEGACY}))
        self.assertEqual(canonical(original), expected)
        self.assertEqual(original, snapshot)
        self.assertNotIn("max_range", canonical(original))
        explicit = {**original, "image_encoding": ELEVATION}
        self.assertEqual(canonical(explicit), json.loads(json.dumps(explicit)))
        self.assertEqual(canonical({}), {"image_encoding": LEGACY})

    def test_construct_accepts_matching_spec_and_old_missing_key_only_for_legacy(self):
        for variant in ("A0", "H"):
            for model_encoding, spec_encoding in ((LEGACY, LEGACY), (LEGACY, MISSING),
                                                   (MISSING, MISSING), (ELEVATION, ELEVATION)):
                with self.subTest(variant=variant, model=model_encoding, spec=spec_encoding):
                    algorithm, _ = construct(variant, model_encoding, spec_encoding)
                    expected = LEGACY if model_encoding == MISSING else model_encoding
                    self.assertEqual(algorithm.perception.image_encoding, expected)

    def test_construct_rejects_bidirectional_spec_encoding_mismatch(self):
        for variant in ("A0", "H"):
            for model_encoding, spec_encoding in ((LEGACY, ELEVATION), (ELEVATION, LEGACY),
                                                   (ELEVATION, MISSING), (MISSING, ELEVATION)):
                with self.subTest(variant=variant, model=model_encoding, spec=spec_encoding):
                    with self.assertRaises(ValueError):
                        construct(variant, model_encoding, spec_encoding)

    def test_full_checkpoints_reject_cross_mode_for_strict_and_nonstrict_perception_load(self):
        enabled_loads = (None, {"perception": True}, {"actor": True},
                         {"actor": False, "perception": True, "critic": False, "optimizer": False})
        for variant in ("A0", "H"):
            for source_mode, target_mode in ((LEGACY, ELEVATION), (ELEVATION, LEGACY)):
                source, _ = construct(variant, source_mode)
                saved = deepcopy(source.save())
                for strict in (True, False):
                    for load_cfg in enabled_loads:
                        with self.subTest(variant=variant, source=source_mode, target=target_mode,
                                          strict=strict, load_cfg=load_cfg):
                            target, _ = construct(variant, target_mode)
                            before = deepcopy(target.perception.state_dict())
                            with self.assertRaises(ValueError):
                                target.load(deepcopy(saved), load_cfg, strict)
                            self.assert_state_equal(before, target.perception.state_dict())

    def test_explicitly_disabled_perception_load_allows_actor_only_transfer(self):
        for variant in ("A0", "H"):
            with self.subTest(variant=variant):
                source, _ = construct(variant, LEGACY)
                target, _ = construct(variant, ELEVATION)
                cnn = deepcopy(target.perception.encoder.state_dict())
                target.load(deepcopy(source.save()),
                            {"actor": True, "perception": False, "critic": False, "optimizer": False}, True)
                self.assert_state_equal(cnn, target.perception.encoder.state_dict())
                self.assertEqual(target.perception.image_encoding, ELEVATION)

    def test_old_metadata_missing_mode_strictly_loads_only_legacy(self):
        for variant in ("A0", "H"):
            with self.subTest(variant=variant):
                source, _ = construct(variant, LEGACY)
                saved = old_metadata(source.save())
                target, _ = construct(variant, LEGACY)
                self.assertTrue(target.load(deepcopy(saved), None, True))
                self.assert_state_equal(source.perception.state_dict(), target.perception.state_dict())
                self.assertEqual(target.save()["perception_model_cfg"]["image_encoding"], LEGACY)
                for strict in (True, False):
                    candidate, _ = construct(variant, ELEVATION)
                    with self.assertRaises(ValueError):
                        candidate.load(deepcopy(saved), None, strict)

    def test_old_metadata_missing_mode_does_not_relax_other_strict_contract_fields(self):
        for variant in ("A0", "H"):
            source, _ = construct(variant, LEGACY)
            legacy = old_metadata(source.save())
            for key, value in (("vertical_fov", [51., -90.]), ("input_normalized", False), ("max_range", 20.)):
                with self.subTest(variant=variant, altered_field=key):
                    saved = deepcopy(legacy)
                    saved["perception_model_cfg"][key] = value
                    target, _ = construct(variant, LEGACY)
                    with self.assertRaises(ValueError):
                        target.load(saved, None, True)
            for key in ("vertical_fov", "input_normalized", "max_range"):
                with self.subTest(variant=variant, missing_field=key):
                    saved = deepcopy(legacy)
                    saved["perception_model_cfg"].pop(key)
                    target, _ = construct(variant, LEGACY)
                    with self.assertRaises(ValueError):
                        target.load(saved, None, True)

    def test_strict_load_still_requires_effective_model_contract_and_weights(self):
        for variant in ("A0", "H"):
            source, _ = construct(variant, LEGACY)
            for key in ("perception_model_cfg", "perception_state_dict"):
                with self.subTest(variant=variant, missing=key):
                    saved = deepcopy(source.save())
                    saved.pop(key)
                    target, _ = construct(variant, LEGACY)
                    with self.assertRaises(KeyError):
                        target.load(saved, None, True)

    def test_elevation_same_mode_serialized_checkpoint_and_list_tuple_metadata_roundtrip(self):
        for variant in ("A0", "H"):
            with self.subTest(variant=variant):
                source, _ = construct(variant, ELEVATION)
                saved = deepcopy(source.save())
                saved["perception_model_cfg"] = json.loads(json.dumps(saved["perception_model_cfg"]))
                stream = io.BytesIO()
                torch.save(saved, stream)
                stream.seek(0)
                saved = torch.load(stream, map_location="cpu", weights_only=False)
                target, _ = construct(variant, ELEVATION)
                self.assertTrue(target.load(saved, None, True))
                self.assertEqual(target.perception_model_cfg["image_encoding"], ELEVATION)
                self.assertEqual(target.save()["perception_model_cfg"]["image_encoding"], ELEVATION)
                self.assert_state_equal(source.perception.state_dict(), target.perception.state_dict())
                self.assert_state_equal(source.actor.state_dict(), target.actor.state_dict())
                if variant == "H":
                    self.assert_ownership(target)

    def test_real_elevation_h_ppo_aux_rollout_keeps_compact_cache_and_shared_ownership(self):
        algorithm, env = construct("H", ELEVATION)
        self.assert_ownership(algorithm)
        self.assert_compact(algorithm)
        collect(algorithm, env)
        self.assertGreater(algorithm.perception_storage.num_anchors, 0)
        self.assert_compact(algorithm)
        cnn_before = algorithm.perception.encoder[0].conv.weight.detach().clone()
        reader_before = algorithm.actor.reader.readout.in_proj_weight.detach().clone()
        phase = {"ppo": False, "ppo_steps": 0, "aux_steps": 0}
        parent_update = AMPPPO.update
        encode = algorithm.perception.encode
        ppo_step = algorithm.optimizer.step
        aux_step = algorithm.perception_optimizer.step

        def guarded_parent(instance):
            phase["ppo"] = True
            try:
                return parent_update(instance)
            finally:
                phase["ppo"] = False

        def guarded_encode(*args, **kwargs):
            self.assertFalse(phase["ppo"], "PPO attempted to encode raw images")
            return encode(*args, **kwargs)

        def guarded_ppo_step(*args, **kwargs):
            if phase["ppo"]:
                phase["ppo_steps"] += 1
                for module in (algorithm.perception.encoder, algorithm.perception.gru,
                               algorithm.perception.decoder, algorithm.actor.reader.support_head):
                    self.assertTrue(all(p.grad is None for p in module.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in algorithm.actor.reader.readout.parameters()))
            return ppo_step(*args, **kwargs)

        def guarded_aux_step(*args, **kwargs):
            phase["aux_steps"] += 1
            self.assertFalse(phase["ppo"])
            for module in (algorithm.perception.encoder, algorithm.perception.gru, algorithm.perception.decoder):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()))
            return aux_step(*args, **kwargs)

        with patch.object(AMPPPO, "update", guarded_parent), \
                patch.object(algorithm.perception, "encode", guarded_encode), \
                patch.object(algorithm.optimizer, "step", guarded_ppo_step), \
                patch.object(algorithm.perception_optimizer, "step", guarded_aux_step):
            result = algorithm.update()
        self.assert_finite_metrics(result)
        self.assertEqual(algorithm.last_perception_receipt["image_encoding"], ELEVATION)
        self.assertGreater(phase["ppo_steps"], 0)
        self.assertGreater(phase["aux_steps"], 0)
        self.assertFalse(torch.equal(cnn_before, algorithm.perception.encoder[0].conv.weight))
        self.assertFalse(torch.equal(reader_before, algorithm.actor.reader.readout.in_proj_weight))
        self.assertEqual(algorithm.perception_storage.num_anchors, 0)
        self.assertFalse(algorithm.last_perception_receipt["ppo_uses_cached_features_only"])
        self.assert_ownership(algorithm)
        self.assert_compact(algorithm)
        resumed, new_env = construct("H", ELEVATION)
        saved = deepcopy(algorithm.save())
        self.assertTrue(resumed.load(saved, None, True))
        self.assert_state_equal(algorithm.perception.state_dict(), resumed.perception.state_dict())
        self.assert_state_equal(algorithm.actor.state_dict(), resumed.actor.state_dict())
        self.assertIsNone(resumed._live_policy.hidden)
        self.assertIsNone(resumed._live_policy.features)
        self.assert_ownership(resumed)
        collect(resumed, new_env)
        self.assert_finite_metrics(resumed.update())
        self.assertEqual(resumed.last_perception_receipt["image_encoding"], ELEVATION)
        self.assertEqual(resumed.perception.image_encoding, ELEVATION)
        self.assert_compact(resumed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
