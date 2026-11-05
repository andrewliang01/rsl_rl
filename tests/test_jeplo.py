"""Reference equivalence, deployable-input isolation and sequence integration."""
from copy import deepcopy
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.algorithms.jeplo_amp_ppo import JEPLOAMPPPO, StairMasterAMP
from rsl_rl.algorithms.multi_ppo import MultiPPO
from rsl_rl.models import AMPDiscriminator
from rsl_rl.models.jeplo import JEPLOEstimator, JEPLOState, scale_policy_body
from rsl_rl.storage import ReplayBuffer
from rsl_rl.storage.jeplo_batches import StreamingCTSStorage
from rsl_rl.third_party.jeplo.rollout_storage_jepa_ss import RolloutStorageJepaSS


def observations(n=8, actions=2, t=0):
    p = 9 + 3 * actions
    return TensorDict({
        "policy": torch.randn(n, p), "critic": torch.randn(n, p+3),
        "height_scan_critic": torch.randn(n, 12), "amp": torch.randn(n, 3),
        "jeplo_depth": torch.rand(n, 2, 40, 96),
        "jeplo_control_id": torch.full((n, 1), t, dtype=torch.long),
        "jeplo_capture_id": torch.full((n, 1), t//5, dtype=torch.long),
        "jeplo_episode_id": torch.zeros(n, 1, dtype=torch.long),
    }, batch_size=[n])


def algorithm(n=8, actions=2, steps=12):
    cfg = {"num_steps_per_env": steps, "actor": {"hidden_dims": [32, 32]},
           "critic": {"hidden_dims": [32, 32]}, "algorithm": {
               "num_learning_epochs": 1, "num_mini_batches": 2, "amp_cfg": None,
               "jeplo_cfg": {"warmup_iters": (0, 0), "jepa_epochs": 1},
           }}
    with contextlib.redirect_stdout(io.StringIO()):
        return JEPLOAMPPPO(observations(n, actions), SimpleNamespace(num_envs=n, num_actions=actions), cfg, "cpu")


class TestJEPLO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_image_trunk_is_original_dimensions_except_position_grid(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = JEPLOEstimator(temporal_steps=10, num_one_step_obs=96,
                                   prop_hist_dim=960, num_depth_channels=2, action_hist_dim=145)
        self.assertEqual(model.depth_encoder.patch_embed.in_channels, 2)
        self.assertEqual(model.depth_encoder.patch_embed.kernel_size, (4, 8))
        self.assertEqual(len(model.depth_encoder.blocks), 6)
        self.assertEqual(model.depth_encoder.blocks[0].num_heads, 4)
        self.assertEqual(model.depth_encoder.pos_embed.shape, (1, 120, 128))
        self.assertEqual(model.pred_gru.input_size, 96)
        self.assertEqual(model.pred_gru.hidden_size, 512)

    def test_g1_body_order_scaling_keeps_every_joint_and_command(self):
        p = torch.arange(96.).reshape(1, -1)
        out = scale_policy_body(p, 29)
        torch.testing.assert_close(out[:, :3], p[:, :3] * .25)
        torch.testing.assert_close(out[:, 6:9], p[:, 64:67] * torch.tensor((2., 2., .25)))
        torch.testing.assert_close(out[:, 9:38], p[:, 6:35])
        torch.testing.assert_close(out[:, 38:67], p[:, 35:64] * .05)
        torch.testing.assert_close(out[:, 67:], p[:, 67:] * .1)

    def test_held_control_is_idempotent_and_episode_reset_removes_history(self):
        obs = observations()
        state = JEPLOState(2)
        first = state.observe(obs).clone()
        state.hidden.fill_(3)
        state.features.fill_(4)
        torch.testing.assert_close(state.observe(obs), first)
        obs["jeplo_control_id"] += 1
        state.observe(obs)
        torch.testing.assert_close(state.body[:, 1], first[:, :15])
        obs["jeplo_episode_id"][0] += 1
        state.observe(obs)
        self.assertTrue(torch.all(state.body[0, 1:] == 0))
        self.assertTrue(torch.all(state.hidden[0] == 0))
        self.assertTrue(torch.all(state.features[0] == 0))
        self.assertEqual(state.capture[0].item(), -1)

    def test_deployed_student_does_not_read_privileged_groups(self):
        alg = algorithm()
        alg.eval_mode()
        obs = observations()
        public = obs.exclude("critic", "height_scan_critic", "amp")
        full = alg.get_policy()(obs)
        alg.get_policy().reset()
        only_public = alg.get_policy()(public)
        torch.testing.assert_close(full, only_public, rtol=0, atol=0)

    def test_gradient_boundaries_match_original_student_distillation(self):
        alg = algorithm()
        obs = observations()
        alg.policy.state.observe(obs)
        packed = alg.policy.pack_training_observations(obs)
        features = torch.randn(8, 512, requires_grad=True)
        actions = alg.network.act_student(packed, features)
        loss = -alg.network.get_actions_log_prob(actions).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in alg.network.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in alg.network.student_encoder.parameters()))
        self.assertIsNone(features.grad)
        alg.network.zero_grad(set_to_none=True)
        student = alg.network.encode_student_latent(packed, features.detach())
        with torch.no_grad():
            teacher = alg.network.encode_teacher_latent(packed)
        (student-teacher).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in alg.network.student_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in alg.network.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in alg.estimator.parameters()))

    def test_sequence_update_and_resume_next_iteration(self):
        alg = algorithm()
        obs = observations()
        for t in range(12):
            with torch.inference_mode():
                action = alg.act(obs)
                self.assertEqual(action.shape, (8, 2))
                nxt = observations(t=t+1)
                dones = torch.zeros(8, dtype=torch.long)
                if t == 3:
                    dones[0] = 1
                    nxt["jeplo_episode_id"][0] = 1
                elif t > 3:
                    nxt["jeplo_episode_id"][0] = 1
                alg.process_env_step(nxt, torch.ones(8), dones, {})
                obs = nxt
        with torch.inference_mode():
            alg.compute_returns(obs)
        self.assertEqual(alg.estimator_storage.step, 3)
        self.assertEqual(alg.estimator_storage.dones[0, 0].item(), 1)
        losses = alg.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in losses.values()))
        self.assertEqual(losses["jepa_updates"], 1)
        saved = deepcopy(alg.save())
        saved["iter"] = 0
        restored = algorithm()
        self.assertTrue(restored.load(saved))
        self.assertEqual(saved["iter"], 1)
        self.assertEqual(restored.next_iteration, 1)
        self.assertEqual(restored.cts.optimizer.state_dict()["param_groups"],
                         alg.cts.optimizer.state_dict()["param_groups"])
        for key, value in alg.estimator.state_dict().items():
            torch.testing.assert_close(restored.estimator.state_dict()[key], value)

    def test_original_aux_sampling_caps_at_256_and_one_batch_per_epoch(self):
        storage = RolloutStorageJepaSS(4096, 2, (1,), (2, 1, 1), (1,), 1)
        storage.step = 2
        batches = list(storage.mini_batch_generator(2, 5))
        self.assertEqual(len(batches), 5)
        self.assertTrue(all(b["depth_stack"].shape[:2] == (2, 256) for b in batches))

    def test_streaming_batches_match_original_values_order_and_reiteration(self):
        storage = StreamingCTSStorage(num_envs=8, teacher_num_envs=6,
                                      num_transitions_per_env=12, obs_shape=[17],
                                      high_feature_shape=[4], actions_shape=[2])
        for key in ("observations", "high_features", "actions", "values", "advantages",
                    "returns", "actions_log_prob", "mu", "sigma"):
            getattr(storage, key).normal_()
        torch.manual_seed(88)
        expected = list(storage.mini_batch_generator(2, 3))
        torch.manual_seed(88)
        plan = storage.reusable_batches(2, 3)
        for replay in (list(plan), list(plan)):
            self.assertEqual(len(replay), len(expected))
            for left, right in zip(expected, replay):
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_streaming_cts_loss_gradients_and_adam_equal_original_cached_update(self):
        source = Path(os.environ.get("JEPLO_REFERENCE_ROOT", "/home/andew/RL/JEPLO"))
        if not source.exists():
            self.skipTest("Local JEPLO source required for original cached-update oracle.")
        import types
        raw = (source/"training/rsl_rl/rsl_rl/algorithms/teacher_cts.py").read_text()
        raw = raw.replace("from rsl_rl.modules.actor_critic_teacher_cts import ActorCriticTeacherCTS",
                          "from rsl_rl.third_party.jeplo.actor_critic_teacher_cts import ActorCriticTeacherCTS")
        raw = raw.replace("from rsl_rl.storage.rollout_storage_teacher_cts import RolloutStorageTeacherCTS",
                          "from rsl_rl.third_party.jeplo.rollout_storage_teacher_cts import RolloutStorageTeacherCTS")
        oracle = {}
        exec(compile(raw, str(source/"training/rsl_rl/rsl_rl/algorithms/teacher_cts.py"), "exec"), oracle)
        a, b = algorithm(), algorithm()
        b.load(deepcopy(a.save()))
        a.cts.update = types.MethodType(oracle["TeacherCTS"].update, a.cts)
        obs = observations()
        for t in range(12):
            torch.manual_seed(400+t)
            left = a.act(obs)
            torch.manual_seed(400+t)
            right = b.act(obs)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            nxt = observations(t=t+1)
            for alg in (a, b):
                alg.process_env_step(nxt, torch.ones(8), torch.zeros(8, dtype=torch.long), {})
            obs = nxt
        for alg in (a, b):
            alg.compute_returns(obs)
        torch.manual_seed(341)
        left = a.cts.update()
        torch.manual_seed(341)
        right = b.cts.update()
        self.assertEqual(left, right)
        for (name, p), (other, q) in zip(a.network.named_parameters(), b.network.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_amp_mixing_matches_existing_amp_ppo_terminal_contract(self):
        disc = AMPDiscriminator(6, [8, 4], amp_reward_coef=.35, task_reward_lerp=.8)
        previous = torch.randn(8, 3)
        obs = {"amp": torch.randn(8, 3)}
        rewards = torch.randn(8)
        dones = torch.tensor([0, 1, 0, 0, 0, 1, 0, 0])
        for extras in ({}, {"amp_next_obs": torch.randn(8, 3)}):
            current = StairMasterAMP.__new__(StairMasterAMP)
            current.amp_discriminator = deepcopy(disc)
            current.amp_storage = ReplayBuffer(3, 32, "cpu")
            mixed = current.mix(previous, obs, rewards, dones, extras)
            reference = AMPPPO.__new__(AMPPPO)
            reference.__dict__.update(
                amp_discriminator=deepcopy(disc), amp_storage=ReplayBuffer(3, 32, "cpu"),
                transition=SimpleNamespace(observations={"amp": previous}),
                _gmq_compat=False, is_multi_critic=False,
                _extract_amp_task_rewards=lambda x: x,
            )
            with patch.object(MultiPPO, "process_env_step"):
                AMPPPO.process_env_step(reference, obs, rewards, dones, extras)
            torch.testing.assert_close(mixed, reference.final_rewards)
            torch.testing.assert_close(current.style_rewards, reference.style_rewards)
            self.assertEqual(current.amp_storage.num_samples, reference.amp_storage.num_samples)

    def test_vendored_source_receipt_matches_upstream_and_ported_files(self):
        root = Path(__file__).parents[1] / "rsl_rl/third_party/jeplo"
        source = Path(os.environ.get("JEPLO_REFERENCE_ROOT", "/home/andew/RL/JEPLO"))
        manifest = json.loads((root/"PROVENANCE.json").read_text())
        for record in manifest["files"]:
            data = (root/record["destination"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["ported_sha256"])
            if source.exists():
                self.assertEqual(hashlib.sha256((source/record["source"]).read_bytes()).hexdigest(),
                                 record["source_sha256"])

    def test_original_forward_loss_gradient_and_adam_step_equivalence(self):
        source = Path(os.environ.get("JEPLO_REFERENCE_ROOT", "/home/andew/RL/JEPLO"))
        if not source.exists():
            self.skipTest("Local original JEPLO checkout needed for independent numerical oracle.")
        # Load original LeJEPA's minimal dependency closure, without borrowing
        # the vendored implementation or installing its older RSL-RL package.
        def package(name, path):
            module = ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
            return module
        base = source/"training/lejepa/lejepa"
        root = package("jeplo_oracle_lejepa", base)
        uv = package("jeplo_oracle_lejepa.univariate", base/"univariate")
        mv = package("jeplo_oracle_lejepa.multivariate", base/"multivariate")
        def module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            obj = importlib.util.module_from_spec(spec)
            sys.modules[name] = obj
            spec.loader.exec_module(obj)
            return obj
        module(uv.__name__+".base", base/"univariate/base.py")
        uv.EppsPulley = module(uv.__name__+".epps_pulley", base/"univariate/epps_pulley.py").EppsPulley
        mv.SlicingUnivariateTest = module(mv.__name__+".slicing", base/"multivariate/slicing.py").SlicingUnivariateTest
        root.univariate, root.multivariate = uv, mv
        with patch.dict(sys.modules, {"lejepa": root}):
            original = module("jeplo_oracle_estimator", source/"training/rsl_rl/rsl_rl/modules/jepa_estimator_sensor_ss.py")
        args = dict(temporal_steps=10, num_one_step_obs=45, prop_hist_dim=450,
                    num_depth_channels=2, action_hist_dim=60)
        with contextlib.redirect_stdout(io.StringIO()):
            ref = original.JepaEstimatorSensorSS(**args)
            port = JEPLOEstimator(image_height=25, image_width=60, **args)
        port.load_state_dict(ref.state_dict())
        batch = dict(prop_hist=torch.randn(3, 4, 450), depth_stack=torch.rand(3, 4, 2, 25, 60),
                     action_hist=torch.randn(3, 4, 60), hidden_states=torch.randn(3, 4, 512),
                     dones=torch.zeros(3, 4, 1, dtype=torch.bool))
        ref.eval();port.eval()
        x = (batch["prop_hist"][0], batch["depth_stack"][0], batch["hidden_states"][0])
        for a, b in zip(ref(*x), port(*x)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        ref.train();port.train()
        torch.manual_seed(317)
        left = ref.update(batch)
        torch.manual_seed(317)
        right = port.update(batch)
        self.assertEqual(left, right)
        for (name, a), (other, b) in zip(ref.named_parameters(), port.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
