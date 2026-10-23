"""History clock, replay ownership, and actual PPO integration regressions."""

import torch

from rsl_rl.algorithms.range_jepa_amp_ppo import _model_configuration
from rsl_rl.storage.perception_storage import PerceptionStorage
from rsl_rl.storage.range_history import RangeHistory


class ScalarEncoder:
    scale = 1.0

    def encode_history(self, images):
        return {"tokens": images[:, 0, 0, 0, None, None] * self.scale, "valid": images[:, 1, 0, 0, None].bool()}


def images(value, n=2):
    x = torch.ones(n, 2, 2, 2)
    x[:, 0] *= value
    return x


def test_exact_clock_reset_and_missing_capture():
    h = RangeHistory(2, [1, 2, 10], 0.1, "cpu")
    enc = ScalarEncoder()
    env = torch.arange(2)
    for t in range(11):
        ids, age, context = h.select(env, torch.full((2,), t))
        if t == 10:
            torch.testing.assert_close(age[0], torch.tensor([0.1, 0.2, 1.0]))
            torch.testing.assert_close(context["history_tokens"][0, :, 0, 0], torch.tensor([9.0, 8.0, 0.0]))
        x = images(t)
        h.append(env, torch.full((2,), t), x, enc.encode_history(x))
    h.reset(torch.tensor([True, False]))
    ids, age, context = h.select(env, torch.tensor([11, 13]))
    assert (ids[0] == -1).all()
    assert (ids[1, :2] == -1).all()  # absent 12,11; not silently nearest captures
    assert context["history_tokens"][1, 2, 0, 0] == 3
    assert not context["history_valid"][0].any()


def test_replay_survives_ring_wrap_and_uses_current_encoder():
    h = RangeHistory(2, [1, 2], 0.1, "cpu")
    enc = ScalarEncoder()
    env = torch.arange(2)
    for t in range(3):
        x = images(t + 1)
        h.append(env, torch.full((2,), t), x, enc.encode_history(x))
    ids, age, _ = h.select(env, torch.full((2,), 3))
    for t in range(3, 9):
        x = images(t + 1)
        h.append(env, torch.full((2,), t), x, enc.encode_history(x))
    enc.scale = 7.0
    replay = h.replay(ids[None], age[None], enc, batch_size=1)
    torch.testing.assert_close(replay["history_tokens"][0, 0, :, 0, 0], torch.tensor([21.0, 14.0]))
    assert not replay["history_tokens"].requires_grad
    assert len(h.chunks) == 9
    h.prune()
    assert len(h.chunks) == 3


def test_history_metadata_packs_with_episode_sequences():
    storage = PerceptionStorage()
    env = torch.arange(2)
    for t in range(3):
        x = images(t)
        owned = storage.add_indexed(x, x, x, torch.zeros(2, 4), env, torch.tensor([0, t // 2]), torch.full((2,), t))
        owned["history_ids"] = torch.tensor([[t - 1], [10 + t]])
        owned["history_age"] = torch.full((2, 1), 0.1)
    batches = storage.sequence_batches(2, 2)
    assert sum(b.anchor_count for b in batches) == 6
    for batch in batches:
        torch.testing.assert_close(
            batch.history_ids[..., 0][batch.active & (batch.env_ids == 0)],
            (batch.capture_ids - 1)[batch.active & (batch.env_ids == 0)],
        )
        assert ((batch.episode_ids == batch.episode_ids[:1]) | ~batch.active).all()


def test_a0_model_contract_is_unchanged():
    original = _model_configuration({})
    explicit = _model_configuration({"variant": "A0", "history_encode_batch_size": 32})
    assert original == explicit and "variant" not in original


def test_actual_runner_history_crosses_rollouts_and_checkpoint():
    from copy import deepcopy

    from test_range_jepa_amp_ppo import TinyEnv, cfg

    from rsl_rl.runners import OnPolicyRunner

    config = cfg()
    config["algorithm"]["perception_cfg"].update(
        variant="B", history_frames=5, history_seconds=0.5, num_learning_epochs=1, history_encode_batch_size=8
    )
    runner = OnPolicyRunner(TinyEnv(), config, device="cpu")
    alg = runner.alg
    consumed = []
    original = alg.perception.auxiliary_loss

    def capture(*args, **kwargs):
        consumed.append(kwargs["history_valid"].any().item())
        assert not kwargs["history_tokens"].requires_grad
        return original(*args, **kwargs)

    alg.perception.auxiliary_loss = capture
    runner.learn(2)
    assert any(consumed)
    assert alg.perception_update_count == 2
    assert not alg._live_policy.features.requires_grad
    assert alg._live_policy.history.chunks
    state = deepcopy(alg.save())
    clone = OnPolicyRunner(TinyEnv(), config, device="cpu")
    clone.alg.load(state, None, True)
    assert clone.alg._live_policy.history is None
    for a, b in zip(alg.perception.parameters(), clone.alg.perception.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_held_capture_and_episode_change_do_not_leak_history():
    from test_range_jepa_amp_ppo import TinyEnv, cfg

    from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAAMPPPO

    config = cfg()
    config["algorithm"]["perception_cfg"].update(variant="B", history_frames=3, num_learning_epochs=1)
    env = TinyEnv()
    alg = RangeJEPAAMPPPO.construct_algorithm(env.get_observations(), env, config, "cpu")
    policy = alg._inference_policy
    obs = env.get_observations()
    first = policy.prepare(obs)["range_features"].clone()
    for _ in range(3):
        torch.testing.assert_close(policy.prepare(obs)["range_features"], first, rtol=0, atol=0)
    assert policy.history.next_chunk == 1
    obs["range_capture_id"][:] = 1
    policy.prepare(obs)
    assert policy.history.next_chunk == 2
    obs["range_episode_id"][0] += 1
    obs["range_capture_id"][0] = 0
    policy.prepare(obs)
    ids, _, _ = policy.history.select(torch.tensor([0, 1]), torch.tensor([1, 2]))
    assert ids[0, 0] == 8  # new chunk2 * batch4 + env0
    assert ids[0, 1] == -1
    assert ids[1, 0] == 5  # other env retained prior chunk1


def test_all_experiment_variants_complete_original_runner_updates():
    from test_range_jepa_amp_ppo import TinyEnv, cfg

    from rsl_rl.runners import OnPolicyRunner

    for variant in ("A1", "B", "C", "D", "E", "F"):
        config = cfg()
        config["algorithm"]["perception_cfg"].update(
            variant=variant, num_learning_epochs=1, history_seconds=1.0, history_encode_batch_size=16
        )
        if variant in ("E", "F"):
            config["actor"].update(
                class_name="rsl_rl.models.range_bev:RangeBEVActor",
                representation="height" if variant == "E" else "latent",
            )
        runner = OnPolicyRunner(TinyEnv(), config, device="cpu")
        runner.learn(2)
        assert runner.alg.perception_update_count == 2
        assert runner.alg.storage.observations["range_features"].shape[-1] == (4096 if variant in ("E", "F") else 512)


def test_replay_cache_is_bounded_exact_and_refreshed_each_pass():
    h = RangeHistory(2, [1, 2], 0.1, "cpu")

    class CountingEncoder(ScalarEncoder):
        rows = 0

        def encode_history(self, x):
            self.rows += x.shape[0]
            return super().encode_history(x)

    enc = CountingEncoder()
    env = torch.arange(2)
    for t in range(3):
        x = images(t + 1)
        h.append(env, torch.full((2,), t), x, enc.encode_history(x))
    ids, age, _ = h.select(env, torch.full((2,), 3))
    enc.rows = 0
    h.begin_replay_pass(1024)
    first = h.replay(ids, age, enc)
    assert enc.rows == 4
    second = h.replay(ids, age, enc)
    assert enc.rows == 4  # same selected raw rows were encoded exactly once
    torch.testing.assert_close(first["history_tokens"], second["history_tokens"], rtol=0, atol=0)
    assert h.replay_cache_bytes <= 1024
    h.end_replay_pass()
    assert h.replay_cache is None
    enc.scale = 2.0
    h.begin_replay_pass(0)
    fresh = h.replay(ids, age, enc)
    torch.testing.assert_close(fresh["history_tokens"], first["history_tokens"] * 2)
    assert h.replay_cache_bytes == 0
    assert enc.rows == 8
    h.end_replay_pass()
