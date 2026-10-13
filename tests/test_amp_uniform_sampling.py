"""Distribution and transition contracts, without launching Isaac Sim."""

import json

import numpy as np
import pytest
import torch

from rsl_rl.utils.motion_loader import AMPLoader


def make_motions(tmp_path, lengths=(11, 51, 101), fps=(50, 50, 50), mode="continuous", **kwargs):
    paths = []
    for i, (length, rate) in enumerate(zip(lengths, fps)):
        path = tmp_path / f"clip_{i}.json"
        frames = np.zeros((length, 70))
        frames[:, 0] = i
        frames[:, 1] = np.arange(length) / rate
        path.write_text(json.dumps({"Frames": frames.tolist(), "FrameDuration": 1 / rate,
                                    "MotionWeight": [0, 10000, .1][i]}))
        paths.append(str(path))
    categories = kwargs.pop("motion_categories", {"stairs": paths[:1], "walk": paths[1:]})
    return AMPLoader(device="cpu", motion_files=paths, time_between_frames=.02,
                     expert_sampling="uniform", expert_sampling_mode=mode,
                     motion_categories=categories, **kwargs), paths


@pytest.mark.parametrize("mode", ["continuous", "adjacent"])
def test_categories_files_and_time_are_uniform_even_with_discriminator_decimation(tmp_path, mode):
    np.random.seed(41)
    loader, _ = make_motions(tmp_path, mode=mode)
    # Two categories, one with a single zero-weight file, another with two
    # files of different lengths/weights. Their intended probabilities are .5/.25/.25.
    np.testing.assert_allclose(loader.trajectory_weights, [.5, .25, .25])
    accepted = []
    for _ in range(4):  # New generators must not reset any file cursor.
        for batch_id, (states, next_states) in enumerate(loader.feed_forward_generator(20, 1024)):
            if batch_id % 4 == 0:
                accepted.append(states)
                torch.testing.assert_close(states[:, 0], next_states[:, 0])
                torch.testing.assert_close(next_states[:, 1] - states[:, 1],
                                           torch.full((1024,), .02), atol=3e-7, rtol=1e-5)
    states = torch.cat(accepted).numpy()
    ids = states[:, 0].astype(int)
    np.testing.assert_allclose(np.bincount(ids) / len(ids), [.5, .25, .25], atol=.012)
    for clip_id in range(3):
        times = states[ids == clip_id, 1]
        if mode == "continuous":
            bins = np.histogram(times / (loader.trajectory_lens[clip_id] - .02), bins=5, range=(0, 1))[0]
            assert not np.any(times == 0)  # Old clamping put extra mass here.
        else:
            frames = np.rint(times * 50).astype(int)
            valid_count = int(loader.trajectory_num_frames[clip_id]) - 1
            assert frames.min() == 0 and frames.max() == valid_count - 1
            bins = np.bincount(frames * 5 // valid_count, minlength=5)
        np.testing.assert_allclose(bins / bins.sum(), np.full(5, .2), atol=.025)


def test_continuous_mixed_fps_keeps_physical_time_interval(tmp_path):
    loader, _ = make_motions(tmp_path, fps=(30, 50, 100))
    states, next_states = next(loader.feed_forward_generator(1, 4096))
    torch.testing.assert_close(next_states[:, 1] - states[:, 1], torch.full((4096,), .02), atol=2e-7, rtol=1e-5)


def test_adjacent_rejects_mixed_fps_with_filename(tmp_path):
    with pytest.raises(ValueError, match=r"clip_0: frames=11, fps=30"):
        make_motions(tmp_path, fps=(30, 50, 50), mode="adjacent")


def test_rejects_short_clips(tmp_path):
    with pytest.raises(ValueError, match="Invalid AMP transitions"):
        make_motions(tmp_path, lengths=(1, 51, 101))


def test_uniform_files_without_categories(tmp_path):
    loader, _ = make_motions(tmp_path, motion_categories=None)
    np.testing.assert_allclose(loader.trajectory_weights, np.full(3, 1 / 3))


def test_missing_and_duplicate_category_assignments_fail(tmp_path):
    loader, paths = make_motions(tmp_path)
    with pytest.raises(ValueError, match="unassigned"):
        loader._uniform_category_weights({"incomplete": paths[:1]})
    with pytest.raises(ValueError, match="Duplicate"):
        loader._uniform_category_weights({"a": paths, "b": paths[:1]})
    with pytest.raises(ValueError, match="not be empty"):
        loader._uniform_category_weights({})


def test_weighted_sampling_preserves_metadata_distribution(tmp_path):
    _, paths = make_motions(tmp_path)
    loader = AMPLoader(device="cpu", motion_files=paths, time_between_frames=.02, expert_sampling="weighted")
    np.testing.assert_allclose(loader.trajectory_weights, np.asarray([0, 10000, .1]) / 10000.1)


def test_packed_clips_keep_file_mass_and_do_not_cross_boundaries():
    loader = object.__new__(AMPLoader)
    loader.trajectory_files = ["/tmp/a.npz", "/tmp/a.npz", "/tmp/b.npz"]
    loader.trajectory_num_frames = np.asarray([3, 7, 4])
    loader.expert_sampling_mode = "adjacent"
    loader.time_between_frames = .02
    np.testing.assert_allclose(loader._uniform_category_weights(None), [.125, .375, .5])
    loader.device = "cpu"
    loader._amp_obs_dim = 1
    loader.trajectories_full = [torch.arange(n).float().unsqueeze(1) + offset
                                for n, offset in [(3, 0), (7, 100), (4, 200)]]
    states, next_states = loader.get_adjacent_frame_batch(np.tile([0, 1, 2], 100))
    torch.testing.assert_close(next_states - states, torch.ones_like(states))


def test_ppo_only_draws_experts_for_actual_discriminator_updates(tmp_path, monkeypatch):
    from tensordict import TensorDict
    from rsl_rl.algorithms.amp_ppo import AMPPPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage

    _, paths = make_motions(tmp_path)
    obs = TensorDict({"policy": torch.randn(4, 3), "amp": torch.randn(4, 70)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(obs, groups, "actor", 2, hidden_dims=[8],
                     distribution_cfg={"class_name": "rsl_rl.modules.distribution.GaussianDistribution"})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[8])
    storage = RolloutStorage("rl", 4, 4, obs, (2,), "cpu")
    ppo = AMPPPO(actor, critic, storage, device="cpu", num_learning_epochs=5, num_mini_batches=4,
                 amp_cfg={"motion_files": paths, "expert_sampling": "uniform",
                          "discriminator_hidden_dims": [8, 8], "discriminator_update_decimation": 4})
    samples = []
    original = ppo.amp_expert_data.feed_forward_generator

    def counted_generator(*args):
        for pair in original(*args):
            samples.append(pair[0].shape[0])
            yield pair

    monkeypatch.setattr(ppo.amp_expert_data, "feed_forward_generator", counted_generator)
    for _ in range(2):
        with torch.no_grad():
            for _ in range(4):
                ppo.act(obs)
                next_obs = obs.clone()
                next_obs["amp"] += .01
                ppo.process_env_step(next_obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
                obs = next_obs
            ppo.compute_returns(obs)
        losses = ppo.update()
        assert np.isfinite(losses["amp"])
    assert ppo.disc_update_counter == 40
    assert len(samples) == 10  # 5 real D updates per 20 PPO minibatches, not 20 draws.
