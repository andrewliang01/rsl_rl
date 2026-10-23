"""Verify actual runner exposes geometry errors as diagnostics, not loss terms."""

import math

from test_range_jepa_amp_ppo import TinyEnv, cfg

from rsl_rl.runners import OnPolicyRunner


def run_metrics(variant):
    config = cfg()
    config["algorithm"]["perception_cfg"].update(variant=variant, num_learning_epochs=1)
    if variant == "F":
        config["actor"].update(class_name="rsl_rl.models.range_bev:RangeBEVActor", representation="latent")
    runner = OnPolicyRunner(TinyEnv(), config, device="cpu")
    measured = []
    original = runner.alg._update_perception

    def record():
        metrics = original()
        measured.append(metrics)
        return metrics

    runner.alg._update_perception = record
    runner.learn(1)
    return measured[-1]


def test_depth_metrics_reach_actual_training_logs():
    result = run_metrics("A1")
    for name in ("depth_mse_m2", "near_depth_mse_m2"):
        assert math.isfinite(result["range_jepa/" + name])
        assert result["range_jepa/" + name] >= 0
    assert "range_jepa/surface_height_mse_m2" not in result


def test_surface_metrics_have_correct_name():
    result = run_metrics("F")
    assert math.isfinite(result["range_jepa/surface_height_mse_m2"])
    assert "range_jepa/depth_mse_m2" not in result


def test_a0_retains_original_metric_contract():
    result = run_metrics("A0")
    assert not any(name.endswith("_mse_m2") for name in result)
    assert result["range_jepa/optimizer_steps"] == 1


def test_profiled_history_runs_one_encoder_and_preserves_features():
    import copy

    import torch
    from tensordict import TensorDict

    from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAInferencePolicy
    from rsl_rl.models.range_experiments import RangeExperiment
    from rsl_rl.utils.range_online_profiler import RangeOnlineProfiler

    model = RangeExperiment(image_height=8, image_width=16, variant="C")
    plain = RangeJEPAInferencePolicy(torch.nn.Identity(), model)
    measured = RangeJEPAInferencePolicy(torch.nn.Identity(), copy.deepcopy(model))
    calls = []
    hook = measured.perception.encoder.register_forward_hook(lambda *args: calls.append(1))
    profile = RangeOnlineProfiler(True, "cpu")
    for capture in range(3):
        x = torch.ones(2, 2, 8, 16)
        x[:, 0] = 0.05 + torch.rand(2, 8, 16) * 0.1
        obs = TensorDict(
            {
                "range_obs": x,
                "range_capture_id": torch.full((2, 1), capture, dtype=torch.long),
                "range_episode_id": torch.zeros(2, 1, dtype=torch.long),
            },
            batch_size=[2],
        )
        expected = plain.prepare(obs)["range_features"]
        actual = measured.prepare(obs, profile=profile)["range_features"]
        torch.testing.assert_close(actual, expected)
    hook.remove()
    report = profile.finish()
    assert report["online_cnn_calls"] == report["memory_attention_calls"] == 3
    assert len(calls) == 3  # No second online encoding for history insertion.
    assert measured.history.next_chunk == 3


def test_strict_checkpoint_rejects_semantic_history_change_before_loading():
    import copy

    import torch

    config = cfg()
    config["algorithm"]["perception_cfg"].update(variant="C", history_seconds=5.0)
    original = OnPolicyRunner(TinyEnv(), config, device="cpu")
    state = copy.deepcopy(original.alg.save())
    config["algorithm"]["perception_cfg"]["history_seconds"] = 10.0
    changed = OnPolicyRunner(TinyEnv(), config, device="cpu")
    before = [p.clone() for p in changed.alg.perception.parameters()]
    try:
        changed.alg.load(state, None, True)
    except ValueError as error:
        assert "configuration differs" in str(error)
    else:
        raise AssertionError("Same-shaped C5 checkpoint must not silently resume as C10")
    for param, saved in zip(changed.alg.perception.parameters(), before):
        torch.testing.assert_close(param, saved)


if __name__ == "__main__":
    import torch

    torch.set_num_threads(1)
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
