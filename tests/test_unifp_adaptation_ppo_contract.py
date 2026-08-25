"""Contracts for reusable UniFP algorithm naming and validation."""

from __future__ import annotations

from rsl_rl.algorithms.unifp_adaptation_ppo import _adaptation_part_names


def test_legacy_adaptation_names_remain_backward_compatible():
    assert _adaptation_part_names(2) == (
        "adaptation_base_velocity",
        "adaptation_gripper_pos",
    )
    assert _adaptation_part_names(3) == (
        "adaptation_base_velocity",
        "adaptation_gripper_pos",
        "adaptation_gripper_orn",
    )
    assert _adaptation_part_names(4)[2:] == (
        "adaptation_force_ee",
        "adaptation_force_base",
    )
