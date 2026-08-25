"""Numerical contracts for Gaussian policy distributions."""

import torch

from rsl_rl.modules.distribution import GaussianDistribution


def test_scalar_std_is_projected_positive_before_sampling():
    distribution = GaussianDistribution(output_dim=3, init_std=0.5, std_type="scalar")
    with torch.no_grad():
        distribution.std_param.copy_(torch.tensor([-0.2, 0.0, 0.3]))

    distribution.update(torch.zeros(4, 3))

    assert torch.all(distribution.std > 0.0)
    assert torch.isfinite(distribution.sample()).all()
    assert torch.allclose(distribution.std[0], torch.tensor([1.0e-6, 1.0e-6, 0.3]))
