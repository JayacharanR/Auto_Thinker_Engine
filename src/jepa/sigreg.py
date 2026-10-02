"""
SIGReg: Sketched Isotropic Gaussian Regularization (LeJEPA, Balestriero & LeCun 2025).

An alternative to the EMA teacher for preventing representation collapse.
Embeddings are pushed towards an isotropic Gaussian: they are projected onto
random unit directions, and each 1-D projection is compared with N(0, 1)
using the Epps–Pulley statistic, a weighted L2 distance between the
empirical and the Gaussian characteristic functions. A collapsed embedding
(all samples alike) is far from N(0, 1) along every direction, so the
statistic is large; Gaussian embeddings give a value near zero.
"""

from typing import Optional

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Args:
        num_slices: Number of random projection directions M.
        num_points: Integration points of the characteristic function on [-t_max, t_max].
        t_max: Integration range.
        max_samples: Embeddings used per call (random subset) to bound memory.
    """

    def __init__(
        self,
        num_slices: int = 256,
        num_points: int = 17,
        t_max: float = 5.0,
        max_samples: Optional[int] = 4096,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.max_samples = max_samples
        t = torch.linspace(-t_max, t_max, num_points)
        self.register_buffer("t", t, persistent=False)
        # Characteristic function of N(0, 1), also used as the integration weight.
        self.register_buffer("gauss_cf", torch.exp(-0.5 * t**2), persistent=False)

    def forward(self, z: torch.Tensor, step: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            z: (..., D) embeddings; all leading dimensions are treated as samples.
            step: Seed for the projection directions (fresh directions each step).

        Returns:
            Scalar: mean Epps–Pulley statistic over the projection directions.
        """
        z = z.reshape(-1, z.shape[-1]).float()
        if self.max_samples is not None and z.shape[0] > self.max_samples:
            z = z[torch.randperm(z.shape[0], device=z.device)[: self.max_samples]]
        n = z.shape[0]

        generator = torch.Generator(device=z.device)
        if step is not None:
            generator.manual_seed(int(step))
        else:
            generator.seed()
        directions = torch.randn(
            z.shape[1], self.num_slices, device=z.device, generator=generator
        )
        directions = directions / directions.norm(dim=0, keepdim=True)

        x_t = (z @ directions).unsqueeze(-1) * self.t  # (N, M, T)
        # Empirical characteristic function E[exp(i t x)] = E[cos] + i E[sin].
        real = torch.cos(x_t).mean(0)
        imag = torch.sin(x_t).mean(0)
        err = ((real - self.gauss_cf) ** 2 + imag**2) * self.gauss_cf  # (M, T)
        statistic = torch.trapezoid(err, self.t, dim=-1) * n  # (M,)
        return statistic.mean()
