"""Fourier / NeRF-style positional encoding for spatial coordinates."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FourierPositionalEncoding(nn.Module):
    """Map x in R^{in_dim} to Fourier features.

    gamma(x) = [sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^{L-1} pi x), cos(2^{L-1} pi x)]
    optionally concatenated with the raw coordinates.
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_frequencies: int = 8,
        include_input: bool = True,
        log_sampling: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (num_frequencies - 1), num_frequencies)
        # Register as buffer so it moves with .to(device)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def out_dim(self) -> int:
        enc = self.in_dim * self.num_frequencies * 2
        return enc + self.in_dim if self.include_input else enc

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., in_dim] coordinates (expected roughly in [-1, 1])
        Returns:
            encoded: [..., out_dim]
        """
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dim {self.in_dim}, got {x.shape[-1]}")

        # [..., F, in_dim]
        xb = x.unsqueeze(-2) * self.freq_bands.view(*([1] * (x.ndim - 1)), -1, 1) * math.pi
        sin = torch.sin(xb)
        cos = torch.cos(xb)
        enc = torch.cat([sin, cos], dim=-1).reshape(*x.shape[:-1], -1)
        if self.include_input:
            enc = torch.cat([x, enc], dim=-1)
        return enc
