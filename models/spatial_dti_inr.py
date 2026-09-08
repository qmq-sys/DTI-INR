"""Spatial DTI-INR (Model A): x -> {S0(x), D(x)} with D = L L^T."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.positional_encoding import FourierPositionalEncoding
from physics.dti_forward import EPS, cholesky_params_to_D, ensure_positive_s0


class SpatialDTIINR(nn.Module):
    """Implicit spatial field for DTI parameters.

    Diffusion directions are NOT network inputs; they enter only through
    the DTI physics layer when predicting signals.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_frequencies: int = 8,
        activation: str = "gelu",
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.pe = FourierPositionalEncoding(
            in_dim=3,
            num_frequencies=num_frequencies,
            include_input=include_input,
        )

        layers: list[nn.Module] = []
        in_dim = self.pe.out_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            if activation.lower() == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        self.mlp = nn.Sequential(*layers)

        self.s0_head = nn.Linear(hidden_dim, 1)
        self.d_head = nn.Linear(hidden_dim, 6)
        self._init_heads()

    def _init_heads(self) -> None:
        # Bias S0 toward ~softplus(0.5) after signal normalization (~0.97).
        nn.init.zeros_(self.s0_head.weight)
        nn.init.constant_(self.s0_head.bias, 0.5)
        # Isotropic MD ≈ 0.95 in b-scaled units (~0.95e-3 mm^2/s).
        nn.init.zeros_(self.d_head.weight)
        with torch.no_grad():
            self.d_head.bias.zero_()
            # [a11, a21, a22, a31, a32, a33] — diagonals softplus(0.5)+eps
            self.d_head.bias[0] = 0.5
            self.d_head.bias[2] = 0.5
            self.d_head.bias[5] = 0.5

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: [N, 3] normalized coordinates in [-1, 1]

        Returns:
            S0: [N]
            D:  [N, 3, 3]  (diffusivity in training units; see b_scale)
        """
        h = self.mlp(self.pe(x))
        raw_s0 = self.s0_head(h).squeeze(-1)
        chol = self.d_head(h)
        S0 = ensure_positive_s0(raw_s0, eps=EPS)
        D = cholesky_params_to_D(chol, eps=EPS)
        return S0, D
