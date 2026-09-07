"""Classical DTI forward model (independent implementation).

Signal model
------------
    S(x, g, b) = S0(x) * exp(-b * g^T D(x) g)

Diffusion tensor parameterization
---------------------------------
Predict unconstrained Cholesky parameters, then form

    L = [[l11, 0,   0  ],
         [l21, l22, 0  ],
         [l31, l32, l33]]

with l_ii = softplus(a_ii) + eps > 0, and

    D = L L^T  ⪰  0.
"""

from __future__ import annotations

import torch
from torch import Tensor


EPS = 1e-6
# Clamp exponent to avoid float underflow/overflow while keeping S > 0.
_EXP_CLAMP = 60.0


def cholesky_params_to_L(chol_params: Tensor, eps: float = EPS) -> Tensor:
    """Build lower-triangular L from 6 unconstrained parameters.

    Last dimension order: [a11, a21, a22, a31, a32, a33].

    Args:
        chol_params: [..., 6]
        eps: positive offset on diagonal entries

    Returns:
        L: [..., 3, 3]
    """
    if chol_params.shape[-1] != 6:
        raise ValueError(f"Expected last dim 6, got {chol_params.shape[-1]}")

    a11, a21, a22, a31, a32, a33 = torch.unbind(chol_params, dim=-1)

    l11 = torch.nn.functional.softplus(a11) + eps
    l22 = torch.nn.functional.softplus(a22) + eps
    l33 = torch.nn.functional.softplus(a33) + eps
    l21, l31, l32 = a21, a31, a32

    zeros = torch.zeros_like(l11)
    row0 = torch.stack([l11, zeros, zeros], dim=-1)
    row1 = torch.stack([l21, l22, zeros], dim=-1)
    row2 = torch.stack([l31, l32, l33], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def L_to_D(L: Tensor) -> Tensor:
    """D = L @ L^T with shape [..., 3, 3]."""
    return L @ L.transpose(-1, -2)


def cholesky_params_to_D(chol_params: Tensor, eps: float = EPS) -> Tensor:
    """Map 6 Cholesky parameters to a symmetric PSD tensor D."""
    return L_to_D(cholesky_params_to_L(chol_params, eps=eps))


def ensure_positive_s0(raw_s0: Tensor, eps: float = EPS) -> Tensor:
    """S0 = softplus(raw_s0) + eps > 0."""
    return torch.nn.functional.softplus(raw_s0) + eps


def quadratic_form(D: Tensor, g: Tensor) -> Tensor:
    """Compute g^T D g.

    Args:
        D: [N, 3, 3]
        g: [N, 3]

    Returns:
        q: [N]
    """
    Dg = torch.einsum("nij,nj->ni", D, g)
    return torch.einsum("ni,ni->n", g, Dg)


def dti_forward(
    S0: Tensor,
    D: Tensor,
    g: Tensor,
    b: Tensor,
) -> Tensor:
    """Evaluate S = S0 * exp(-b * g^T D g).

    Args:
        S0: [N]  (expected > 0)
        D:  [N, 3, 3]
        g:  [N, 3]
        b:  [N]

    Returns:
        S_pred: [N]
    """
    if S0.ndim != 1:
        raise ValueError(f"S0 must be [N], got {tuple(S0.shape)}")
    if D.ndim != 3 or D.shape[-2:] != (3, 3):
        raise ValueError(f"D must be [N, 3, 3], got {tuple(D.shape)}")
    if g.ndim != 2 or g.shape[-1] != 3:
        raise ValueError(f"g must be [N, 3], got {tuple(g.shape)}")
    if b.ndim != 1:
        raise ValueError(f"b must be [N], got {tuple(b.shape)}")

    n = S0.shape[0]
    if D.shape[0] != n or g.shape[0] != n or b.shape[0] != n:
        raise ValueError(
            f"Batch mismatch: S0={n}, D={D.shape[0]}, g={g.shape[0]}, b={b.shape[0]}"
        )

    q = quadratic_form(D, g)
    exponent = torch.clamp(-b * q, min=-_EXP_CLAMP, max=_EXP_CLAMP)
    return S0 * torch.exp(exponent)


def dti_forward_from_cholesky(
    raw_s0: Tensor,
    chol_params: Tensor,
    g: Tensor,
    b: Tensor,
    eps: float = EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map unconstrained outputs to (S0, D, S_pred)."""
    S0 = ensure_positive_s0(raw_s0, eps=eps)
    D = cholesky_params_to_D(chol_params, eps=eps)
    S_pred = dti_forward(S0, D, g, b)
    return S0, D, S_pred
