"""Synthetic unit tests for the DTI forward model (Task 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.dti_forward import (  # noqa: E402
    EPS,
    cholesky_params_to_D,
    cholesky_params_to_L,
    dti_forward,
    dti_forward_from_cholesky,
    ensure_positive_s0,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _random_unit_g(n: int) -> torch.Tensor:
    g = torch.randn(n, 3)
    return g / (g.norm(dim=-1, keepdim=True) + 1e-12)


def test_forward_finite_and_positive() -> None:
    torch.manual_seed(0)
    n = 64

    # Realistic diffusivity scale (~1e-3 mm^2/s)
    chol = torch.randn(n, 6)
    D = cholesky_params_to_D(chol) * 1e-3
    S0 = ensure_positive_s0(torch.randn(n) + 2.0)
    g = _random_unit_g(n)
    b = torch.full((n,), 1000.0)

    S = dti_forward(S0, D, g, b)

    _assert(torch.isfinite(S).all().item(), "forward output must be finite")
    _assert((S > 0).all().item(), "signal must be positive")
    _assert((S0 > 0).all().item(), "S0 must be positive")


def test_D_symmetric_and_psd() -> None:
    torch.manual_seed(1)
    n = 32
    chol = torch.randn(n, 6)
    L = cholesky_params_to_L(chol)
    D = cholesky_params_to_D(chol)

    sym_err = (D - D.transpose(-1, -2)).abs().max().item()
    _assert(sym_err < 1e-6, f"D not symmetric, max err={sym_err}")

    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    _assert((diag > 0).all().item(), "Cholesky diagonal must be positive")

    evals = torch.linalg.eigvalsh(D)
    _assert((evals >= -1e-6).all().item(), f"D not PSD, min eig={evals.min().item()}")


def test_matches_analytic_formula() -> None:
    torch.manual_seed(2)
    n = 16
    chol = torch.randn(n, 6)
    D = cholesky_params_to_D(chol) * 1e-3
    S0 = ensure_positive_s0(torch.randn(n))
    g = _random_unit_g(n)
    b = torch.rand(n) * 2000.0

    S = dti_forward(S0, D, g, b)
    q = torch.einsum("ni,nij,nj->n", g, D, g)
    # Mirror the same clamp used in the implementation for fair comparison
    exponent = torch.clamp(-b * q, min=-60.0, max=60.0)
    S_ref = S0 * torch.exp(exponent)
    err = (S - S_ref).abs().max().item()
    _assert(err < 1e-6, f"forward mismatch vs analytic, max err={err}")


def test_b0_recovers_s0() -> None:
    torch.manual_seed(3)
    n = 8
    chol = torch.randn(n, 6)
    D = cholesky_params_to_D(chol) * 1e-3
    S0 = ensure_positive_s0(torch.randn(n) + 1.0)
    g = _random_unit_g(n)
    b = torch.zeros(n)

    S = dti_forward(S0, D, g, b)
    err = (S - S0).abs().max().item()
    _assert(err < 1e-6, f"b=0 should give S=S0, max err={err}")


def test_from_cholesky_path() -> None:
    torch.manual_seed(4)
    n = 12
    raw_s0 = torch.randn(n)
    # Scale Cholesky so D is in a physical range
    chol = torch.randn(n, 6) * 0.03
    g = _random_unit_g(n)
    b = torch.full((n,), 1000.0)

    S0, D, S = dti_forward_from_cholesky(raw_s0, chol, g, b)
    _assert((S0 >= EPS).all().item(), "S0 must be >= eps")
    _assert(torch.isfinite(S).all().item(), "S must be finite")
    _assert((S > 0).all().item(), "S must be positive")
    evals = torch.linalg.eigvalsh(D)
    _assert((evals >= -1e-6).all().item(), "D from cholesky path must be PSD")


def test_extreme_attenuation_stays_positive() -> None:
    """Even for large b * g^T D g, clamped forward must stay finite and > 0."""
    torch.manual_seed(5)
    n = 8
    D = cholesky_params_to_D(torch.randn(n, 6))  # intentionally large
    S0 = ensure_positive_s0(torch.ones(n))
    g = _random_unit_g(n)
    b = torch.full((n,), 1000.0)
    S = dti_forward(S0, D, g, b)
    _assert(torch.isfinite(S).all().item(), "extreme case must be finite")
    _assert((S > 0).all().item(), "extreme case must stay positive after clamp")


def main() -> None:
    tests = [
        test_forward_finite_and_positive,
        test_D_symmetric_and_psd,
        test_matches_analytic_formula,
        test_b0_recovers_s0,
        test_from_cholesky_path,
        test_extreme_attenuation_stays_positive,
    ]
    print("=" * 60)
    print("Task 1 — DTI forward synthetic tests")
    print("=" * 60)
    failed = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {exc}")
    print("-" * 60)
    if failed == 0:
        print("RESULT: PASS")
        sys.exit(0)
    print(f"RESULT: FAIL ({failed}/{len(tests)} failed)")
    sys.exit(1)


if __name__ == "__main__":
    main()
