"""Synthetic unit tests for the WLS-DTI baseline (Task 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.wls_dti import fit_wls_dti, fit_wls_dti_summary  # noqa: E402
from physics.dti_forward import cholesky_params_to_D, dti_forward, ensure_positive_s0  # noqa: E402

import torch  # noqa: E402


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _make_synthetic_volume(
    nx: int = 4,
    ny: int = 4,
    nz: int = 2,
    n_dir: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a small synthetic DWI volume from known PSD tensors."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # bvecs on sphere + one b0
    bvecs = rng.normal(size=(n_dir, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True) + 1e-12
    bvals = np.full(n_dir, 1000.0, dtype=np.float64)
    # prepend a b0
    bvecs = np.vstack([np.array([[0.0, 0.0, 0.0]]), bvecs])
    bvals = np.concatenate([[0.0], bvals])
    n_vol = bvals.shape[0]

    n_vox = nx * ny * nz
    chol = torch.randn(n_vox, 6)
    # Scale so diffusivities are in a realistic ballpark (~1e-3 mm^2/s)
    D = cholesky_params_to_D(chol) * 1e-3
    S0 = ensure_positive_s0(torch.randn(n_vox) + 3.0) * 100.0

    signals = []
    for i in range(n_vol):
        g = torch.from_numpy(np.broadcast_to(bvecs[i], (n_vox, 3)).copy()).float()
        # Avoid zero-norm at b0: use a dummy unit vector; b=0 makes S=S0 anyway
        norms = g.norm(dim=-1, keepdim=True)
        g = torch.where(norms > 0, g / (norms + 1e-12), torch.tensor([0.0, 0.0, 1.0]))
        b = torch.full((n_vox,), float(bvals[i]))
        S = dti_forward(S0, D.float(), g, b)
        signals.append(S.numpy())
    dwi_flat = np.stack(signals, axis=-1)  # [V, N]
    dwi = dwi_flat.reshape(nx, ny, nz, n_vol)

    # Mild Rician-like noise (Gaussian approx for high SNR)
    noise = rng.normal(0.0, 1.0, size=dwi.shape)
    dwi = np.clip(dwi + noise, 1e-3, None)

    gt_D = D.numpy().reshape(nx, ny, nz, 3, 3)
    return dwi, bvals, bvecs, gt_D


def test_wls_runs_and_shapes() -> None:
    dwi, bvals, bvecs, _ = _make_synthetic_volume()
    mask = np.ones(dwi.shape[:3], dtype=bool)
    out = fit_wls_dti(dwi, bvals, bvecs, mask=mask)

    expected_keys = {"D", "FA", "MD", "AD", "RD", "V1", "S0", "evals", "evecs"}
    _assert(expected_keys.issubset(out.keys()), f"missing keys: {expected_keys - out.keys()}")

    spatial = dwi.shape[:3]
    _assert(out["FA"].shape == spatial, f"FA shape {out['FA'].shape}")
    _assert(out["MD"].shape == spatial, f"MD shape {out['MD'].shape}")
    _assert(out["AD"].shape == spatial, f"AD shape {out['AD'].shape}")
    _assert(out["RD"].shape == spatial, f"RD shape {out['RD'].shape}")
    _assert(out["S0"].shape == spatial, f"S0 shape {out['S0'].shape}")
    _assert(out["V1"].shape == spatial + (3,), f"V1 shape {out['V1'].shape}")
    _assert(out["D"].shape == spatial + (3, 3), f"D shape {out['D'].shape}")

    _assert(np.isfinite(out["FA"]).all(), "FA must be finite")
    _assert(np.isfinite(out["MD"]).all(), "MD must be finite")
    _assert((out["S0"] >= 0).all(), "S0 must be non-negative")
    _assert((out["FA"] >= -1e-6).all() and (out["FA"] <= 1.0 + 1e-6).all(), "FA in [0,1]")


def test_wls_recovers_tensor_approximately() -> None:
    dwi, bvals, bvecs, gt_D = _make_synthetic_volume(seed=7)
    out = fit_wls_dti(dwi, bvals, bvecs)

    # Compare tensor Frobenius error (relative)
    pred = out["D"]
    num = np.linalg.norm(pred - gt_D, axis=(-2, -1))
    den = np.linalg.norm(gt_D, axis=(-2, -1)) + 1e-12
    rel = num / den
    mean_rel = float(np.mean(rel))
    # With 30 dirs + noise, WLS should roughly recover the tensor field
    _assert(mean_rel < 0.35, f"mean relative D error too high: {mean_rel:.4f}")

    summary = fit_wls_dti_summary(out)
    _assert("fa_mean" in summary, "summary missing fa_mean")
    _assert(summary["fa_mean"] >= 0.0, "fa_mean should be >= 0")


def test_wls_flat_voxel_input() -> None:
    dwi, bvals, bvecs, _ = _make_synthetic_volume(nx=3, ny=2, nz=1, n_dir=20)
    flat = dwi.reshape(-1, dwi.shape[-1])
    out = fit_wls_dti(flat, bvals, bvecs)
    _assert(out["FA"].shape == (flat.shape[0],), f"flat FA shape {out['FA'].shape}")
    _assert(out["D"].shape == (flat.shape[0], 3, 3), f"flat D shape {out['D'].shape}")


def main() -> None:
    tests = [
        test_wls_runs_and_shapes,
        test_wls_recovers_tensor_approximately,
        test_wls_flat_voxel_input,
    ]
    print("=" * 60)
    print("Task 1 — WLS-DTI baseline synthetic tests")
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
