"""Weighted least-squares (WLS) DTI baseline via DIPY.

Used only for:
  1) baseline comparison
  2) reference evaluation maps

Do NOT use WLS outputs as supervision for INR training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy, mean_diffusivity


def _axial_diffusivity(evals: np.ndarray) -> np.ndarray:
    return evals[..., 0]


def _radial_diffusivity(evals: np.ndarray) -> np.ndarray:
    return 0.5 * (evals[..., 1] + evals[..., 2])


def _principal_eigenvectors(evecs: np.ndarray) -> np.ndarray:
    """V1 with shape [..., 3]. DIPY sorts eigenvalues descending."""
    return evecs[..., :, 0]


def fit_wls_dti(
    dwi: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fit a WLS diffusion tensor model.

    Args:
        dwi:   [X, Y, Z, N] or [V, N]
        bvals: [N]
        bvecs: [N, 3] or [3, N]
        mask:  optional boolean mask over spatial dims

    Returns:
        dict with keys: D, FA, MD, AD, RD, V1, S0, evals, evecs
    """
    dwi = np.asarray(dwi, dtype=np.float64)
    bvals = np.asarray(bvals, dtype=np.float64).ravel()
    bvecs = np.asarray(bvecs, dtype=np.float64)

    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        raise ValueError(f"bvecs must be [N, 3] or [3, N], got {bvecs.shape}")
    if bvecs.shape[0] != bvals.shape[0]:
        raise ValueError(
            f"bvals/bvecs length mismatch: {bvals.shape[0]} vs {bvecs.shape[0]}"
        )
    if dwi.shape[-1] != bvals.shape[0]:
        raise ValueError(
            f"dwi last dim {dwi.shape[-1]} != number of volumes {bvals.shape[0]}"
        )

    gtab = gradient_table(bvals, bvecs=bvecs)
    # return_S0_hat=True is required in DIPY >=1.x to populate S0
    model = TensorModel(gtab, fit_method="WLS", return_S0_hat=True)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != dwi.shape[:-1]:
            raise ValueError(
                f"mask shape {mask.shape} != dwi spatial shape {dwi.shape[:-1]}"
            )
        fit = model.fit(dwi, mask=mask)
    else:
        fit = model.fit(dwi)

    D = np.asarray(fit.quadratic_form, dtype=np.float64)
    evals = np.asarray(fit.evals, dtype=np.float64)
    evecs = np.asarray(fit.evecs, dtype=np.float64)

    if fit.model_S0 is not None:
        S0 = np.asarray(fit.model_S0, dtype=np.float64)
    elif fit.S0_hat is not None:
        S0 = np.asarray(fit.S0_hat, dtype=np.float64)
    else:
        raise RuntimeError(
            "WLS fit did not return S0. Ensure TensorModel(return_S0_hat=True)."
        )

    FA = np.asarray(fractional_anisotropy(evals), dtype=np.float64)
    MD = np.asarray(mean_diffusivity(evals), dtype=np.float64)
    AD = np.asarray(_axial_diffusivity(evals), dtype=np.float64)
    RD = np.asarray(_radial_diffusivity(evals), dtype=np.float64)
    V1 = np.asarray(_principal_eigenvectors(evecs), dtype=np.float64)

    for arr in (FA, MD, AD, RD, S0):
        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr[bad] = 0.0
    bad_v1 = ~np.isfinite(V1).all(axis=-1)
    if np.any(bad_v1):
        V1[bad_v1] = 0.0
    bad_d = ~np.isfinite(D).all(axis=(-1, -2))
    if np.any(bad_d):
        D[bad_d] = 0.0

    return {
        "D": D,
        "FA": FA,
        "MD": MD,
        "AD": AD,
        "RD": RD,
        "V1": V1,
        "S0": S0,
        "evals": evals,
        "evecs": evecs,
    }


def fit_wls_dti_summary(result: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compact numeric summary for logging / JSON."""
    fa = result["FA"]
    md = result["MD"]
    return {
        "fa_mean": float(np.nanmean(fa)),
        "fa_std": float(np.nanstd(fa)),
        "md_mean": float(np.nanmean(md)),
        "md_std": float(np.nanstd(md)),
        "s0_mean": float(np.nanmean(result["S0"])),
        "shape_FA": list(fa.shape),
    }
