"""DTI scalar / orientation metrics (WLS as reference, not GT)."""

from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> float:
    p, r = _masked(pred, ref, mask)
    return float(np.mean(np.abs(p - r)))


def rmse(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> float:
    p, r = _masked(pred, ref, mask)
    return float(np.sqrt(np.mean((p - r) ** 2)))


def _masked(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if mask is None:
        return pred.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    m = mask.astype(bool)
    return pred[m].astype(np.float64).ravel(), ref[m].astype(np.float64).ravel()


def tensor_to_scalars(D: np.ndarray) -> dict[str, np.ndarray]:
    """Eigen-decompose D [...,3,3] -> FA, MD, AD, RD, V1."""
    D = np.asarray(D, dtype=np.float64)
    # Symmetrize for numerical stability
    D = 0.5 * (D + np.swapaxes(D, -1, -2))
    evals, evecs = np.linalg.eigh(D)
    # eigh ascending -> reverse to descending
    evals = evals[..., ::-1]
    evecs = evecs[..., :, ::-1]
    # Clip tiny negatives from numerics
    evals = np.clip(evals, 0.0, None)

    l1, l2, l3 = evals[..., 0], evals[..., 1], evals[..., 2]
    md = (l1 + l2 + l3) / 3.0
    ad = l1
    rd = 0.5 * (l2 + l3)

    # FA
    num = np.sqrt(((l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2) * 1.5)
    den = np.sqrt(l1**2 + l2**2 + l3**2) + 1e-12
    fa = np.clip(num / den, 0.0, 1.0)
    v1 = evecs[..., :, 0]

    return {"FA": fa, "MD": md, "AD": ad, "RD": rd, "V1": v1, "evals": evals}


def angular_error_degrees(
    v_pred: np.ndarray,
    v_ref: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Unsigned angular error in degrees: arccos(|v_pred · v_ref|)."""
    if mask is not None:
        m = mask.astype(bool)
        vp = v_pred[m].astype(np.float64)
        vr = v_ref[m].astype(np.float64)
    else:
        vp = v_pred.reshape(-1, 3).astype(np.float64)
        vr = v_ref.reshape(-1, 3).astype(np.float64)

    vp_n = np.linalg.norm(vp, axis=-1, keepdims=True) + 1e-12
    vr_n = np.linalg.norm(vr, axis=-1, keepdims=True) + 1e-12
    vp = vp / vp_n
    vr = vr / vr_n
    dots = np.clip(np.abs(np.sum(vp * vr, axis=-1)), 0.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return {
        "mean_deg": float(np.mean(ang)),
        "median_deg": float(np.median(ang)),
    }


def compare_dti_maps(
    pred: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute FA/MD/AD/RD MAE/RMSE and V1 angular error vs reference."""
    out: dict[str, float] = {}
    for key in ("FA", "MD", "AD", "RD"):
        out[f"{key}_MAE"] = mae(pred[key], ref[key], mask)
        out[f"{key}_RMSE"] = rmse(pred[key], ref[key], mask)
    ang = angular_error_degrees(pred["V1"], ref["V1"], mask)
    out["V1_ang_mean_deg"] = ang["mean_deg"]
    out["V1_ang_median_deg"] = ang["median_deg"]
    return out
