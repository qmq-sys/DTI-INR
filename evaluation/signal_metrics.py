"""Signal-level auxiliary metrics."""

from __future__ import annotations

import numpy as np


def signal_mse(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean((pred - obs) ** 2))


def signal_mae(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - obs)))


def signal_psnr(pred: np.ndarray, obs: np.ndarray, peak: float | None = None) -> float:
    mse = signal_mse(pred, obs)
    if mse <= 0:
        return float("inf")
    if peak is None:
        peak = float(np.max(obs)) if np.max(obs) > 0 else 1.0
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(mse))


def signal_nrmse(pred: np.ndarray, obs: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized RMSE: RMSE / RMS(obs)."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    denom = float(np.sqrt(np.mean(obs**2)))
    return rmse / max(denom, eps)
