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
