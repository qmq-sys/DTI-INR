"""Comparison figure for WLS vs INR DTI maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _pick_axial_slice(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(0, 1))
    return int(np.argmax(counts))


def _v1_rgb(v1: np.ndarray, fa: np.ndarray) -> np.ndarray:
    rgb = np.abs(v1)
    rgb = rgb / (np.max(rgb) + 1e-8)
    return np.clip(rgb * fa[..., None], 0.0, 1.0)


def save_comparison_figure(
    wls: dict[str, np.ndarray],
    inr: dict[str, np.ndarray],
    mask: np.ndarray,
    out_path: str | Path,
    slice_z: int | None = None,
    title: str = "WLS (reference) vs Spatial DTI-INR",
) -> int:
    """Save multi-row comparison PNG. Returns chosen axial slice index."""
    if slice_z is None:
        slice_z = _pick_axial_slice(mask)

    msl = mask[:, :, slice_z]
    fig, axes = plt.subplots(5, 3, figsize=(10, 14))
    fig.suptitle(
        f"{title}\nWLS-DTI is used as a reference, not as training supervision.\n"
        f"axial slice z={slice_z}",
        fontsize=11,
    )

    scalar_keys = ["FA", "MD", "AD", "RD"]
    for r, key in enumerate(scalar_keys):
        w = np.where(msl, wls[key][:, :, slice_z], np.nan)
        p = np.where(msl, inr[key][:, :, slice_z], np.nan)
        err = np.where(msl, np.abs(p - w), np.nan)
        vmax = float(np.nanpercentile([w[msl], p[msl]], 99)) if msl.any() else 1.0
        emax = float(np.nanpercentile(err[msl], 99)) if msl.any() else 1.0
        panels = [
            (f"WLS {key}", w, 0.0, max(vmax, 1e-8), "gray"),
            (f"INR {key}", p, 0.0, max(vmax, 1e-8), "gray"),
            (f"|Error| {key}", err, 0.0, max(emax, 1e-8), "magma"),
        ]
        for c, (lab, img, vmin, vmax_, cmap) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(np.rot90(img), cmap=cmap, vmin=vmin, vmax=vmax_)
            ax.set_title(lab, fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # V1 row
    w_rgb = _v1_rgb(wls["V1"][:, :, slice_z], wls["FA"][:, :, slice_z])
    p_rgb = _v1_rgb(inr["V1"][:, :, slice_z], inr["FA"][:, :, slice_z])
    vp = inr["V1"][:, :, slice_z]
    vr = wls["V1"][:, :, slice_z]
    dots = np.clip(np.abs(np.sum(vp * vr, axis=-1)), 0.0, 1.0)
    ang = np.where(msl, np.degrees(np.arccos(dots)), np.nan)
    amax = float(np.nanpercentile(ang[msl], 99)) if msl.any() else 90.0

    axes[4, 0].imshow(np.rot90(w_rgb))
    axes[4, 0].set_title("WLS V1", fontsize=9)
    axes[4, 0].axis("off")
    axes[4, 1].imshow(np.rot90(p_rgb))
    axes[4, 1].set_title("INR V1", fontsize=9)
    axes[4, 1].axis("off")
    im = axes[4, 2].imshow(np.rot90(ang), cmap="hot", vmin=0.0, vmax=max(amax, 1e-8))
    axes[4, 2].set_title("Angular err (deg)", fontsize=9)
    axes[4, 2].axis("off")
    fig.colorbar(im, ax=axes[4, 2], fraction=0.046, pad=0.02)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return slice_z
