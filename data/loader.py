"""HCP-YA diffusion data loader (independent implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass
class SubjectData:
    subject_id: str
    dwi: np.ndarray  # [X,Y,Z,N] selected volumes, float32
    bvals: np.ndarray  # [N]
    bvecs: np.ndarray  # [N,3]
    mask: np.ndarray  # [X,Y,Z] bool
    affine: np.ndarray
    coords_xyz: np.ndarray  # [V,3] voxel indices of masked voxels
    coords_norm: np.ndarray  # [V,3] in [-1,1]
    signal_scale: float
    shell_indices: np.ndarray  # indices into original volumes
    volume_shape: tuple[int, int, int]


def resolve_diffusion_dir(hcp_root: str | Path, subject_id: str) -> Path:
    """Resolve nested or flat HCP Diffusion folder."""
    root = Path(hcp_root)
    candidates = [
        root / subject_id / subject_id / "T1w" / "Diffusion",
        root / subject_id / "T1w" / "Diffusion",
    ]
    for c in candidates:
        if (c / "data.nii.gz").is_file() and (c / "bvals").is_file():
            return c
    raise FileNotFoundError(
        f"Diffusion folder not found for subject {subject_id} under {hcp_root}"
    )


def _read_bvals(path: Path) -> np.ndarray:
    text = path.read_text().replace(",", " ")
    return np.asarray([float(x) for x in text.split() if x], dtype=np.float64)


def _read_bvecs(path: Path) -> np.ndarray:
    text = path.read_text().replace(",", " ")
    vals = np.asarray([float(x) for x in text.split() if x], dtype=np.float64)
    if vals.size % 3 != 0:
        raise ValueError(f"bvecs length not divisible by 3: {vals.size}")
    n = vals.size // 3
    # HCP style: 3 rows x N cols flattened row-major
    arr = vals.reshape(3, n).T  # [N,3]
    return arr


def select_b0_b1000(
    bvals: np.ndarray,
    b0_thresh: float = 50.0,
    shell: float = 1000.0,
    shell_tol: float = 100.0,
) -> np.ndarray:
    """Return indices for b≈0 and b≈1000 volumes."""
    b0 = np.where(bvals < b0_thresh)[0]
    shell_idx = np.where(np.abs(bvals - shell) <= shell_tol)[0]
    if b0.size == 0:
        raise RuntimeError("No b0 volumes found")
    if shell_idx.size == 0:
        raise RuntimeError(f"No volumes near b={shell} (±{shell_tol})")
    return np.concatenate([b0, shell_idx]).astype(np.int64)


def normalize_coords(ijk: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Map integer voxel indices to [-1, 1] using volume shape."""
    scale = np.asarray(shape, dtype=np.float64) - 1.0
    scale = np.maximum(scale, 1.0)
    return (2.0 * ijk.astype(np.float64) / scale) - 1.0


def load_hcp_subject(
    hcp_root: str | Path,
    subject_id: str,
    shells: list[float] | None = None,
    b0_thresh: float = 50.0,
    shell_tol: float = 100.0,
    signal_percentile: float = 99.0,
) -> SubjectData:
    """Load one HCP-YA subject (b0 + selected shells; default b1000 only)."""
    if shells is None:
        shells = [1000.0]

    diff_dir = resolve_diffusion_dir(hcp_root, subject_id)
    img = nib.load(str(diff_dir / "data.nii.gz"))
    dwi_full = np.asanyarray(img.dataobj, dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float64)

    bvals = _read_bvals(diff_dir / "bvals")
    bvecs = _read_bvecs(diff_dir / "bvecs")
    if dwi_full.shape[-1] != bvals.shape[0]:
        raise ValueError(
            f"Volume count mismatch: dwi {dwi_full.shape[-1]} vs bvals {bvals.shape[0]}"
        )

    # b0 + requested shells (default: b1000 only; ignore b2000/b3000 in round 1)
    b0_idx = np.where(bvals < b0_thresh)[0]
    if b0_idx.size == 0:
        raise RuntimeError("No b0 volumes found")
    shell_parts = []
    for s in shells:
        part = np.where(np.abs(bvals - float(s)) <= shell_tol)[0]
        if part.size == 0:
            raise RuntimeError(f"No volumes near b={s} (±{shell_tol})")
        shell_parts.append(part)
    shell_idx = np.concatenate(shell_parts).astype(np.int64)
    sel = np.concatenate([b0_idx, shell_idx]).astype(np.int64)

    dwi_sel = dwi_full[..., sel]
    bvals_sel = bvals[sel].copy()
    bvecs_sel = bvecs[sel].copy()

    mask_img = nib.load(str(diff_dir / "nodif_brain_mask.nii.gz"))
    mask = np.asanyarray(mask_img.dataobj) > 0
    if mask.shape != dwi_sel.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} != dwi spatial {dwi_sel.shape[:3]}")

    # Signal scale from brain-masked mean-b0 (robust percentile)
    b0_local = np.where(bvals_sel < b0_thresh)[0]
    b0_mean_vol = dwi_sel[..., b0_local].mean(axis=-1)
    brain_vals = b0_mean_vol[mask]
    brain_vals = brain_vals[np.isfinite(brain_vals) & (brain_vals > 0)]
    if brain_vals.size == 0:
        raise RuntimeError("No positive b0 signal inside mask")
    signal_scale = float(np.percentile(brain_vals, signal_percentile))
    signal_scale = max(signal_scale, 1.0)

    dwi_sel = dwi_sel / signal_scale
    dwi_sel = np.clip(dwi_sel, 0.0, None).astype(np.float32)

    # Collapse multiple b0 volumes into one mean b0 so MSE is not dominated by b0.
    shell_local = np.where(bvals_sel >= b0_thresh)[0]
    mean_b0 = dwi_sel[..., b0_local].mean(axis=-1, keepdims=True)
    dwi = np.concatenate([mean_b0, dwi_sel[..., shell_local]], axis=-1).astype(np.float32)
    bvals_out = np.concatenate(
        [[0.0], bvals_sel[shell_local].astype(np.float64)]
    ).astype(np.float32)
    bvecs_out = np.concatenate(
        [np.zeros((1, 3), dtype=np.float32), bvecs_sel[shell_local].astype(np.float32)],
        axis=0,
    )
    # Normalize non-zero gradient directions
    norms = np.linalg.norm(bvecs_out, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 1e-8
    bvecs_out[nonzero] = bvecs_out[nonzero] / norms[nonzero]
    bvecs_out[~nonzero] = 0.0

    dwi = dwi
    bvals_sel = bvals_out
    bvecs_sel = bvecs_out

    coords_xyz = np.argwhere(mask).astype(np.int64)  # [V,3]
    coords_norm = normalize_coords(coords_xyz, dwi.shape[:3]).astype(np.float32)

    return SubjectData(
        subject_id=subject_id,
        dwi=dwi,
        bvals=bvals_sel.astype(np.float32),
        bvecs=bvecs_sel.astype(np.float32),
        mask=mask,
        affine=affine,
        coords_xyz=coords_xyz,
        coords_norm=coords_norm,
        signal_scale=signal_scale,
        shell_indices=sel,
        volume_shape=tuple(int(x) for x in dwi.shape[:3]),
    )


def masked_signals(data: SubjectData) -> np.ndarray:
    """Return [V, N] signals for masked voxels."""
    x, y, z = data.coords_xyz[:, 0], data.coords_xyz[:, 1], data.coords_xyz[:, 2]
    return data.dwi[x, y, z, :]
