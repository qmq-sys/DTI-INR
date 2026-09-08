"""Direction subsample helpers for sparse DTI experiments."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from data.loader import SubjectData


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return v / n


def greedy_even_sphere_indices(bvecs: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Pick ``n`` approximately evenly spaced directions (unsigned, |v|).

    Uses a greedy max-min angular separation heuristic on the sphere.
    """
    v = _unit(np.asarray(bvecs, dtype=np.float64).reshape(-1, 3))
    m = v.shape[0]
    if n <= 0:
        raise ValueError("n must be > 0")
    if n > m:
        raise ValueError(f"Requested {n} directions but only {m} available")
    if n == m:
        return np.arange(m, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    # Start from a reproducible random seed direction
    start = int(rng.integers(0, m))
    chosen = [start]
    # Precompute absolute dots for unsigned angle
    abs_dots = np.abs(v @ v.T)
    abs_dots = np.clip(abs_dots, 0.0, 1.0)

    while len(chosen) < n:
        # For each candidate, min |dot| to already chosen → want smallest |dot|
        # equivalently maximize angular separation: minimize max |dot| to chosen set
        max_sim = abs_dots[:, chosen].max(axis=1)
        max_sim[chosen] = np.inf  # never re-pick
        nxt = int(np.argmin(max_sim))
        chosen.append(nxt)
    return np.asarray(sorted(chosen), dtype=np.int64)


def subset_volumes(data: SubjectData, vol_indices: np.ndarray) -> SubjectData:
    """Return a SubjectData with only the selected volume indices."""
    idx = np.asarray(vol_indices, dtype=np.int64).ravel()
    return replace(
        data,
        dwi=data.dwi[..., idx].astype(np.float32, copy=False),
        bvals=data.bvals[idx].astype(np.float32, copy=False),
        bvecs=data.bvecs[idx].astype(np.float32, copy=False),
        shell_indices=data.shell_indices[idx] if data.shell_indices.ndim == 1 else idx,
    )


def make_sparse_protocol(
    data: SubjectData,
    n_directions: int,
    *,
    seed: int = 42,
    b0_thresh: float = 50.0,
) -> tuple[SubjectData, dict]:
    """Build ``1 b0 + n_directions b1000`` subset.

    ``n_directions`` does **not** count b0.
    Assumes ``data`` already has a single collapsed mean-b0 at index 0
    (as produced by ``load_hcp_subject`` with ``collapse_b0=True``).
    """
    bvals = np.asarray(data.bvals, dtype=np.float64).ravel()
    b0 = np.where(bvals < b0_thresh)[0]
    dw = np.where(bvals >= b0_thresh)[0]
    if b0.size < 1:
        raise RuntimeError("No b0 volume in data")
    if dw.size < int(n_directions):
        raise RuntimeError(
            f"Need {n_directions} DW directions, only {dw.size} available"
        )

    # Use first b0 (mean-b0 if collapsed)
    b0_i = int(b0[0])
    pick_local = greedy_even_sphere_indices(data.bvecs[dw], int(n_directions), seed=seed)
    dw_pick = dw[pick_local]
    vol_idx = np.concatenate([[b0_i], dw_pick]).astype(np.int64)
    sparse = subset_volumes(data, vol_idx)
    meta = {
        "n_directions": int(n_directions),
        "n_volumes": int(vol_idx.size),
        "includes_b0": True,
        "seed": int(seed),
        "volume_indices_in_full": vol_idx.tolist(),
        "bvals": sparse.bvals.astype(float).tolist(),
        "bvecs": sparse.bvecs.astype(float).tolist(),
    }
    return sparse, meta
