"""Full-data Reference Sparse DTI Estimation.

Experiment 1 (pilot)
--------------------
1 subject, one sampling seed, directions in {6,15,30,45}.
Sparse WLS + DTI-INR vs Full Reference.

Experiment 2 (multi-seed sparse robustness)
-------------------------------------------
subject=109830, directions=[6,15,30,45], seeds=[1,2,3,4,5]
Methods: Sparse WLS, DTI-INR
Metrics: FA/MD/AD/RD MAE, V1 angular error, Signal NRMSE
Output: mean ± std tables + curves vs #directions.

Usage::

    # Exp1
    python scripts/run_sparse_vs_full_ref.py --config configs/sparse_vs_full_ref.yaml

    # Exp2
    python scripts/run_sparse_vs_full_ref.py --config configs/sparse_vs_full_ref_exp2.yaml --exp2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.wls_dti import fit_wls_dti  # noqa: E402
from data.loader import SubjectData, load_hcp_subject  # noqa: E402
from data.sampling import make_sparse_protocol  # noqa: E402
from evaluation.dti_metrics import compare_dti_maps  # noqa: E402
from evaluation.signal_metrics import signal_nrmse  # noqa: E402
from training.train_v2_dti import (  # noqa: E402
    _forward_s0_d,
    build_model,
    infer_dti_maps,
    predict_signals_multi,
    set_seed,
)
from visualization.parameter_maps import save_comparison_figure  # noqa: E402

REF_KEYS = ("FA", "MD", "AD", "RD", "V1", "D", "S0")
EXP2_METRIC_KEYS = ("FA_MAE", "MD_MAE", "AD_MAE", "RD_MAE", "V1_error", "Signal_NRMSE")


def save_reference_maps(
    maps: dict[str, np.ndarray],
    affine: np.ndarray,
    out_dir: Path,
    *,
    meta: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in REF_KEYS:
        path = out_dir / f"{key}_ref.nii.gz"
        arr = np.asarray(maps[key], dtype=np.float32)
        nib.save(nib.Nifti1Image(arr, affine), str(path))
    if meta is not None:
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def load_reference_maps(ref_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in REF_KEYS:
        path = ref_dir / f"{key}_ref.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        out[key] = np.asanyarray(nib.load(str(path)).dataobj, dtype=np.float32)
    return out


def build_or_load_full_reference(
    *,
    hcp_root: str,
    subject_id: str,
    ref_root: Path,
    shells: list[float],
    force: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, Path, SubjectData]:
    ref_dir = ref_root / str(subject_id)
    full = load_hcp_subject(
        hcp_root=hcp_root,
        subject_id=subject_id,
        shells=shells,
        collapse_b0=False,
    )
    need = [ref_dir / f"{k}_ref.nii.gz" for k in REF_KEYS]
    if (not force) and all(p.is_file() for p in need):
        print(f"[ref] load existing {ref_dir}")
        maps = load_reference_maps(ref_dir)
        return maps, full.affine, ref_dir, full

    print(
        f"[ref] Fitting Full WLS on subject {subject_id}: "
        f"shape={full.dwi.shape}, n_b0={(full.bvals < 50).sum()}, "
        f"n_dw={(full.bvals >= 50).sum()}"
    )
    wls = fit_wls_dti(full.dwi, full.bvals, full.bvecs, mask=full.mask)
    maps = {
        "FA": wls["FA"].astype(np.float32),
        "MD": wls["MD"].astype(np.float32),
        "AD": wls["AD"].astype(np.float32),
        "RD": wls["RD"].astype(np.float32),
        "V1": wls["V1"].astype(np.float32),
        "D": wls["D"].astype(np.float32),
        "S0": (wls["S0"] * full.signal_scale).astype(np.float32),
    }
    meta = {
        "subject_id": subject_id,
        "hcp_root": str(hcp_root),
        "shells": shells,
        "n_volumes": int(full.dwi.shape[-1]),
        "n_b0": int((full.bvals < 50).sum()),
        "n_dw": int((full.bvals >= 50).sum()),
        "signal_scale": float(full.signal_scale),
        "volume_shape": list(full.volume_shape),
        "note": "Full b0 + Full b≈1000 DIPY WLS reference. Not used as INR supervision.",
    }
    save_reference_maps(maps, full.affine, ref_dir, meta=meta)
    print(f"[ref] saved -> {ref_dir}")
    return maps, full.affine, ref_dir, full


def train_inr_on_sparse(
    sparse: SubjectData,
    config: dict,
    device: torch.device,
    *,
    train_seed: int | None = None,
) -> torch.nn.Module:
    seed = int(config["training"]["seed"] if train_seed is None else train_seed)
    set_seed(seed)
    b_scale = float(config["training"].get("b_scale", 1000.0))
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["learning_rate"])
    val_frac = float(config["training"].get("val_fraction", 0.1))
    early_stop = bool(config["training"].get("early_stop", True))
    patience = int(config["training"].get("early_stop_patience", 25))
    min_delta = float(config["training"].get("early_stop_min_delta", 1e-6))

    x_np = sparse.coords_xyz
    sig = sparse.dwi[x_np[:, 0], x_np[:, 1], x_np[:, 2], :].astype(np.float32)
    n_vox = sig.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_vox)
    n_val = max(1, int(round(val_frac * n_vox)))
    idx_val = perm[:n_val]
    idx_train = perm[n_val:]

    g = torch.from_numpy(sparse.bvecs).to(device)
    b = torch.from_numpy(sparse.bvals / b_scale).to(device)
    bvals_t = torch.from_numpy(sparse.bvals).to(device)
    bvecs_t = torch.from_numpy(sparse.bvecs).to(device)
    signals_t = torch.from_numpy(sig)
    coords_t = torch.from_numpy(sparse.coords_norm.astype(np.float32))

    model = build_model(config, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_q = getattr(model, "uses_q_feature", False)

    while batch_size > 256:
        try:
            idx = torch.arange(min(batch_size, idx_train.shape[0]))
            x = coords_t[idx].to(device)
            y = signals_t[idx].to(device)
            S0, D = _forward_s0_d(model, x, s=y, bvals=bvals_t, bvecs=bvecs_t)
            _ = predict_signals_multi(S0, D, g, b)
            del S0, D, _
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            batch_size //= 2
            if device.type == "cuda":
                torch.cuda.empty_cache()

    best_val = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(idx_train.shape[0])
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, idx_train.shape[0], batch_size):
            sel = idx_train[order[start : start + batch_size]]
            x = coords_t[sel].to(device)
            y = signals_t[sel].to(device)
            S0, D = _forward_s0_d(
                model,
                x,
                s=y if use_q else None,
                bvals=bvals_t if use_q else None,
                bvecs=bvecs_t if use_q else None,
            )
            pred = predict_signals_multi(S0, D, g, b)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_steps += 1
        epoch_loss /= max(n_steps, 1)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            v_steps = 0
            for start in range(0, idx_val.shape[0], batch_size):
                sel = idx_val[start : start + batch_size]
                x = coords_t[sel].to(device)
                y = signals_t[sel].to(device)
                S0, D = _forward_s0_d(
                    model,
                    x,
                    s=y if use_q else None,
                    bvals=bvals_t if use_q else None,
                    bvecs=bvecs_t if use_q else None,
                )
                pred = predict_signals_multi(S0, D, g, b)
                val_loss += float(torch.mean((pred - y) ** 2).item())
                v_steps += 1
            val_loss /= max(v_steps, 1)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"  [inr] epoch {epoch:4d}/{epochs}  "
                f"train={epoch_loss:.4e}  val={val_loss:.4e}  best={best_val:.4e}"
            )
        if early_stop and stale >= patience and epoch >= 40:
            print(f"  [inr] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def metrics_vs_ref(
    pred: dict[str, np.ndarray], ref: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, float]:
    raw = compare_dti_maps(pred, ref, mask)
    return {
        "FA_MAE": raw["FA_MAE"],
        "FA_RMSE": raw["FA_RMSE"],
        "MD_MAE": raw["MD_MAE"],
        "MD_RMSE": raw["MD_RMSE"],
        "AD_MAE": raw["AD_MAE"],
        "AD_RMSE": raw["AD_RMSE"],
        "RD_MAE": raw["RD_MAE"],
        "RD_RMSE": raw["RD_RMSE"],
        "V1_error": raw["V1_ang_mean_deg"],
        "V1_error_median": raw["V1_ang_median_deg"],
    }


def _masked_signals(data: SubjectData) -> np.ndarray:
    x, y, z = data.coords_xyz[:, 0], data.coords_xyz[:, 1], data.coords_xyz[:, 2]
    return data.dwi[x, y, z, :].astype(np.float32)


def wls_predict_signals(
    wls: dict[str, np.ndarray],
    data: SubjectData,
) -> np.ndarray:
    """Forward-simulate sparse signals from WLS (S0, D) on masked voxels."""
    xyz = data.coords_xyz
    S0 = wls["S0"][xyz[:, 0], xyz[:, 1], xyz[:, 2]].astype(np.float64)  # normalized units
    D = wls["D"][xyz[:, 0], xyz[:, 1], xyz[:, 2]].astype(np.float64)  # [V,3,3]
    g = data.bvecs.astype(np.float64)
    b = data.bvals.astype(np.float64)
    Dg = np.einsum("vij,nj->vni", D, g)
    q = np.einsum("nj,vnj->vn", g, Dg)
    exponent = np.clip(-b[None, :] * q, -60.0, 60.0)
    return (S0[:, None] * np.exp(exponent)).astype(np.float32)


@torch.no_grad()
def inr_predict_signals(
    model: torch.nn.Module,
    data: SubjectData,
    device: torch.device,
    b_scale: float,
    chunk: int = 4096,
) -> np.ndarray:
    model.eval()
    g = torch.from_numpy(data.bvecs).to(device)
    b = torch.from_numpy(data.bvals / b_scale).to(device)
    bvals = torch.from_numpy(data.bvals).to(device)
    bvecs = torch.from_numpy(data.bvecs).to(device)
    coords = torch.from_numpy(data.coords_norm).to(device)
    sig = torch.from_numpy(_masked_signals(data)).to(device)
    use_q = getattr(model, "uses_q_feature", False)
    preds = []
    for i in range(0, coords.shape[0], chunk):
        sl = slice(i, i + chunk)
        S0, D = _forward_s0_d(
            model,
            coords[sl],
            s=sig[sl] if use_q else None,
            bvals=bvals if use_q else None,
            bvecs=bvecs if use_q else None,
        )
        preds.append(predict_signals_multi(S0, D, g, b).cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_one_sparse_trial(
    *,
    subject_id: str,
    n_dir: int,
    sample_seed: int,
    full_collapsed: SubjectData,
    ref_maps: dict[str, np.ndarray],
    config: dict,
    device: torch.device,
    out_dir: Path | None,
    save_figures: bool,
) -> tuple[dict, dict]:
    """Run Sparse WLS + DTI-INR for one (n_dir, seed). Returns (wls_row, inr_row)."""
    b_scale = float(config["training"].get("b_scale", 1000.0))
    sparse, proto = make_sparse_protocol(full_collapsed, n_dir, seed=sample_seed)
    obs = _masked_signals(sparse)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "sparse_protocol.json", "w", encoding="utf-8") as f:
            json.dump(proto, f, indent=2)

    print(f"[wls] n={n_dir} seed={sample_seed}")
    wls = fit_wls_dti(sparse.dwi, sparse.bvals, sparse.bvecs, mask=sparse.mask)
    wls_maps = {
        "FA": wls["FA"].astype(np.float32),
        "MD": wls["MD"].astype(np.float32),
        "AD": wls["AD"].astype(np.float32),
        "RD": wls["RD"].astype(np.float32),
        "V1": wls["V1"].astype(np.float32),
        "S0": (wls["S0"] * sparse.signal_scale).astype(np.float32),
        "D": wls["D"].astype(np.float32),
    }
    # Signal NRMSE uses normalized S0 (same units as sparse.dwi)
    wls_sig = wls_predict_signals(
        {"S0": wls["S0"].astype(np.float32), "D": wls["D"].astype(np.float32)},
        sparse,
    )
    wls_m = metrics_vs_ref(wls_maps, ref_maps, sparse.mask)
    wls_m["Signal_NRMSE"] = signal_nrmse(wls_sig, obs)
    row_wls = {
        "subject_id": subject_id,
        "n_directions": n_dir,
        "seed": sample_seed,
        "method": "Sparse_WLS",
        **wls_m,
    }

    print(f"[inr] n={n_dir} seed={sample_seed}")
    # Direction sample seed + train seed share the trial seed for reproducibility
    cfg = dict(config)
    cfg["training"] = dict(config["training"])
    cfg["training"]["seed"] = int(sample_seed)
    model = train_inr_on_sparse(sparse, cfg, device, train_seed=int(sample_seed))
    inr_maps = infer_dti_maps(model, sparse, device, b_scale=b_scale)
    inr_sig = inr_predict_signals(model, sparse, device, b_scale=b_scale)
    inr_m = metrics_vs_ref(inr_maps, ref_maps, sparse.mask)
    inr_m["Signal_NRMSE"] = signal_nrmse(inr_sig, obs)
    row_inr = {
        "subject_id": subject_id,
        "n_directions": n_dir,
        "seed": sample_seed,
        "method": "DTI_INR",
        **inr_m,
    }

    if out_dir is not None:
        with open(out_dir / "metrics_wls.json", "w", encoding="utf-8") as f:
            json.dump(row_wls, f, indent=2)
        with open(out_dir / "metrics_inr.json", "w", encoding="utf-8") as f:
            json.dump(row_inr, f, indent=2)
        torch.save(
            {"model_state": model.state_dict(), "config": cfg, "protocol": proto},
            out_dir / "checkpoint.pt",
        )
        if save_figures:
            save_comparison_figure(
                ref_maps,
                inr_maps,
                sparse.mask,
                out_dir / "comparison_inr_vs_ref.png",
                title=f"{subject_id} n={n_dir} seed={sample_seed}: FullRef | INR | |Error|",
            )
            save_comparison_figure(
                ref_maps,
                wls_maps,
                sparse.mask,
                out_dir / "comparison_wls_vs_ref.png",
                title=f"{subject_id} n={n_dir} seed={sample_seed}: FullRef | SparseWLS | |Error|",
            )

    print(
        f"[table] n={n_dir} seed={sample_seed}\n"
        f"  Sparse_WLS  FA={wls_m['FA_MAE']:.4f} MD={wls_m['MD_MAE']:.4g} "
        f"AD={wls_m['AD_MAE']:.4g} RD={wls_m['RD_MAE']:.4g} "
        f"V1={wls_m['V1_error']:.2f} NRMSE={wls_m['Signal_NRMSE']:.4f}\n"
        f"  DTI_INR     FA={inr_m['FA_MAE']:.4f} MD={inr_m['MD_MAE']:.4g} "
        f"AD={inr_m['AD_MAE']:.4g} RD={inr_m['RD_MAE']:.4g} "
        f"V1={inr_m['V1_error']:.2f} NRMSE={inr_m['Signal_NRMSE']:.4f}"
    )
    return row_wls, row_inr


def run_subject(config: dict, subject_id: str, run_dir: Path, device: torch.device) -> list[dict]:
    hcp_root = config["dataset"]["hcp_root"]
    shells = list(config.get("reference", {}).get("shells", [1000.0]))
    ref_root = ROOT / str(config.get("reference", {}).get("root", "reference"))
    n_list = [int(x) for x in config["sparse"]["n_directions"]]
    seed = int(config["sparse"].get("seed", 42))

    ref_maps, _affine, _ref_dir, _full = build_or_load_full_reference(
        hcp_root=hcp_root,
        subject_id=subject_id,
        ref_root=ref_root,
        shells=shells,
    )
    full_collapsed = load_hcp_subject(
        hcp_root=hcp_root,
        subject_id=subject_id,
        shells=shells,
        collapse_b0=True,
    )

    subj_dir = run_dir / f"hcp_{subject_id}"
    subj_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for n_dir in n_list:
        print(f"\n===== {subject_id} | 1 b0 + {n_dir} DW | seed={seed} =====")
        wls_row, inr_row = run_one_sparse_trial(
            subject_id=subject_id,
            n_dir=n_dir,
            sample_seed=seed,
            full_collapsed=full_collapsed,
            ref_maps=ref_maps,
            config=config,
            device=device,
            out_dir=subj_dir / f"n{n_dir}",
            save_figures=True,
        )
        rows.extend([wls_row, inr_row])
    return rows


def write_tables(rows: list[dict], run_dir: Path) -> None:
    keys = [
        "subject_id",
        "n_directions",
        "seed",
        "method",
        "FA_MAE",
        "MD_MAE",
        "AD_MAE",
        "RD_MAE",
        "V1_error",
        "Signal_NRMSE",
        "FA_RMSE",
        "MD_RMSE",
        "AD_RMSE",
        "RD_RMSE",
    ]
    with open(run_dir / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(run_dir / "results_table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "| n_dir | seed | Method | FA MAE | MD MAE | AD MAE | RD MAE | V1 error | Signal NRMSE |",
        "|------:|-----:|:-------|-------:|-------:|-------:|-------:|---------:|-------------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('n_directions', '')} | {r.get('seed', '')} | {r['method']} | "
            f"{r['FA_MAE']:.4f} | {r['MD_MAE']:.4g} | {r['AD_MAE']:.4g} | {r['RD_MAE']:.4g} | "
            f"{r['V1_error']:.2f} | {r.get('Signal_NRMSE', float('nan')):.4f} |"
        )
    (run_dir / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Experiment 2: multi-seed aggregation + plots
# ---------------------------------------------------------------------------


def aggregate_mean_std(rows: list[dict]) -> list[dict]:
    """Group by (n_directions, method) -> mean/std over seeds."""
    buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(int(r["n_directions"]), str(r["method"]))].append(r)

    summary = []
    for (n_dir, method), group in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        row: dict = {
            "n_directions": n_dir,
            "method": method,
            "n_seeds": len(group),
        }
        for k in EXP2_METRIC_KEYS:
            vals = np.asarray([float(g[k]) for g in group], dtype=np.float64)
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            row[f"{k}_mean_std"] = f"{row[f'{k}_mean']:.6g} ± {row[f'{k}_std']:.6g}"
        summary.append(row)
    return summary


def write_exp2_summary(summary: list[dict], raw_rows: list[dict], run_dir: Path) -> None:
    with open(run_dir / "exp2_raw_results.json", "w", encoding="utf-8") as f:
        json.dump(raw_rows, f, indent=2)
    with open(run_dir / "exp2_mean_std.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fieldnames = ["n_directions", "method", "n_seeds"]
    for k in EXP2_METRIC_KEYS:
        fieldnames += [f"{k}_mean", f"{k}_std", f"{k}_mean_std"]
    with open(run_dir / "exp2_mean_std.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in summary:
            w.writerow(r)

    lines = [
        "| n_dir | Method | FA MAE | MD MAE | AD MAE | RD MAE | V1 error | Signal NRMSE |",
        "|------:|:-------|-------:|-------:|-------:|-------:|---------:|-------------:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['n_directions']} | {r['method']} | "
            f"{r['FA_MAE_mean_std']} | {r['MD_MAE_mean_std']} | "
            f"{r['AD_MAE_mean_std']} | {r['RD_MAE_mean_std']} | "
            f"{r['V1_error_mean_std']} | {r['Signal_NRMSE_mean_std']} |"
        )
    text = "\n".join(lines) + "\n"
    (run_dir / "exp2_mean_std.md").write_text(text, encoding="utf-8")
    print("\n=== Experiment 2: mean ± std ===\n" + text)


def plot_exp2_curves(summary: list[dict], run_dir: Path) -> None:
    """Plot metric vs #directions for Sparse_WLS and DTI_INR (mean ± std)."""
    plot_specs = [
        ("FA_MAE", "FA MAE", "FA_MAE_vs_directions.png"),
        ("MD_MAE", "MD MAE", "MD_MAE_vs_directions.png"),
        ("AD_MAE", "AD MAE", "AD_MAE_vs_directions.png"),
        ("RD_MAE", "RD MAE", "RD_MAE_vs_directions.png"),
        ("V1_error", "V1 angular error (deg)", "V1_error_vs_directions.png"),
        ("Signal_NRMSE", "Signal NRMSE", "Signal_NRMSE_vs_directions.png"),
    ]
    methods = ["Sparse_WLS", "DTI_INR"]
    colors = {"Sparse_WLS": "#1f77b4", "DTI_INR": "#d62728"}
    fig_dir = run_dir / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for key, ylabel, fname in plot_specs:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for method in methods:
            rows = [r for r in summary if r["method"] == method]
            rows = sorted(rows, key=lambda r: int(r["n_directions"]))
            if not rows:
                continue
            xs = [int(r["n_directions"]) for r in rows]
            ys = [float(r[f"{key}_mean"]) for r in rows]
            es = [float(r[f"{key}_std"]) for r in rows]
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                marker="o",
                capsize=4,
                label=method.replace("_", " "),
                color=colors.get(method),
                linewidth=1.8,
            )
        ax.set_xlabel("# directions (b1000)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs #directions")
        ax.set_xticks(sorted({int(r["n_directions"]) for r in summary}))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)

    # Combined 2x3 panel
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, (key, ylabel, _) in zip(axes, plot_specs):
        for method in methods:
            rows = [r for r in summary if r["method"] == method]
            rows = sorted(rows, key=lambda r: int(r["n_directions"]))
            if not rows:
                continue
            xs = [int(r["n_directions"]) for r in rows]
            ys = [float(r[f"{key}_mean"]) for r in rows]
            es = [float(r[f"{key}_std"]) for r in rows]
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                marker="o",
                capsize=3,
                label=method.replace("_", " "),
                color=colors.get(method),
                linewidth=1.5,
            )
        ax.set_xlabel("# directions")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({int(r["n_directions"]) for r in summary}))
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Experiment 2: Multi-seed Sparse Robustness (mean ± std)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "all_metrics_vs_directions.png", dpi=160)
    plt.close(fig)
    print(f"[plots] saved under {fig_dir}")


def run_experiment2(config: dict, run_dir: Path, device: torch.device) -> list[dict]:
    e2 = dict(config.get("experiment2") or {})
    subject_id = str(e2.get("subject") or config["dataset"]["subjects"][0])
    n_list = [int(x) for x in e2.get("n_directions", config["sparse"]["n_directions"])]
    seeds = [int(x) for x in e2.get("seeds", config["sparse"].get("seeds", [1, 2, 3, 4, 5]))]
    save_figures = bool(e2.get("save_comparison_figures", False))

    hcp_root = config["dataset"]["hcp_root"]
    shells = list(config.get("reference", {}).get("shells", [1000.0]))
    ref_root = ROOT / str(config.get("reference", {}).get("root", "reference"))

    print(f"[exp2] subject={subject_id} directions={n_list} seeds={seeds}")
    ref_maps, _affine, _ref_dir, _full = build_or_load_full_reference(
        hcp_root=hcp_root,
        subject_id=subject_id,
        ref_root=ref_root,
        shells=shells,
    )
    full_collapsed = load_hcp_subject(
        hcp_root=hcp_root,
        subject_id=subject_id,
        shells=shells,
        collapse_b0=True,
    )

    subj_dir = run_dir / f"hcp_{subject_id}"
    rows: list[dict] = []
    for n_dir in n_list:
        for seed in seeds:
            tag = f"n{n_dir}_seed{seed}"
            print(f"\n===== Exp2 {subject_id} | 1 b0 + {n_dir} DW | seed={seed} =====")
            out = subj_dir / tag
            # Skip if already done (resume-friendly)
            wls_p = out / "metrics_wls.json"
            inr_p = out / "metrics_inr.json"
            if wls_p.is_file() and inr_p.is_file():
                print(f"[skip] existing {tag}")
                with open(wls_p, encoding="utf-8") as f:
                    rows.append(json.load(f))
                with open(inr_p, encoding="utf-8") as f:
                    rows.append(json.load(f))
                continue
            wls_row, inr_row = run_one_sparse_trial(
                subject_id=subject_id,
                n_dir=n_dir,
                sample_seed=seed,
                full_collapsed=full_collapsed,
                ref_maps=ref_maps,
                config=config,
                device=device,
                out_dir=out,
                save_figures=save_figures,
            )
            rows.extend([wls_row, inr_row])
            # Incremental save
            write_tables(rows, run_dir)
            summary = aggregate_mean_std(rows)
            write_exp2_summary(summary, rows, run_dir)

    summary = aggregate_mean_std(rows)
    write_exp2_summary(summary, rows, run_dir)
    plot_exp2_curves(summary, run_dir)
    write_tables(rows, run_dir)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse DTI vs Full Reference (+ Exp2 multi-seed)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "sparse_vs_full_ref.yaml"),
    )
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--n-directions", nargs="*", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Exp2 sampling seeds")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force-ref", action="store_true")
    parser.add_argument("--ref-only", action="store_true")
    parser.add_argument(
        "--exp2",
        action="store_true",
        help="Run Experiment 2: multi-seed sparse robustness",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Resume / write into existing experiment folder",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.subjects:
        config["dataset"]["subjects"] = list(args.subjects)
    if args.n_directions:
        config["sparse"]["n_directions"] = list(args.n_directions)
        config.setdefault("experiment2", {})["n_directions"] = list(args.n_directions)
    if args.seeds:
        config.setdefault("experiment2", {})["seeds"] = list(args.seeds)
        config.setdefault("sparse", {})["seeds"] = list(args.seeds)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)

    # Auto-enable exp2 if config says so
    if config.get("experiment2", {}).get("enabled", False):
        args.exp2 = True

    subjects = [str(s) for s in config["dataset"]["subjects"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "exp2_multiseed" if args.exp2 else "sparse_vs_full_ref"
        run_dir = ROOT / "experiments" / tag / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"[run] {run_dir}")
    print(f"[run] subjects={subjects} device={device} exp2={args.exp2}")

    if args.ref_only or args.force_ref:
        ref_root = ROOT / str(config.get("reference", {}).get("root", "reference"))
        shells = list(config.get("reference", {}).get("shells", [1000.0]))
        for sid in subjects:
            build_or_load_full_reference(
                hcp_root=config["dataset"]["hcp_root"],
                subject_id=sid,
                ref_root=ref_root,
                shells=shells,
                force=bool(args.force_ref),
            )
        if args.ref_only:
            print("[done] reference only")
            return

    if args.exp2:
        run_experiment2(config, run_dir, device)
    else:
        all_rows: list[dict] = []
        for sid in subjects:
            all_rows.extend(run_subject(config, sid, run_dir, device))
        write_tables(all_rows, run_dir)

    print(f"\n[done] {run_dir}")
    print(f"[ref]  {ROOT / config.get('reference', {}).get('root', 'reference')}")


if __name__ == "__main__":
    main()
