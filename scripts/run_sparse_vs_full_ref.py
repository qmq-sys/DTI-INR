"""Full-data Reference Sparse DTI Estimation (pilot).

Pipeline
--------
1) Load Full b0 + Full b=1000
2) DIPY WLS -> D_ref / FA_ref / ...  saved under ``reference/<subject_id>/``
3) Sparse subsample: 1 b0 + {6,15,30,45} b1000 directions
4) Sparse WLS and DTI-INR on identical sparse input
5) Score both against Full Reference (not against each other)

Usage::

    cd DTI-INR
    python scripts/run_sparse_vs_full_ref.py --config configs/sparse_vs_full_ref.yaml
    python scripts/run_sparse_vs_full_ref.py --subjects 109830 --n-directions 6
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

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
from training.train_v2_dti import (  # noqa: E402
    build_model,
    infer_dti_maps,
    predict_signals_multi,
    set_seed,
    _forward_s0_d,
)
from visualization.parameter_maps import save_comparison_figure  # noqa: E402

REF_KEYS = ("FA", "MD", "AD", "RD", "V1", "D", "S0")


def save_reference_maps(
    maps: dict[str, np.ndarray],
    affine: np.ndarray,
    out_dir: Path,
    *,
    meta: dict | None = None,
) -> None:
    """Save Full Reference under ``reference/<subject_id>/`` with ``*_ref`` names."""
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
    """Fit Full WLS reference (all b0 + all b1000) and save to reference/<id>/."""
    ref_dir = ref_root / str(subject_id)
    full = load_hcp_subject(
        hcp_root=hcp_root,
        subject_id=subject_id,
        shells=shells,
        collapse_b0=False,  # Full b0 + Full b1000
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
    # WLS on physical-ish intensities: data already / signal_scale
    wls = fit_wls_dti(full.dwi, full.bvals, full.bvecs, mask=full.mask)
    # Store S0 in original intensity units for readability
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
) -> torch.nn.Module:
    """Train SpatialDTIParamField on sparse volumes (signal MSE only)."""
    set_seed(int(config["training"]["seed"]))
    b_scale = float(config["training"].get("b_scale", 1000.0))
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["learning_rate"])
    val_frac = float(config["training"].get("val_fraction", 0.1))
    early_stop = bool(config["training"].get("early_stop", True))
    patience = int(config["training"].get("early_stop_patience", 25))
    min_delta = float(config["training"].get("early_stop_min_delta", 1e-6))

    x_np = sparse.coords_xyz
    # signals from sparse.dwi
    sig = sparse.dwi[x_np[:, 0], x_np[:, 1], x_np[:, 2], :].astype(np.float32)
    n_vox = sig.shape[0]
    rng = np.random.default_rng(int(config["training"]["seed"]))
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
                model, x, s=y if use_q else None, bvals=bvals_t if use_q else None, bvecs=bvecs_t if use_q else None
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


def metrics_vs_ref(pred: dict[str, np.ndarray], ref: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
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


def run_subject(config: dict, subject_id: str, run_dir: Path, device: torch.device) -> list[dict]:
    hcp_root = config["dataset"]["hcp_root"]
    shells = list(config.get("reference", {}).get("shells", [1000.0]))
    ref_root = ROOT / str(config.get("reference", {}).get("root", "reference"))
    n_list = [int(x) for x in config["sparse"]["n_directions"]]
    seed = int(config["sparse"].get("seed", 42))
    b_scale = float(config["training"].get("b_scale", 1000.0))

    ref_maps, affine, ref_dir, _full_for_meta = build_or_load_full_reference(
        hcp_root=hcp_root,
        subject_id=subject_id,
        ref_root=ref_root,
        shells=shells,
    )

    # Collapsed mean-b0 + full b1000 for sparse sampling / INR training
    full_collapsed = load_hcp_subject(
        hcp_root=hcp_root,
        subject_id=subject_id,
        shells=shells,
        collapse_b0=True,
    )
    # Align signal_scale with reference fit if possible
    meta_path = ref_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # full_collapsed already uses same percentile scale from loader

    subj_dir = run_dir / f"hcp_{subject_id}"
    subj_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for n_dir in n_list:
        tag = f"n{n_dir}"
        out = subj_dir / tag
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n===== {subject_id} | 1 b0 + {n_dir} DW =====")

        sparse, proto = make_sparse_protocol(full_collapsed, n_dir, seed=seed)
        with open(out / "sparse_protocol.json", "w", encoding="utf-8") as f:
            json.dump(proto, f, indent=2)

        # --- Sparse WLS ---
        print("[wls] fitting sparse WLS ...")
        wls = fit_wls_dti(sparse.dwi, sparse.bvals, sparse.bvecs, mask=sparse.mask)
        wls_maps = {
            "FA": wls["FA"].astype(np.float32),
            "MD": wls["MD"].astype(np.float32),
            "AD": wls["AD"].astype(np.float32),
            "RD": wls["RD"].astype(np.float32),
            "V1": wls["V1"].astype(np.float32),
            "S0": (wls["S0"] * sparse.signal_scale).astype(np.float32),
        }
        wls_m = metrics_vs_ref(wls_maps, ref_maps, sparse.mask)
        row_wls = {"subject_id": subject_id, "n_directions": n_dir, "method": "Sparse_WLS", **wls_m}
        rows.append(row_wls)
        with open(out / "metrics_wls.json", "w", encoding="utf-8") as f:
            json.dump(row_wls, f, indent=2)

        # --- DTI-INR ---
        print("[inr] training DTI-INR on sparse input ...")
        model = train_inr_on_sparse(sparse, config, device)
        inr_maps = infer_dti_maps(model, sparse, device, b_scale=b_scale)
        inr_m = metrics_vs_ref(inr_maps, ref_maps, sparse.mask)
        row_inr = {"subject_id": subject_id, "n_directions": n_dir, "method": "DTI_INR", **inr_m}
        rows.append(row_inr)
        with open(out / "metrics_inr.json", "w", encoding="utf-8") as f:
            json.dump(row_inr, f, indent=2)

        torch.save(
            {"model_state": model.state_dict(), "config": config, "protocol": proto},
            out / "checkpoint.pt",
        )

        # Comparison figure: Ref | SparseWLS | INR  for FA (and full panel vs ref for INR)
        save_comparison_figure(
            ref_maps,
            inr_maps,
            sparse.mask,
            out / "comparison_inr_vs_ref.png",
            title=f"{subject_id} n={n_dir}: FullRef | INR | |Error|",
        )
        save_comparison_figure(
            ref_maps,
            wls_maps,
            sparse.mask,
            out / "comparison_wls_vs_ref.png",
            title=f"{subject_id} n={n_dir}: FullRef | SparseWLS | |Error|",
        )

        # Compact table print
        print(
            f"[table] n={n_dir}\n"
            f"  Sparse_WLS  FA_MAE={wls_m['FA_MAE']:.4f}  MD_MAE={wls_m['MD_MAE']:.4g}  "
            f"AD_MAE={wls_m['AD_MAE']:.4g}  RD_MAE={wls_m['RD_MAE']:.4g}  V1={wls_m['V1_error']:.2f}°\n"
            f"  DTI_INR     FA_MAE={inr_m['FA_MAE']:.4f}  MD_MAE={inr_m['MD_MAE']:.4g}  "
            f"AD_MAE={inr_m['AD_MAE']:.4g}  RD_MAE={inr_m['RD_MAE']:.4g}  V1={inr_m['V1_error']:.2f}°"
        )

    return rows


def write_tables(rows: list[dict], run_dir: Path) -> None:
    keys = [
        "subject_id",
        "n_directions",
        "method",
        "FA_MAE",
        "MD_MAE",
        "AD_MAE",
        "RD_MAE",
        "V1_error",
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

    # Pretty markdown-style summary for the requested Method columns
    lines = [
        "| n_dir | Method | FA MAE | MD MAE | AD MAE | RD MAE | V1 error |",
        "|------:|:-------|-------:|-------:|-------:|-------:|---------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n_directions']} | {r['method']} | {r['FA_MAE']:.4f} | {r['MD_MAE']:.4g} | "
            f"{r['AD_MAE']:.4g} | {r['RD_MAE']:.4g} | {r['V1_error']:.2f} |"
        )
    (run_dir / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse DTI vs Full Reference pilot")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "sparse_vs_full_ref.yaml"),
    )
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--n-directions", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force-ref", action="store_true", help="Recompute Full Reference")
    parser.add_argument("--ref-only", action="store_true", help="Only build reference/, skip sparse")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.subjects:
        config["dataset"]["subjects"] = list(args.subjects)
    if args.n_directions:
        config["sparse"]["n_directions"] = list(args.n_directions)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)

    subjects = [str(s) for s in config["dataset"]["subjects"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / "sparse_vs_full_ref" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"[run] {run_dir}")
    print(f"[run] subjects={subjects} device={device}")

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

    all_rows: list[dict] = []
    for sid in subjects:
        all_rows.extend(run_subject(config, sid, run_dir, device))
    write_tables(all_rows, run_dir)
    print(f"\n[done] {run_dir}")
    print(f"[ref]  {ROOT / config.get('reference', {}).get('root', 'reference')}")


if __name__ == "__main__":
    main()
