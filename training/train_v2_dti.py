"""Train version-two SpatialDTIParamField (DTI only) on multiple HCP subjects.

Results are written under::

    experiments/v2_dti_param_field/<timestamp>/
        summary.csv / summary.json
        <subject_id>/
            metrics.json, metrics.csv
            checkpoint.pt, loss_curve.png
            comparison.png
            maps/ ...

Metrics (vs WLS-DTI reference, not supervision):
  FA/MD/AD/RD MAE & RMSE, V1 angular error, Signal MSE/MAE/PSNR.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.wls_dti import fit_wls_dti  # noqa: E402
from data.loader import SubjectData, load_hcp_subject, masked_signals  # noqa: E402
from evaluation.dti_metrics import compare_dti_maps, tensor_to_scalars  # noqa: E402
from evaluation.signal_metrics import signal_mae, signal_mse, signal_psnr  # noqa: E402
from models.spatial_dti_param_field import (  # noqa: E402
    SpatialDTIParamField,
    SpatialDTIParamFieldWithQ,
)
from visualization.parameter_maps import save_comparison_figure  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_signals_multi(S0: Tensor, D: Tensor, g: Tensor, b: Tensor) -> Tensor:
    """S0 [B] or [B,1], D [B,3,3], g [N,3], b [N] -> S [B,N]."""
    if S0.ndim == 2:
        S0 = S0.squeeze(-1)
    Dg = torch.einsum("bij,nj->bni", D, g)
    q = torch.einsum("nj,bnj->bn", g, Dg)
    exponent = torch.clamp(-b[None, :] * q, min=-60.0, max=60.0)
    return S0[:, None] * torch.exp(exponent)


def volume_from_masked(
    values: np.ndarray,
    coords_xyz: np.ndarray,
    shape: tuple[int, int, int],
    n_comp: int | None = None,
) -> np.ndarray:
    if n_comp is None:
        out = np.zeros(shape, dtype=np.float32)
        out[coords_xyz[:, 0], coords_xyz[:, 1], coords_xyz[:, 2]] = values.astype(np.float32)
    else:
        out = np.zeros(shape + (n_comp,), dtype=np.float32)
        out[coords_xyz[:, 0], coords_xyz[:, 1], coords_xyz[:, 2]] = values.astype(np.float32)
    return out


def build_model(config: dict, device: torch.device) -> torch.nn.Module:
    mcfg = config["model"]
    hash_kw = dict(mcfg.get("hash_encoding") or {})
    common = dict(
        hidden=int(mcfg["hidden_dim"]),
        layers=int(mcfg["num_layers"]),
        use_hashgrid=bool(mcfg.get("use_hashgrid", True)),
        hashgrid_concat_xyz=bool(mcfg.get("hashgrid_concat_xyz", True)),
        hash_encoding_kwargs=hash_kw,
    )
    if bool(mcfg.get("use_q_feature", False)):
        model = SpatialDTIParamFieldWithQ(
            **common,
            q_embed_dim=int(mcfg.get("q_embed_dim", 32)),
        )
    else:
        model = SpatialDTIParamField(**common)
    return model.to(device)


def _forward_s0_d(
    model: torch.nn.Module,
    x: Tensor,
    *,
    s: Tensor | None,
    bvals: Tensor | None,
    bvecs: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if getattr(model, "uses_q_feature", False):
        return model(x, s=s, bvals=bvals, bvecs=bvecs)
    return model(x)


@torch.no_grad()
def infer_dti_maps(
    model: torch.nn.Module,
    data: SubjectData,
    device: torch.device,
    b_scale: float,
    chunk: int = 8192,
) -> dict[str, np.ndarray]:
    model.eval()
    coords = torch.from_numpy(data.coords_norm).to(device)
    sig = torch.from_numpy(masked_signals(data)).to(device)
    bvals = torch.from_numpy(data.bvals).to(device)
    bvecs = torch.from_numpy(data.bvecs).to(device)
    S0_list, D_list = [], []
    for i in range(0, coords.shape[0], chunk):
        sl = slice(i, i + chunk)
        S0, D = _forward_s0_d(
            model,
            coords[sl],
            s=sig[sl] if getattr(model, "uses_q_feature", False) else None,
            bvals=bvals if getattr(model, "uses_q_feature", False) else None,
            bvecs=bvecs if getattr(model, "uses_q_feature", False) else None,
        )
        S0_list.append(S0.squeeze(-1).cpu().numpy())
        D_list.append((D / b_scale).cpu().numpy())
    S0_v = np.concatenate(S0_list, axis=0)
    D_v = np.concatenate(D_list, axis=0)
    scalars = tensor_to_scalars(D_v)
    shape = data.volume_shape
    xyz = data.coords_xyz
    D_vol = np.zeros(shape + (3, 3), dtype=np.float32)
    D_vol[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = D_v.astype(np.float32)
    return {
        "S0": volume_from_masked(S0_v * data.signal_scale, xyz, shape),
        "D": D_vol,
        "FA": volume_from_masked(scalars["FA"], xyz, shape),
        "MD": volume_from_masked(scalars["MD"], xyz, shape),
        "AD": volume_from_masked(scalars["AD"], xyz, shape),
        "RD": volume_from_masked(scalars["RD"], xyz, shape),
        "V1": volume_from_masked(scalars["V1"], xyz, shape, n_comp=3),
    }


@torch.no_grad()
def eval_signal_metrics(
    model: torch.nn.Module,
    data: SubjectData,
    device: torch.device,
    b_scale: float,
    chunk: int = 4096,
) -> dict[str, float]:
    model.eval()
    g = torch.from_numpy(data.bvecs).to(device)
    b = torch.from_numpy(data.bvals / b_scale).to(device)
    bvals = torch.from_numpy(data.bvals).to(device)
    bvecs = torch.from_numpy(data.bvecs).to(device)
    sig = masked_signals(data)
    coords = torch.from_numpy(data.coords_norm).to(device)
    sig_t = torch.from_numpy(sig).to(device)
    preds = []
    for i in range(0, coords.shape[0], chunk):
        sl = slice(i, i + chunk)
        S0, D = _forward_s0_d(
            model,
            coords[sl],
            s=sig_t[sl] if getattr(model, "uses_q_feature", False) else None,
            bvals=bvals if getattr(model, "uses_q_feature", False) else None,
            bvecs=bvecs if getattr(model, "uses_q_feature", False) else None,
        )
        preds.append(predict_signals_multi(S0, D, g, b).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return {
        "Signal_MSE": signal_mse(pred, sig),
        "Signal_MAE": signal_mae(pred, sig),
        "PSNR": signal_psnr(pred, sig, peak=float(np.percentile(sig, 99))),
    }


def save_nifti_maps(maps: dict[str, np.ndarray], affine: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("FA", "MD", "AD", "RD", "V1", "S0"):
        nib.save(nib.Nifti1Image(maps[key].astype(np.float32), affine), str(out_dir / f"{key}.nii.gz"))


def format_metrics(dti: dict[str, float], sig: dict[str, float]) -> dict[str, float]:
    """Map internal names to the requested metric keys."""
    return {
        "FA_MAE": dti["FA_MAE"],
        "FA_RMSE": dti["FA_RMSE"],
        "MD_MAE": dti["MD_MAE"],
        "MD_RMSE": dti["MD_RMSE"],
        "AD_MAE": dti["AD_MAE"],
        "AD_RMSE": dti["AD_RMSE"],
        "RD_MAE": dti["RD_MAE"],
        "RD_RMSE": dti["RD_RMSE"],
        "V1_angular_error": dti["V1_ang_mean_deg"],
        "V1_angular_error_median": dti["V1_ang_median_deg"],
        "Signal_MSE": sig["Signal_MSE"],
        "Signal_MAE": sig["Signal_MAE"],
        "PSNR": sig["PSNR"],
    }


def train_one_subject(
    config: dict,
    subject_id: str,
    run_dir: Path,
    device: torch.device,
) -> dict:
    set_seed(int(config["training"]["seed"]))
    b_scale = float(config["training"].get("b_scale", 1000.0))
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["learning_rate"])
    val_frac = float(config["training"].get("val_fraction", 0.1))
    early_stop = bool(config["training"].get("early_stop", True))
    patience = int(config["training"].get("early_stop_patience", 25))
    min_delta = float(config["training"].get("early_stop_min_delta", 1e-6))

    subj_dir = run_dir / f"hcp_{subject_id}"
    maps_dir = subj_dir / "maps"
    subj_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(exist_ok=True)

    print(f"\n========== Subject {subject_id} ==========")
    data = load_hcp_subject(
        hcp_root=config["dataset"]["hcp_root"],
        subject_id=subject_id,
        shells=config["sampling"].get("shells", [1000.0]),
    )
    print(
        f"[data] shape={data.dwi.shape}, voxels={data.coords_xyz.shape[0]}, "
        f"vols={data.bvals.shape[0]}, scale={data.signal_scale:.2f}"
    )

    signals = masked_signals(data)
    n_vox = signals.shape[0]
    rng = np.random.default_rng(int(config["training"]["seed"]))
    perm = rng.permutation(n_vox)
    n_val = max(1, int(round(val_frac * n_vox)))
    idx_val = perm[:n_val]
    idx_train = perm[n_val:]

    g = torch.from_numpy(data.bvecs).to(device)
    b = torch.from_numpy(data.bvals / b_scale).to(device)
    bvals_t = torch.from_numpy(data.bvals).to(device)
    bvecs_t = torch.from_numpy(data.bvecs).to(device)
    signals_t = torch.from_numpy(signals.astype(np.float32))
    coords_t = torch.from_numpy(data.coords_norm.astype(np.float32))

    model = build_model(config, device)
    backend = "n/a"
    if getattr(model, "hashgrid", None) is not None:
        backend = getattr(model.hashgrid, "backend", "n/a")
    print(f"[model] {type(model).__name__}, hash_backend={backend}, device={device}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # OOM-safe batch
    while batch_size > 256:
        try:
            idx = torch.arange(min(batch_size, idx_train.shape[0]))
            x = coords_t[idx].to(device)
            s = signals_t[idx].to(device)
            S0, D = _forward_s0_d(model, x, s=s, bvals=bvals_t, bvecs=bvecs_t)
            _ = predict_signals_multi(S0, D, g, b)
            del S0, D, _
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            batch_size //= 2
            print(f"[train] OOM — batch_size -> {batch_size}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    loss_history: list[float] = []
    val_history: list[float] = []
    best_val = float("inf")
    best_state: dict | None = None
    stale = 0

    use_q = getattr(model, "uses_q_feature", False)

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
        loss_history.append(epoch_loss)

        # Validation
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
        val_history.append(val_loss)

        improved = val_loss < (best_val - min_delta)
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"[train] {subject_id} epoch {epoch:4d}/{epochs}  "
                f"train_MSE={epoch_loss:.6e}  val_MSE={val_loss:.6e}  best={best_val:.6e}"
            )

        if early_stop and stale >= patience and epoch >= 40:
            print(f"[train] early stop at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "subject_id": subject_id,
            "signal_scale": data.signal_scale,
            "b_scale": b_scale,
            "loss_history": loss_history,
            "val_history": val_history,
            "best_val_MSE": best_val,
            "batch_size_used": batch_size,
        },
        subj_dir / "checkpoint.pt",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history, label="train")
    ax.plot(val_history, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("signal MSE")
    ax.set_title(f"{subject_id}: SpatialDTIParamField")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(subj_dir / "loss_curve.png", dpi=140)
    plt.close(fig)

    print("[eval] Inferring INR maps ...")
    inr_maps = infer_dti_maps(model, data, device, b_scale=b_scale)

    print("[eval] Fitting WLS-DTI reference ...")
    wls = fit_wls_dti(data.dwi, data.bvals, data.bvecs, mask=data.mask)
    wls_maps = {
        "FA": wls["FA"].astype(np.float32),
        "MD": wls["MD"].astype(np.float32),
        "AD": wls["AD"].astype(np.float32),
        "RD": wls["RD"].astype(np.float32),
        "V1": wls["V1"].astype(np.float32),
        "S0": wls["S0"].astype(np.float32),
    }

    dti_raw = compare_dti_maps(inr_maps, wls_maps, data.mask)
    sig_raw = eval_signal_metrics(model, data, device, b_scale=b_scale)
    metrics = format_metrics(dti_raw, sig_raw)

    save_nifti_maps(inr_maps, data.affine, maps_dir / "inr")
    save_nifti_maps(wls_maps, data.affine, maps_dir / "wls_reference")

    slice_z = save_comparison_figure(
        wls_maps,
        inr_maps,
        data.mask,
        subj_dir / "comparison.png",
        title=f"Subject {subject_id}: WLS | INR | Error  (SpatialDTIParamField)",
    )

    payload = {
        "note": "WLS-DTI is used as a reference, not as training supervision.",
        "subject_id": subject_id,
        "model": type(model).__name__,
        "hash_backend": backend,
        "device": str(device),
        "batch_size_used": batch_size,
        "epochs_run": len(loss_history),
        "final_train_MSE": loss_history[-1] if loss_history else None,
        "best_val_MSE": best_val if best_val < float("inf") else None,
        "comparison_slice_z": slice_z,
        **metrics,
    }
    with open(subj_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(subj_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in payload.items():
            writer.writerow([k, v])

    print("[metrics]", json.dumps(metrics, indent=2))
    return payload


METRIC_KEYS = [
    "FA_MAE",
    "FA_RMSE",
    "MD_MAE",
    "MD_RMSE",
    "AD_MAE",
    "AD_RMSE",
    "RD_MAE",
    "RD_RMSE",
    "V1_angular_error",
    "Signal_MSE",
    "Signal_MAE",
    "PSNR",
]


def _load_existing_metrics(run_dir: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for path in sorted(run_dir.glob("hcp_*/metrics.json")):
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        sid = str(row.get("subject_id") or path.parent.name.replace("hcp_", ""))
        if "error" not in row and all(k in row for k in METRIC_KEYS):
            found[sid] = row
    return found


def _write_summary(run_dir: Path, rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r and all(k in r for k in METRIC_KEYS)]
    summary: dict = {"run_dir": str(run_dir), "n_subjects": len(ok), "subjects": rows}
    if ok:
        summary["MEAN"] = {k: float(np.mean([r[k] for r in ok])) for k in METRIC_KEYS}
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", *METRIC_KEYS])
        for r in ok:
            writer.writerow([r["subject_id"], *[r[k] for k in METRIC_KEYS]])
        if ok:
            writer.writerow(["MEAN", *[summary["MEAN"][k] for k in METRIC_KEYS]])
    return summary


def train(config: dict, run_dir: Path | None = None, skip_done: bool = True) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / "experiments" / "v2_dti_param_field" / stamp
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    subjects = [str(s) for s in config["dataset"]["subjects"]]
    existing = _load_existing_metrics(run_dir) if skip_done else {}
    print(f"[run] {run_dir}")
    print(f"[run] subjects={subjects}, device={device}")
    if existing:
        print(f"[run] already done: {sorted(existing)}")

    rows_by_id: dict[str, dict] = dict(existing)
    for sid in subjects:
        if sid in rows_by_id:
            print(f"[skip] subject {sid} already has metrics")
            continue
        try:
            rows_by_id[sid] = train_one_subject(config, sid, run_dir, device)
        except Exception as exc:
            print(f"[error] subject {sid} failed: {exc}")
            rows_by_id[sid] = {"subject_id": sid, "error": str(exc)}
        # Persist partial summary after each subject
        ordered = [rows_by_id[s] for s in subjects if s in rows_by_id]
        _write_summary(run_dir, ordered)

    ordered = [rows_by_id.get(s, {"subject_id": s, "error": "missing"}) for s in subjects]
    # Also include any completed subjects not in the requested list (resume safety)
    for sid, row in rows_by_id.items():
        if sid not in subjects:
            ordered.append(row)
    summary = _write_summary(run_dir, ordered)

    print("\n[done]", run_dir)
    if "MEAN" in summary:
        print("[MEAN]", json.dumps(summary["MEAN"], indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train version-two SpatialDTIParamField (DTI only) on HCP subjects"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "v2_dti_param_field.yaml"),
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="*",
        default=None,
        help="Override subject list from config",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Resume / write into an existing experiment folder",
    )
    parser.add_argument(
        "--no-skip-done",
        action="store_true",
        help="Retrain subjects even if metrics.json already exists",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.subjects:
        config["dataset"]["subjects"] = list(args.subjects)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    run_dir = Path(args.run_dir) if args.run_dir else None
    train(config, run_dir=run_dir, skip_done=not args.no_skip_done)


if __name__ == "__main__":
    main()
