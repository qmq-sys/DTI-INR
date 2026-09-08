"""Train Spatial DTI-INR (Model A) on one HCP-YA subject."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

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
from models.spatial_dti_inr import SpatialDTIINR  # noqa: E402
from visualization.parameter_maps import save_comparison_figure  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_signals_multi(
    S0: Tensor,
    D: Tensor,
    g: Tensor,
    b: Tensor,
) -> Tensor:
    """Batched DTI forward.

    Args:
        S0: [B]
        D:  [B, 3, 3]
        g:  [N, 3]
        b:  [N]   (already / b_scale if used)

    Returns:
        S: [B, N]
    """
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


@torch.no_grad()
def infer_dti_maps(
    model: SpatialDTIINR,
    data: SubjectData,
    device: torch.device,
    b_scale: float,
    chunk: int = 65536,
) -> dict[str, np.ndarray]:
    model.eval()
    coords = torch.from_numpy(data.coords_norm).to(device)
    S0_list = []
    D_list = []
    for i in range(0, coords.shape[0], chunk):
        S0, D = model(coords[i : i + chunk])
        S0_list.append(S0.cpu().numpy())
        # Convert training D units -> mm^2/s
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
    model: SpatialDTIINR,
    data: SubjectData,
    device: torch.device,
    b_scale: float,
    chunk: int = 4096,
) -> dict[str, float]:
    model.eval()
    g = torch.from_numpy(data.bvecs).to(device)
    b = torch.from_numpy(data.bvals / b_scale).to(device)
    sig = masked_signals(data)
    coords = torch.from_numpy(data.coords_norm).to(device)
    preds = []
    for i in range(0, coords.shape[0], chunk):
        S0, D = model(coords[i : i + chunk])
        S = predict_signals_multi(S0, D, g, b)
        preds.append(S.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return {
        "signal_MSE": signal_mse(pred, sig),
        "signal_MAE": signal_mae(pred, sig),
        "signal_PSNR": signal_psnr(pred, sig, peak=float(np.percentile(sig, 99))),
    }


def save_nifti_maps(
    maps: dict[str, np.ndarray],
    affine: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("FA", "MD", "AD", "RD", "V1", "S0"):
        path = out_dir / f"{key}.nii.gz"
        nib.save(nib.Nifti1Image(maps[key].astype(np.float32), affine), str(path))


def train(config: dict) -> Path:
    set_seed(int(config["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subject_id = str(config["dataset"]["subjects"][0])
    hcp_root = config["dataset"]["hcp_root"]
    b_scale = float(config["training"].get("b_scale", 1000.0))
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["learning_rate"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = ROOT / "experiments" / "spatial_dti_inr" / stamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = exp_dir / "maps"
    maps_dir.mkdir(exist_ok=True)

    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"[data] Loading subject {subject_id} ...")
    data = load_hcp_subject(
        hcp_root=hcp_root,
        subject_id=subject_id,
        shells=config["sampling"].get("shells", [1000.0]),
    )
    print(
        f"[data] shape={data.dwi.shape}, masked_voxels={data.coords_xyz.shape[0]}, "
        f"n_vol={data.bvals.shape[0]}, signal_scale={data.signal_scale:.2f}"
    )

    signals = masked_signals(data)  # [V,N]
    coords_np = data.coords_norm
    g = torch.from_numpy(data.bvecs).to(device)
    b = torch.from_numpy(data.bvals / b_scale).to(device)
    signals_t = torch.from_numpy(signals.astype(np.float32))
    coords_t = torch.from_numpy(coords_np.astype(np.float32))

    model = SpatialDTIINR(
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        num_frequencies=int(config["model"]["num_frequencies"]),
        activation=str(config["model"].get("activation", "gelu")),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_vox = coords_t.shape[0]
    # Auto-reduce batch if OOM
    while batch_size > 512:
        try:
            idx = torch.arange(min(batch_size, n_vox))
            S0, D = model(coords_t[idx].to(device))
            _ = predict_signals_multi(S0, D, g, b)
            del S0, D, _
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            batch_size //= 2
            print(f"[train] OOM — reducing batch_size to {batch_size}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    loss_history: list[float] = []
    print(f"[train] device={device}, batch_size={batch_size}, epochs={epochs}")

    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_vox)
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, n_vox, batch_size):
            idx = perm[start : start + batch_size]
            x = coords_t[idx].to(device)
            y = signals_t[idx].to(device)
            S0, D = model(x)
            pred = predict_signals_multi(S0, D, g, b)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_steps += 1
        epoch_loss /= max(n_steps, 1)
        loss_history.append(epoch_loss)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"[train] epoch {epoch:4d}/{epochs}  MSE={epoch_loss:.6e}")

    ckpt = {
        "model_state": model.state_dict(),
        "config": config,
        "subject_id": subject_id,
        "signal_scale": data.signal_scale,
        "b_scale": b_scale,
        "loss_history": loss_history,
    }
    torch.save(ckpt, exp_dir / "checkpoint.pt")

    # Loss curve
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("signal MSE")
    ax.set_title("Spatial DTI-INR training loss")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(exp_dir / "loss_curve.png", dpi=140)
    plt.close(fig)

    print("[eval] Inferring INR DTI maps ...")
    inr_maps = infer_dti_maps(model, data, device, b_scale=b_scale)

    print("[eval] Fitting WLS-DTI reference (not used as supervision) ...")
    wls = fit_wls_dti(data.dwi, data.bvals, data.bvecs, mask=data.mask)
    wls_maps = {
        "FA": wls["FA"].astype(np.float32),
        "MD": wls["MD"].astype(np.float32),
        "AD": wls["AD"].astype(np.float32),
        "RD": wls["RD"].astype(np.float32),
        "V1": wls["V1"].astype(np.float32),
        "S0": wls["S0"].astype(np.float32),
    }

    dti_metrics = compare_dti_maps(inr_maps, wls_maps, data.mask)
    sig_metrics = eval_signal_metrics(model, data, device, b_scale=b_scale)

    metrics = {
        "note": "WLS-DTI is used as a reference, not as training supervision.",
        "subject_id": subject_id,
        "final_train_MSE": loss_history[-1] if loss_history else None,
        "device": str(device),
        "batch_size_used": batch_size,
        **{f"DTI/{k}": v for k, v in dti_metrics.items()},
        **{f"signal/{k}": v for k, v in sig_metrics.items()},
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(exp_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

    save_nifti_maps(inr_maps, data.affine, maps_dir)
    # also store WLS reference maps
    wls_dir = maps_dir / "wls_reference"
    save_nifti_maps(wls_maps, data.affine, wls_dir)

    slice_z = save_comparison_figure(
        wls_maps,
        inr_maps,
        data.mask,
        exp_dir / "comparison.png",
        title=f"Subject {subject_id}: WLS vs Spatial DTI-INR",
    )
    metrics["comparison_slice_z"] = slice_z
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[done]", exp_dir)
    print(json.dumps({k: metrics[k] for k in metrics if k.startswith("DTI/") or k.startswith("signal/") or k == "final_train_MSE"}, indent=2))
    return exp_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spatial DTI-INR on HCP-YA")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "a_spatial.yaml"),
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
