"""Compare two v2 DTI-INR runs (e.g. no-q vs with-q) using summary.csv / MEAN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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

# Lower is better except PSNR
HIGHER_BETTER = {"PSNR"}


def _load_mean(run_dir: Path) -> dict[str, float]:
    sj = run_dir / "summary.json"
    if sj.is_file():
        data = json.loads(sj.read_text(encoding="utf-8"))
        if "MEAN" in data:
            return {k: float(data["MEAN"][k]) for k in METRIC_KEYS if k in data["MEAN"]}
    sc = run_dir / "summary.csv"
    if not sc.is_file():
        raise FileNotFoundError(f"No summary.json/csv in {run_dir}")
    with open(sc, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mean_row = next((r for r in rows if r.get("subject_id") == "MEAN"), None)
    if mean_row is None:
        raise RuntimeError(f"No MEAN row in {sc}")
    return {k: float(mean_row[k]) for k in METRIC_KEYS}


def compare(run_a: Path, run_b: Path, label_a: str, label_b: str) -> None:
    a = _load_mean(run_a)
    b = _load_mean(run_b)
    print(f"{'metric':<22} {label_a:>14} {label_b:>14} {'delta(B-A)':>14} {'better':>10}")
    print("-" * 78)
    for k in METRIC_KEYS:
        va, vb = a[k], b[k]
        d = vb - va
        if k in HIGHER_BETTER:
            win = label_b if vb > va else (label_a if va > vb else "tie")
        else:
            win = label_b if vb < va else (label_a if va < vb else "tie")
        print(f"{k:<22} {va:14.6g} {vb:14.6g} {d:14.6g} {win:>10}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two v2_dti_param_field run MEANs")
    p.add_argument("--run-a", type=str, required=True, help="Baseline run dir (e.g. no q_feature)")
    p.add_argument("--run-b", type=str, required=True, help="New run dir (e.g. with q_feature)")
    p.add_argument("--label-a", type=str, default="no_q")
    p.add_argument("--label-b", type=str, default="with_q")
    args = p.parse_args()
    compare(Path(args.run_a), Path(args.run_b), args.label_a, args.label_b)


if __name__ == "__main__":
    main()
