"""Simple report figure: FA | MD | AD | RD | V1 from Full Reference maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(subject_id: str = "109830", out: str | None = None) -> Path:
    ref = ROOT / "reference" / subject_id
    fa = np.asanyarray(nib.load(str(ref / "FA_ref.nii.gz")).dataobj).astype(np.float32)
    md = np.asanyarray(nib.load(str(ref / "MD_ref.nii.gz")).dataobj).astype(np.float32)
    ad = np.asanyarray(nib.load(str(ref / "AD_ref.nii.gz")).dataobj).astype(np.float32)
    rd = np.asanyarray(nib.load(str(ref / "RD_ref.nii.gz")).dataobj).astype(np.float32)
    v1 = np.asanyarray(nib.load(str(ref / "V1_ref.nii.gz")).dataobj).astype(np.float32)

    z = int(np.argmax(np.nansum(fa, axis=(0, 1))))

    def axial(vol: np.ndarray) -> np.ndarray:
        return np.rot90(vol[:, :, z])

    fa_s, md_s, ad_s, rd_s = axial(fa), axial(md), axial(ad), axial(rd)
    v1_s = np.rot90(v1[:, :, z, :])
    mask = fa_s > 1e-6
    rgb = np.clip(np.abs(v1_s) / (np.max(np.abs(v1_s)) + 1e-8) * fa_s[..., None], 0, 1)
    rgb[~mask] = 0

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), facecolor="#e8eef5")
    panels = [
        ("FA", fa_s, "jet", float(np.percentile(fa_s[mask], 99))),
        ("MD", md_s, "turbo", float(np.percentile(md_s[mask], 98))),
        ("AD", ad_s, "hot", float(np.percentile(ad_s[mask], 98))),
        ("RD", rd_s, "turbo", float(np.percentile(rd_s[mask], 98))),
    ]
    for ax, (lab, img, cmap, vmax) in zip(axes[:4], panels):
        ax.set_facecolor("black")
        ax.imshow(np.where(mask, img, np.nan), cmap=cmap, vmin=0, vmax=max(vmax, 1e-8))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(lab, fontsize=14, labelpad=8)

    axes[4].set_facecolor("black")
    axes[4].imshow(rgb)
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    for s in axes[4].spines.values():
        s.set_visible(False)
    axes[4].set_xlabel("V1", fontsize=14, labelpad=8)

    fig.tight_layout(pad=0.6)
    out_path = Path(out) if out else ref / "dti_maps_overview.png"
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path} (axial z={z})")
    return out_path


if __name__ == "__main__":
    main()
