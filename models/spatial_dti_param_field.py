"""Version-two DTI parameter-field architecture (DTI only; no DKI).

Ported from ``version two/models_dti.py``:

* ``HashEncoding`` — multi-resolution hash grid (tiny-cuda-nn if available,
  otherwise a pure-PyTorch multi-resolution hash grid)
* ``SpatialDTIParamField`` — xyz -> (S0, D) via Cholesky
* ``SpatialDTIParamFieldWithQ`` — q-space Transformer + spatial MLP -> (S0, D)

All DKI / joint DTI+DKI modules are intentionally excluded.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hash-grid encoding
# ---------------------------------------------------------------------------


class _CudaSdpMathOnly:
    """Prefer math SDP for MultiheadAttention on CUDA (stability)."""

    def __init__(self, active: bool) -> None:
        self._active = bool(active)
        self._flash: bool | None = None
        self._mem: bool | None = None
        self._math: bool | None = None

    def __enter__(self) -> "_CudaSdpMathOnly":
        if not self._active:
            return self
        try:
            self._flash = torch.backends.cuda.flash_sdp_enabled()
            self._mem = torch.backends.cuda.mem_efficient_sdp_enabled()
            self._math = torch.backends.cuda.math_sdp_enabled()
        except Exception:
            self._active = False
            return self
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._flash is None:
            return
        torch.backends.cuda.enable_flash_sdp(self._flash)
        torch.backends.cuda.enable_mem_efficient_sdp(self._mem)  # type: ignore[arg-type]
        torch.backends.cuda.enable_math_sdp(self._math)  # type: ignore[arg-type]


class _TorchHashEncoding(nn.Module):
    """Pure-PyTorch multi-resolution hash encoding (Instant-NGP style).

    Used when tiny-cuda-nn is unavailable. Coordinates in [-1, 1]^3.
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 1024,
        per_level_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.hashmap_size = 1 << self.log2_hashmap_size
        self.base_resolution = int(base_resolution)
        if per_level_scale is None:
            per_level_scale = float(
                np.exp(
                    (np.log(max_resolution) - np.log(base_resolution))
                    / max(1, self.n_levels - 1)
                )
            )
        self.per_level_scale = float(per_level_scale)
        self._out_dim = self.n_levels * self.n_features_per_level
        self._active_level_fraction = 1.0

        tables = []
        for _ in range(self.n_levels):
            t = nn.Embedding(self.hashmap_size, self.n_features_per_level)
            nn.init.uniform_(t.weight, -1e-4, 1e-4)
            tables.append(t)
        self.tables = nn.ModuleList(tables)

        # Instant-NGP spatial hash primes
        self.register_buffer(
            "_primes",
            torch.tensor([1, 2654435761, 805459861], dtype=torch.int64),
            persistent=False,
        )

    @property
    def out_dim(self) -> int:
        return int(self._out_dim)

    def set_active_level_fraction(self, alpha: float) -> None:
        self._active_level_fraction = float(max(0.0, min(1.0, alpha)))

    def _hash_corners(self, corners: torch.Tensor) -> torch.Tensor:
        # corners: [..., 3] int64
        x = corners[..., 0] * self._primes[0]
        x = x ^ (corners[..., 1] * self._primes[1])
        x = x ^ (corners[..., 2] * self._primes[2])
        return (x & (self.hashmap_size - 1)).long()

    def _encode_level(self, xyz01: torch.Tensor, level: int) -> torch.Tensor:
        res = self.base_resolution * (self.per_level_scale**level)
        scaled = xyz01 * res
        scale_floor = torch.floor(scaled)
        frac = scaled - scale_floor
        # 8 cube corners
        offsets = xyz01.new_tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            dtype=torch.float32,
        )
        corners = (scale_floor.unsqueeze(-2) + offsets).to(torch.int64)  # [V,8,3]
        idx = self._hash_corners(corners)  # [V,8]
        feats = self.tables[level](idx)  # [V,8,F]

        fx, fy, fz = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]
        w = torch.cat(
            [
                (1 - fx) * (1 - fy) * (1 - fz),
                fx * (1 - fy) * (1 - fz),
                (1 - fx) * fy * (1 - fz),
                fx * fy * (1 - fz),
                (1 - fx) * (1 - fy) * fz,
                fx * (1 - fy) * fz,
                (1 - fx) * fy * fz,
                fx * fy * fz,
            ],
            dim=-1,
        ).unsqueeze(-1)  # [V,8,1]
        return (feats * w).sum(dim=-2)  # [V,F]

    def forward(self, xyz_m11: torch.Tensor) -> torch.Tensor:
        xyz01 = ((xyz_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
        feats = [self._encode_level(xyz01, lv) for lv in range(self.n_levels)]
        out = torch.cat(feats, dim=-1)
        alpha = float(self._active_level_fraction)
        if alpha < 1.0 - 1e-6:
            n_act = int(np.ceil(alpha * self.n_levels))
            n_act = max(1, min(self.n_levels, n_act))
            dim_act = n_act * self.n_features_per_level
            mask = torch.zeros_like(out)
            mask[..., :dim_act] = 1.0
            out = out * mask
        return out


class HashEncoding(nn.Module):
    """Multi-resolution hash encoding.

    Prefers tiny-cuda-nn on CUDA; falls back to a pure-PyTorch implementation.
    Input coordinates must lie in [-1, 1]^3.
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 1024,
        per_level_scale: float | None = None,
        force_torch: bool = False,
    ) -> None:
        super().__init__()
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self._active_level_fraction = 1.0
        self.backend = "torch"

        use_tcnn = False
        if not force_torch and torch.cuda.is_available():
            try:
                import tinycudann as tcnn  # type: ignore

                if per_level_scale is None:
                    per_level_scale = float(
                        np.exp(
                            (np.log(max_resolution) - np.log(base_resolution))
                            / max(1, int(n_levels) - 1)
                        )
                    )
                enc = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": int(n_levels),
                        "n_features_per_level": int(n_features_per_level),
                        "log2_hashmap_size": int(log2_hashmap_size),
                        "base_resolution": int(base_resolution),
                        "per_level_scale": float(per_level_scale),
                        "interpolation": "Linear",
                    },
                )
                self.enc = enc
                self._out_dim = int(enc.n_output_dims)
                self.backend = "tinycudann"
                use_tcnn = True
            except Exception:
                use_tcnn = False

        if not use_tcnn:
            self.enc = _TorchHashEncoding(
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=min(int(log2_hashmap_size), 17),  # memory-friendly CPU/GPU torch
                base_resolution=base_resolution,
                max_resolution=max_resolution,
                per_level_scale=per_level_scale,
            )
            self._out_dim = int(self.enc.out_dim)

    @property
    def out_dim(self) -> int:
        return int(self._out_dim)

    def set_active_level_fraction(self, alpha: float) -> None:
        self._active_level_fraction = float(max(0.0, min(1.0, alpha)))
        if hasattr(self.enc, "set_active_level_fraction"):
            self.enc.set_active_level_fraction(alpha)

    def forward(self, xyz_m11: torch.Tensor) -> torch.Tensor:
        if self.backend == "tinycudann":
            xyz01 = ((xyz_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
            feat = self.enc(xyz01)
            alpha = float(self._active_level_fraction)
            if alpha < 1.0 - 1e-6:
                n_act = int(np.ceil(alpha * self.n_levels))
                n_act = max(1, min(self.n_levels, n_act))
                dim_act = n_act * self.n_features_per_level
                mask = torch.zeros_like(feat)
                mask[..., :dim_act] = 1.0
                feat = feat * mask
            return feat
        return self.enc(xyz_m11)


# ---------------------------------------------------------------------------
# Helpers: Cholesky DTI head
# ---------------------------------------------------------------------------


def _params_to_s0_d(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map MLP output [V,7] -> S0 [V,1], D [V,3,3] (exp-Cholesky, version-two style)."""
    logS0 = params[:, 0:1]
    L_raw = params[:, 1:]
    S0 = torch.exp(logS0)
    L11 = torch.exp(L_raw[:, 0])
    L22 = torch.exp(L_raw[:, 1])
    L33 = torch.exp(L_raw[:, 2])
    L21 = L_raw[:, 3]
    L31 = L_raw[:, 4]
    L32 = L_raw[:, 5]
    zeros = torch.zeros_like(L11)
    row1 = torch.stack([L11, zeros, zeros], dim=-1)
    row2 = torch.stack([L21, L22, zeros], dim=-1)
    row3 = torch.stack([L31, L32, L33], dim=-1)
    Lmat = torch.stack([row1, row2, row3], dim=-2)
    D = Lmat @ Lmat.transpose(-1, -2)
    return S0, D


def _build_mlp(in_dim: int, hidden: int, layers: int, out_dim: int) -> nn.Sequential:
    net: list[nn.Module] = []
    last = in_dim
    for _ in range(int(layers)):
        net.append(nn.Linear(last, hidden))
        net.append(nn.ReLU(inplace=True))
        last = hidden
    net.append(nn.Linear(last, out_dim))
    return nn.Sequential(*net)


# ---------------------------------------------------------------------------
# Pure spatial DTI parameter field
# ---------------------------------------------------------------------------


class SpatialDTIParamField(nn.Module):
    """Spatial DTI-INR: HashGrid(xyz) -> MLP -> (S0, D).

    Inputs:
        xyz_norm [V, 3] in [-1, 1]
    Outputs:
        S0 [V, 1], D [V, 3, 3]
    """

    uses_q_feature = False

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 4,
        use_hashgrid: bool = True,
        hashgrid_concat_xyz: bool = True,
        hash_encoding_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.use_hashgrid = bool(use_hashgrid)
        self.hashgrid_concat_xyz = bool(hashgrid_concat_xyz)
        if self.use_hashgrid:
            he_kw = dict(hash_encoding_kwargs or {})
            self.hashgrid = HashEncoding(**he_kw)
            in_dim = self.hashgrid.out_dim + (3 if self.hashgrid_concat_xyz else 0)
        else:
            self.hashgrid = None
            in_dim = 3
        self.mlp = _build_mlp(in_dim, hidden, layers, out_dim=7)

    def forward(self, xyz_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_hashgrid:
            feats = self.hashgrid(xyz_norm)
            if self.hashgrid_concat_xyz:
                feats = torch.cat([feats, xyz_norm], dim=-1)
            params = self.mlp(feats)
        else:
            params = self.mlp(xyz_norm)
        return _params_to_s0_d(params)


# ---------------------------------------------------------------------------
# Q-space encoder + SpatialDTIParamFieldWithQ
# ---------------------------------------------------------------------------


class QSpaceTransformerEncoder(nn.Module):
    """aqDL-like q-space Transformer encoder (direction tokens).

    Returns per-direction states ``[B, N, embed_dim]`` (CLS discarded).
    ``q_sh_lmax`` must be 0 in this DTI-only port (no SH dependency).
    """

    def __init__(
        self,
        *,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        shell_tol: float = 200.0,
        eps: float = 1e-12,
        q_sh_lmax: int = 0,
        q_sh_include_bvec: bool = False,
        b0_threshold: float = 50.0,
    ) -> None:
        super().__init__()
        if int(q_sh_lmax) != 0:
            raise ValueError(
                "This DTI-only port only supports q_sh_lmax=0 "
                "(SH basis from version-two DKI path is excluded)."
            )
        self.embed_dim = int(embed_dim)
        self.shell_tol = float(shell_tol)
        self.eps = float(eps)
        self.shells = (0.0, 1000.0, 2000.0)
        self.q_sh_lmax = 0
        self.q_sh_include_bvec = bool(q_sh_include_bvec)
        self.b0_threshold = float(b0_threshold)
        dir_dim = 3
        in_dim = 1 + len(self.shells) + dir_dim
        self.proj = nn.Linear(in_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(n_heads),
            dim_feedforward=int(ff_mult) * self.embed_dim,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

    def _expand_q_inputs(
        self, s: torch.Tensor, bvals: torch.Tensor, bvecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bvals.ndim == 1:
            bvals = bvals.unsqueeze(0).expand(s.shape[0], -1)
        if bvecs.ndim == 2:
            bvecs = bvecs.unsqueeze(0).expand(s.shape[0], -1, -1)
        return s, bvals, bvecs

    def _onehot_shells(self, bvals: torch.Tensor) -> torch.Tensor:
        shells = torch.tensor(self.shells, device=bvals.device, dtype=bvals.dtype).view(1, 1, -1)
        dist = torch.abs(bvals.unsqueeze(-1) - shells)
        idx = torch.argmin(dist, dim=-1)
        within = torch.amin(dist, dim=-1) <= self.shell_tol
        oh = F.one_hot(idx, num_classes=len(self.shells)).to(dtype=bvals.dtype)
        return oh * within.unsqueeze(-1).to(dtype=bvals.dtype)

    def forward(self, s: torch.Tensor, bvals: torch.Tensor, bvecs: torch.Tensor) -> torch.Tensor:
        s, bvals, bvecs = self._expand_q_inputs(s, bvals, bvecs)
        B = int(s.shape[0])
        ln_s = torch.log(torch.clamp(s, min=self.eps)).unsqueeze(-1)
        oh_b = self._onehot_shells(bvals)
        dir_feat = bvecs.to(dtype=ln_s.dtype)
        feats = torch.cat([ln_s, oh_b, dir_feat], dim=-1)
        tok = self.proj(feats)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tok], dim=1)
        sdp_ctx: _CudaSdpMathOnly | nullcontext = (
            _CudaSdpMathOnly(True) if x.is_cuda else nullcontext()
        )
        with sdp_ctx:
            x = self.encoder(x)
        return x[:, 1:, :]


class SpatialDTIParamFieldWithQ(nn.Module):
    """DTI parameter field with q-space features (no DKI).

    Stage 1: q-space Transformer -> q-feature
    Stage 2: concat(HashGrid(xyz), xyz, q-feature) -> MLP -> (S0, D)
    """

    uses_q_feature = True

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 4,
        use_hashgrid: bool = True,
        hashgrid_concat_xyz: bool = True,
        hash_encoding_kwargs: dict | None = None,
        q_embed_dim: int = 32,
        q_encode_chunk_size: int = 8192,
        q_sh_lmax: int = 0,
        q_sh_include_bvec: bool = False,
        b0_threshold: float = 50.0,
    ) -> None:
        super().__init__()
        self.use_hashgrid = bool(use_hashgrid)
        self.hashgrid_concat_xyz = bool(hashgrid_concat_xyz)
        if self.use_hashgrid:
            he_kw = dict(hash_encoding_kwargs or {})
            self.hashgrid = HashEncoding(**he_kw)
            xyz_dim = self.hashgrid.out_dim + (3 if self.hashgrid_concat_xyz else 0)
        else:
            self.hashgrid = None
            xyz_dim = 3

        self.q_encoder = QSpaceTransformerEncoder(
            embed_dim=int(q_embed_dim),
            q_sh_lmax=int(q_sh_lmax),
            q_sh_include_bvec=bool(q_sh_include_bvec),
            b0_threshold=float(b0_threshold),
        )
        self.q_null = nn.Parameter(torch.zeros(int(q_embed_dim)))
        self.q_encode_chunk_size = int(max(1, q_encode_chunk_size))
        self.mlp = _build_mlp(xyz_dim + int(q_embed_dim), hidden, layers, out_dim=7)

    def encode_q(self, s: torch.Tensor, bvals: torch.Tensor, bvecs: torch.Tensor) -> torch.Tensor:
        if s.ndim == 2 and int(s.shape[0]) > self.q_encode_chunk_size:
            outs: list[torch.Tensor] = []
            step = int(self.q_encode_chunk_size)
            for i in range(0, int(s.shape[0]), step):
                outs.append(self.q_encoder(s[i : i + step], bvals, bvecs))
            return torch.cat(outs, dim=0)
        return self.q_encoder(s, bvals, bvecs)

    def forward_with_qfeat(
        self, xyz_norm: torch.Tensor, q_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_hashgrid:
            feats = self.hashgrid(xyz_norm)
            if self.hashgrid_concat_xyz:
                feats = torch.cat([feats, xyz_norm], dim=-1)
        else:
            feats = xyz_norm
        if q_feat.ndim == 1:
            q_feat = q_feat.unsqueeze(0).expand(xyz_norm.shape[0], -1)
        if q_feat.ndim == 3:
            q_feat = q_feat.mean(dim=1)
        params = self.mlp(torch.cat([feats, q_feat], dim=-1))
        return _params_to_s0_d(params)

    def forward(
        self,
        xyz_norm: torch.Tensor,
        *,
        s: torch.Tensor | None = None,
        bvals: torch.Tensor | None = None,
        bvecs: torch.Tensor | None = None,
        q_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q_feat is None and (s is not None) and (bvals is not None) and (bvecs is not None):
            q_feat = self.encode_q(s, bvals, bvecs)
        if q_feat is None:
            q_feat = self.q_null
        return self.forward_with_qfeat(xyz_norm, q_feat)


__all__ = [
    "HashEncoding",
    "QSpaceTransformerEncoder",
    "SpatialDTIParamField",
    "SpatialDTIParamFieldWithQ",
]
