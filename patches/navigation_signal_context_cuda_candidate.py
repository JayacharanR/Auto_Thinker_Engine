"""Staged sequential FP32 reductions on CUDA; no deployed imports."""

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _reduce(P, A, XY, OUT, N, PW: tl.constexpr, AW: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n, c = lane // 3, lane % 3
    valid = n < N
    x = tl.load(XY + n * 2, valid, other=0)
    y = tl.load(XY + n * 2 + 1, valid, other=0)
    local_sum = tl.full((BLOCK,), 0, tl.float32)
    wide_sum = tl.full((BLOCK,), 0, tl.float32)
    for k in range(1024):
        dy, dx = k // 32, k % 32
        local = tl.load(P + ((y + 32 + dy) * PW + x + 32 + dx) * 3 + c, valid, other=0)
        wide = tl.load(A + ((y + 3 * dy) * AW + x + 3 * dx) * 3 + c, valid, other=0)
        local_sum += local
        wide_sum += wide
    local_mean, wide_mean = local_sum * (1.0 / 1024), wide_sum * (1.0 / 1024)
    local_square = tl.full((BLOCK,), 0, tl.float32)
    wide_square = tl.full((BLOCK,), 0, tl.float32)
    for k in range(1024):
        dy, dx = k // 32, k % 32
        local = tl.load(P + ((y + 32 + dy) * PW + x + 32 + dx) * 3 + c, valid, other=0)
        wide = tl.load(A + ((y + 3 * dy) * AW + x + 3 * dx) * 3 + c, valid, other=0)
        ld, wd = local - local_mean, wide - wide_mean
        local_square += ld * ld
        wide_square += wd * wd
    tl.store(OUT + n * 87 + 75 + c, local_mean, valid)
    tl.store(OUT + n * 87 + 78 + c, libdevice.sqrt_rn(local_square * (1.0 / 1024)), valid)
    tl.store(OUT + n * 87 + 81 + c, wide_mean, valid)
    tl.store(OUT + n * 87 + 84 + c, libdevice.sqrt_rn(wide_square * (1.0 / 1024)), valid)
    for k in range(25):
        dy, dx = k // 5, k % 5
        value = tl.load(P + ((y + 46 + dy) * PW + x + 46 + dx) * 3 + c, valid, other=0)
        tl.store(OUT + n * 87 + c * 25 + k, value, valid)


def cuda_features(prepared, points):
    """Include CPU normalization and host-to-device transfer; output stays on GPU."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2 or not np.issubdtype(points.dtype, np.integer):
        raise ValueError("Integer Nx2 coordinates required")
    if (
        (points < 0).any()
        or (points[:, 0] >= prepared.width).any()
        or (points[:, 1] >= prepared.height).any()
    ):
        raise ValueError("Proposal outside image")
    p = torch.from_numpy(prepared.padded.astype(np.float32) / np.float32(255)).cuda()
    a = torch.from_numpy(prepared.small.astype(np.float32) / np.float32(255)).cuda()
    xy = torch.from_numpy(points.astype(np.int32)).cuda()
    output = torch.empty((len(points), 87), dtype=torch.float32, device="cuda")
    if len(points):
        _reduce[(triton.cdiv(len(points) * 3, 128),)](
            p,
            a,
            xy,
            output,
            len(points),
            p.shape[1],
            a.shape[1],
            128,
            enable_fp_fusion=False,
            num_warps=4,
        )
    return output
