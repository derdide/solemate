"""1-D perspective correction along the ski axis using tape reference marks."""
from __future__ import annotations

import cv2
import numpy as np


def rectify_image(
    img: np.ndarray,
    tape_points_px: list[tuple[float, float]],
    mounting_point_px: tuple[float, float],
    axis_dx: float,
    axis_dy: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Apply 1-D perspective correction along the ski axis.

    Tape marks are assumed to be at EQUAL real-world spacing.
    Only the along-axis component is corrected; lateral position is unchanged.

    Requires at least 3 tape points. Returns (rectified_img, new_mounting_point_px).
    """
    n = len(tape_points_px)
    if n < 3:
        raise ValueError("Need at least 3 tape points for perspective correction")

    mpx, mpy = mounting_point_px

    # Project tape points onto the ski axis (coord relative to mount point), then sort.
    u_obs = sorted(
        (x - mpx) * axis_dx + (y - mpy) * axis_dy
        for x, y in tape_points_px
    )

    # Target positions: equal spacing anchored to the first mark.
    mean_span = float(np.mean([u_obs[i + 1] - u_obs[i] for i in range(n - 1)]))
    u_tgt = [u_obs[0] + i * mean_span for i in range(n)]

    # Fit 1-D projective:  u_tgt[i] = (p·u_obs[i] + q) / (r·u_obs[i] + 1)
    # Linear rearrangement: p·u - u_tgt·r·u + q = u_tgt  →  A·[p,r,q]ᵀ = b
    A = np.array(
        [[u_obs[i], -u_tgt[i] * u_obs[i], 1.0] for i in range(n)],
        dtype=np.float64,
    )
    b_vec = np.array(u_tgt, dtype=np.float64)
    params, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    p, r, q = float(params[0]), float(params[1]), float(params[2])

    # Build remap: for each output pixel find the corresponding source pixel.
    # Output (corrected) coords → source (original) coords via inverse map:
    #   u_orig = (q - u_corr) / (r·u_corr - p)
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    dx = xx - mpx
    dy = yy - mpy
    u_c = dx * axis_dx + dy * axis_dy          # along-axis in corrected space
    v   = -dx * axis_dy + dy * axis_dx         # perpendicular (unchanged)

    denom = r * u_c - p
    # Guard against near-zero denominator far from the image area.
    denom = np.where(np.abs(denom) < 1e-9, np.sign(denom + 1e-18) * 1e-9, denom)
    u_o = (q - u_c) / denom

    src_x = (mpx + u_o * axis_dx - v * axis_dy).astype(np.float32)
    src_y = (mpy + u_o * axis_dy + v * axis_dx).astype(np.float32)

    rectified = cv2.remap(
        img, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Mount point: u_mp_orig = 0  →  u_mp_corrected = f(0) = q
    new_mpx = float(mpx + q * axis_dx)
    new_mpy = float(mpy + q * axis_dy)
    return rectified, (new_mpx, new_mpy)
