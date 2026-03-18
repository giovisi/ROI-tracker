# -*- coding: utf-8 -*-
"""
homography.py
-------------
Camera-intrinsics-aware homography computation, adapted from
RaghavPu/homography-fitting's region_overlay.py.

Provides:
  - estimate_camera_matrix: build K from frame shape
  - compute_oriented_homography: K-aware homography preserving true aspect ratio
  - composite_overlay: warp an overlay image into a quad region
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------
def estimate_camera_matrix(frame_shape: tuple,
                           focal_length: float | None = None) -> np.ndarray:
    """Build K = [[f,0,cx],[0,f,cy],[0,0,1]].

    If focal_length is None, default to max(w, h).
    """
    h, w = frame_shape[:2]
    f = focal_length if focal_length is not None else float(max(h, w))
    cx, cy = w / 2.0, h / 2.0
    return np.array([[f, 0.0, cx],
                     [0.0, f, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Oriented homography
# ---------------------------------------------------------------------------
def compute_oriented_homography(
    corners: np.ndarray,
    K: np.ndarray,
) -> dict:
    """Compute the oriented homography for a planar region using camera intrinsics.

    Parameters
    ----------
    corners : (4, 2) float32, ordered TL, TR, BR, BL.
    K : (3, 3) float64 camera intrinsic matrix.

    Returns
    -------
    dict with keys: H, dst_w, dst_h, dst_rect, aspect, phys_w, phys_h,
                    R, t, normal, pts_3d
    """
    corners_f64 = corners.astype(np.float64)

    unit_src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    H_unit, _ = cv2.findHomography(unit_src, corners_f64)
    if H_unit is None:
        raise ValueError("findHomography failed — degenerate corners?")

    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H_unit, K)

    best = 0
    for i in range(num):
        n = Ns[i].flatten()
        t = Ts[i].flatten()
        if t[2] > 0 and n[2] < 0:
            best = i
            break

    R = Rs[best]
    t = Ts[best].flatten()
    normal = Ns[best].flatten()

    K_inv = np.linalg.inv(K)
    corners_h = np.hstack([corners_f64, np.ones((4, 1))])
    rays = (K_inv @ corners_h.T).T
    dot = rays @ normal
    eps = 1e-6
    dot = np.where(np.abs(dot) < eps, eps, dot)
    depths = 1.0 / dot

    if depths.mean() < 0:
        normal = -normal
        depths = -depths

    pts_3d = rays * depths[:, np.newaxis]

    w_top = np.linalg.norm(pts_3d[1] - pts_3d[0])
    w_bot = np.linalg.norm(pts_3d[2] - pts_3d[3])
    h_left = np.linalg.norm(pts_3d[3] - pts_3d[0])
    h_rgt = np.linalg.norm(pts_3d[2] - pts_3d[1])
    phys_w = (w_top + w_bot) / 2.0
    phys_h = (h_left + h_rgt) / 2.0
    aspect = phys_w / phys_h if phys_h > 1e-6 else 1.0

    DST_H = 256
    DST_W = max(1, int(round(DST_H * aspect)))
    dst_rect = np.array([[0, 0], [DST_W, 0], [DST_W, DST_H], [0, DST_H]],
                        dtype=np.float32)

    H_final, _ = cv2.findHomography(dst_rect, corners.astype(np.float32))

    return dict(
        H=H_final,
        dst_w=DST_W,
        dst_h=DST_H,
        dst_rect=dst_rect,
        aspect=aspect,
        phys_w=phys_w,
        phys_h=phys_h,
        R=R,
        t=t,
        normal=normal,
        pts_3d=pts_3d,
    )


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
def composite_overlay(frame: np.ndarray,
                      corners: np.ndarray,
                      overlay_img: np.ndarray,
                      homo: dict,
                      padding: float = 0.05) -> np.ndarray:
    """Warp overlay_img into the banner region using the oriented homography.

    overlay_img can be RGBA (alpha compositing) or RGB.
    """
    dst_w, dst_h = homo["dst_w"], homo["dst_h"]
    H_final = homo["H"]

    avail_w = int(dst_w * (1 - 2 * padding))
    avail_h = int(dst_h * (1 - 2 * padding))
    ov_h, ov_w = overlay_img.shape[:2]
    scale = min(avail_w / ov_w, avail_h / ov_h)
    new_w = max(1, int(round(ov_w * scale)))
    new_h = max(1, int(round(ov_h * scale)))
    ov_resized = cv2.resize(overlay_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Sample background color from frame's banner region
    H_to_rect, _ = cv2.findHomography(corners, homo["dst_rect"])
    warped_orig = cv2.warpPerspective(frame, H_to_rect, (dst_w, dst_h))
    bg_color = tuple(int(c) for c in cv2.mean(warped_orig)[:3])

    canvas = np.full((dst_h, dst_w, 3), bg_color, dtype=np.uint8)
    ox = (dst_w - new_w) // 2
    oy = (dst_h - new_h) // 2

    if ov_resized.ndim == 3 and ov_resized.shape[2] == 4:
        rgb = ov_resized[:, :, :3].astype(np.float32)
        alpha = ov_resized[:, :, 3:].astype(np.float32) / 255.0
        patch = canvas[oy:oy+new_h, ox:ox+new_w].astype(np.float32)
        canvas[oy:oy+new_h, ox:ox+new_w] = (
            rgb * alpha + patch * (1 - alpha)
        ).astype(np.uint8)
    else:
        canvas[oy:oy+new_h, ox:ox+new_w] = ov_resized[:, :, :3]

    warped_canvas = cv2.warpPerspective(canvas, H_final,
                                        (frame.shape[1], frame.shape[0]))

    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)

    result = frame.copy()
    result[mask > 0] = warped_canvas[mask > 0]
    return result
