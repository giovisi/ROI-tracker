# -*- coding: utf-8 -*-
"""
geometry.py
-----------
Core geometric functions adapted from RaghavPu/homography-fitting's
court_homography.py for per-frame video use.

Provides:
  - extract_mask: diff original vs masked frame → clean binary mask
  - get_hull_vertices: largest contour → convex hull → ≤6 vertices
  - classify_vertices: label each vertex as boundary or internal
  - find_corners: derive 4 parallelogram corners from hull + labels
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Mask extraction
# ---------------------------------------------------------------------------
def extract_mask(original: np.ndarray, masked: np.ndarray,
                 threshold: int = 15, morph_ksize: int = 15) -> np.ndarray:
    """Pixels that changed between original and colour-overlay frame."""
    diff = np.abs(original.astype(np.float32) - masked.astype(np.float32)).max(axis=2)
    _, mask = cv2.threshold(diff.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


# ---------------------------------------------------------------------------
# Hull extraction + vertex classification
# ---------------------------------------------------------------------------
def get_hull_vertices(mask: np.ndarray, max_vertices: int = 6) -> np.ndarray:
    """Largest contour → convex hull → simplified to ≤max_vertices vertices."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in mask.")
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    perim = cv2.arcLength(hull, True)
    pts = hull
    for factor in np.linspace(0.01, 0.5, 1000):
        approx = cv2.approxPolyDP(hull, factor * perim, True)
        if len(approx) <= max_vertices:
            pts = approx
            break
    return pts.reshape(-1, 2).astype(np.float32)


def classify_vertices(pts: np.ndarray, img_shape: tuple,
                      margin: int = 15) -> list[str]:
    """Label each vertex as 'boundary' (near frame edge) or 'internal'."""
    H_img, W_img = img_shape[:2]
    labels = []
    for p in pts:
        near_edge = (p[0] < margin or p[0] > W_img - margin or
                     p[1] < margin or p[1] > H_img - margin)
        labels.append("boundary" if near_edge else "internal")
    return labels


# ---------------------------------------------------------------------------
# Line math
# ---------------------------------------------------------------------------
def _line_from_pts(p1, p2) -> tuple[float, float, float]:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c


def _intersect(l1, l2) -> np.ndarray | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y], dtype=np.float32)


# ---------------------------------------------------------------------------
# Corner extraction
# ---------------------------------------------------------------------------
def _sort_corners(corners: np.ndarray) -> np.ndarray:
    """Sort 4 points into [TL, TR, BR, BL] order."""
    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]
    return np.array([
        corners[np.argmin(s)],  # TL
        corners[np.argmax(d)],  # TR
        corners[np.argmax(s)],  # BR
        corners[np.argmin(d)],  # BL
    ], dtype=np.float32)


def _are_adjacent(i0: int, i1: int, n: int, labels: list[str]) -> bool:
    j = (i0 + 1) % n
    while j != i1:
        if labels[j] == "boundary":
            return False
        j = (j + 1) % n
    return True


def _corners_4(pts, internal_idxs):
    return _sort_corners(pts[internal_idxs].copy())


def _corners_3(pts, labels, internal_idxs):
    n = len(pts)
    for mid_pos in range(3):
        ia = internal_idxs[mid_pos - 1]
        ib = internal_idxs[mid_pos]
        ic = internal_idxs[(mid_pos + 1) % 3]
        if (_are_adjacent(ia, ib, n, labels) and
                _are_adjacent(ib, ic, n, labels)):
            a, b, c = pts[ia], pts[ib], pts[ic]
            d = a + c - b
            return _sort_corners(np.array([a, b, c, d], dtype=np.float32))
    raise RuntimeError("Could not identify middle vertex among 3 internals")


def _corners_2_adjacent(pts, labels, i0, i1, fwd_is_direct):
    n = len(pts)
    if not fwd_is_direct:
        i0, i1 = i1, i0
    b0 = (i0 - 1) % n
    b1 = (i1 + 1) % n
    side_line_0 = _line_from_pts(pts[i0], pts[b0])
    side_line_1 = _line_from_pts(pts[i1], pts[b1])
    known_side = _line_from_pts(pts[i0], pts[i1])
    boundary_pts = [pts[i] for i in range(n) if labels[i] == "boundary"]
    a, b, c = known_side
    norm = np.hypot(a, b)
    ref_dist = (a * float(pts[i0][0]) + b * float(pts[i0][1]) + c) / norm
    farthest = max(boundary_pts,
                   key=lambda p: abs((a * float(p[0]) + b * float(p[1]) + c) / norm - ref_dist))
    c_opp = -(a * float(farthest[0]) + b * float(farthest[1]))
    opp_line = (a, b, c_opp)
    c0 = _intersect(side_line_0, opp_line)
    c1 = _intersect(side_line_1, opp_line)
    if c0 is None or c1 is None:
        raise RuntimeError("Side lines parallel to opposite side.")
    return _sort_corners(np.array([pts[i0], pts[i1], c0, c1], dtype=np.float32))


def _corners_2_opposite(pts, labels, i0, i1):
    n = len(pts)
    b0_prev = (i0 - 1) % n
    b0_next = (i0 + 1) % n
    b1_prev = (i1 - 1) % n
    b1_next = (i1 + 1) % n
    line_a = _line_from_pts(pts[i0], pts[b0_next])
    line_b = _line_from_pts(pts[b1_prev], pts[i1])
    corner_fwd = _intersect(line_a, line_b)
    line_c = _line_from_pts(pts[i1], pts[b1_next])
    line_d = _line_from_pts(pts[b0_prev], pts[i0])
    corner_bwd = _intersect(line_c, line_d)
    if corner_fwd is None or corner_bwd is None:
        raise RuntimeError("Opposite-side lines are parallel.")
    return _sort_corners(
        np.array([pts[i0], pts[i1], corner_fwd, corner_bwd], dtype=np.float32))


def _corners_2(pts, labels, internal_idxs):
    n = len(pts)
    i0, i1 = internal_idxs
    adjacent_fwd = _are_adjacent(i0, i1, n, labels)
    adjacent_bwd = _are_adjacent(i1, i0, n, labels)
    if adjacent_fwd or adjacent_bwd:
        return _corners_2_adjacent(pts, labels, i0, i1, adjacent_fwd)
    else:
        return _corners_2_opposite(pts, labels, i0, i1)


def find_corners(pts: np.ndarray, labels: list[str]) -> np.ndarray:
    """Derive 4 parallelogram corners from hull vertices + labels.

    Returns (4,2) float32 array ordered [TL, TR, BR, BL].
    """
    internal_idxs = [i for i in range(len(pts)) if labels[i] == "internal"]
    num = len(internal_idxs)
    if num >= 4:
        return _corners_4(pts, internal_idxs[:4])
    elif num == 3:
        return _corners_3(pts, labels, internal_idxs)
    elif num == 2:
        return _corners_2(pts, labels, internal_idxs)
    else:
        raise RuntimeError(
            f"Need ≥2 internal vertices to fit a parallelogram, got {num}")
