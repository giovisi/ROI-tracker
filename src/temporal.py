# -*- coding: utf-8 -*-
"""
temporal.py
-----------
Temporal stabilization of homographies and corners across video frames.

Provides:
  - HomographyStabilizer: EMA smoothing with jump/cut detection
  - CornerStabilizer: smooth the 4 corner points directly
"""
from __future__ import annotations

import numpy as np


class CornerStabilizer:
    """Exponential moving average on corner positions with cut detection.

    When corners jump by more than `jump_threshold` pixels (average L2
    between old and new corners), the stabilizer resets to avoid blending
    across scene cuts.
    """

    def __init__(self, alpha: float = 0.3, jump_threshold: float = 80.0):
        """
        Parameters
        ----------
        alpha : float
            EMA weight for new frame (0 = freeze, 1 = no smoothing).
        jump_threshold : float
            If average corner displacement exceeds this (pixels), reset.
        """
        self.alpha = alpha
        self.jump_threshold = jump_threshold
        self._prev: np.ndarray | None = None
        self._frame_count = 0

    def update(self, corners: np.ndarray) -> np.ndarray:
        """Smooth corners and return stabilized (4,2) float32 array.

        Parameters
        ----------
        corners : (4, 2) float32
            New corners from the current frame.

        Returns
        -------
        (4, 2) float32 stabilized corners.
        """
        corners = corners.astype(np.float32)

        if self._prev is None:
            self._prev = corners.copy()
            self._frame_count = 1
            return corners.copy()

        # Compute average corner displacement
        displacements = np.linalg.norm(corners - self._prev, axis=1)
        avg_disp = displacements.mean()

        if avg_disp > self.jump_threshold:
            # Scene cut or big camera move — reset
            self._prev = corners.copy()
            self._frame_count = 1
            return corners.copy()

        # EMA blend
        smoothed = self.alpha * corners + (1 - self.alpha) * self._prev
        self._prev = smoothed.copy()
        self._frame_count += 1
        return smoothed

    def reset(self):
        """Manually reset the stabilizer state."""
        self._prev = None
        self._frame_count = 0


class HomographyStabilizer:
    """Smooth homography matrices directly via element-wise EMA.

    This is a simpler alternative when you want to stabilize H rather than
    corners.  Works best when H doesn't change drastically frame-to-frame.
    """

    def __init__(self, alpha: float = 0.3, jump_threshold: float = 0.1):
        self.alpha = alpha
        self.jump_threshold = jump_threshold
        self._prev: np.ndarray | None = None

    def update(self, H: np.ndarray) -> np.ndarray:
        H = H.astype(np.float64)
        # Normalize so H[2,2] = 1
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

        if self._prev is None:
            self._prev = H.copy()
            return H.copy()

        diff = np.abs(H - self._prev).max()
        if diff > self.jump_threshold:
            self._prev = H.copy()
            return H.copy()

        smoothed = self.alpha * H + (1 - self.alpha) * self._prev
        self._prev = smoothed.copy()
        return smoothed

    def reset(self):
        self._prev = None
