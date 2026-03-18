# -*- coding: utf-8 -*-
"""
video_io.py
-----------
VideoReader and VideoWriter classes, modeled after the tennis-virtual-ads
pipeline's io/video.py interface.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Iterate over video frames with optional start/stride/max/resize."""

    def __init__(
        self,
        path: str,
        start_frame: int = 0,
        max_frames: int | None = None,
        stride: int = 1,
        resize: tuple[int, int] | None = None,
    ):
        self.path = str(Path(path).expanduser().resolve())
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.start_frame = start_frame
        self.max_frames = max_frames
        self.stride = stride
        self.resize = resize  # (width, height)

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        frame_idx = self.start_frame
        frames_yielded = 0

        while True:
            if self.max_frames is not None and frames_yielded >= self.max_frames:
                break
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)
            yield frame_idx, frame
            frames_yielded += 1
            # Skip stride-1 frames
            for _ in range(self.stride - 1):
                frame_idx += 1
                self.cap.read()  # discard
            frame_idx += 1

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()


class VideoWriter:
    """Write frames to an MP4 file."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        self.path = str(Path(path).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.path}")
        self.frames_written = 0

    def write(self, frame: np.ndarray):
        self.writer.write(frame)
        self.frames_written += 1

    def release(self):
        self.writer.release()

    def __del__(self):
        self.release()


def open_paired_readers(
    original_path: str,
    masked_path: str,
    **kwargs,
) -> tuple[VideoReader, VideoReader]:
    """Open two VideoReaders with identical settings for synchronized iteration."""
    reader_orig = VideoReader(original_path, **kwargs)
    reader_mask = VideoReader(masked_path, **kwargs)
    return reader_orig, reader_mask
