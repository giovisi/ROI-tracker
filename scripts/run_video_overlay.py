# -*- coding: utf-8 -*-
"""
run_video_overlay.py
--------------------
Main video pipeline that processes a tennis broadcast frame-by-frame:

  1. Reads paired original + SAM2-masked videos in sync
  2. Per-frame: diff → mask → hull → corners → oriented homography
  3. Temporal smoothing of corners
  4. Composites an overlay logo into the detected region
  5. Writes output video

This adapts RaghavPu/homography-fitting (single-image) to work on video,
using the pipeline architecture from enriquedlh97/tennis-virtual-ads.

Usage
-----
  python scripts/run_video_overlay.py \\
      original.mp4 masked.mp4 output.mp4 \\
      --logo sponsor_logo.png

  # With options:
  python scripts/run_video_overlay.py \\
      original.mp4 masked.mp4 output.mp4 \\
      --logo logo.png \\
      --max_frames 300 \\
      --start_frame 0 \\
      --stride 1 \\
      --resize 1280x720 \\
      --smoothing 0.3 \\
      --debug
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src/ to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from geometry import (
    classify_vertices,
    extract_mask,
    find_corners,
    get_hull_vertices,
)
from homography import (
    composite_overlay,
    compute_oriented_homography,
    estimate_camera_matrix,
)
from temporal import CornerStabilizer
from video_io import VideoReader, VideoWriter


# ---------------------------------------------------------------------------
# Debug drawing
# ---------------------------------------------------------------------------
def draw_debug_overlay(frame: np.ndarray,
                       corners: np.ndarray | None,
                       frame_idx: int,
                       status: str,
                       fps_val: float) -> np.ndarray:
    """Draw debug info on the frame: corners, status, FPS."""
    vis = frame.copy()

    # Frame counter + FPS
    text = f"Frame {frame_idx} | {fps_val:.1f} fps | {status}"
    cv2.putText(vis, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if corners is not None:
        # Draw the quad
        quad = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [quad], True, (0, 255, 255), 2, cv2.LINE_AA)

        # Label corners
        for label, pt in zip(["TL", "TR", "BR", "BL"], corners):
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(vis, (cx, cy), 6, (0, 200, 0), -1)
            cv2.putText(vis, label, (cx + 8, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                        1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------
def process_frame(
    original: np.ndarray,
    masked: np.ndarray,
    K: np.ndarray,
    corner_stabilizer: CornerStabilizer,
    overlay_img: np.ndarray | None = None,
    padding: float = 0.05,
    mask_threshold: int = 15,
    morph_ksize: int = 15,
    min_mask_area: int = 5000,
) -> dict:
    """Process a single frame pair through the full pipeline.

    Returns a dict with:
      corners : (4,2) stabilized corners or None
      homo    : homography dict or None
      result  : composited frame (or original if no region found)
      status  : string describing what happened
    """
    try:
        # Step 1: Extract mask from diff
        mask = extract_mask(original, masked,
                            threshold=mask_threshold,
                            morph_ksize=morph_ksize)

        # Check if mask has enough area
        mask_area = np.count_nonzero(mask)
        if mask_area < min_mask_area:
            return dict(corners=None, homo=None, result=original.copy(),
                        status=f"mask too small ({mask_area}px)")

        # Step 2: Hull vertices + classification
        pts = get_hull_vertices(mask)
        labels = classify_vertices(pts, mask.shape)

        # Step 3: Find 4 corners
        corners = find_corners(pts, labels)

        # Step 4: Temporal smoothing
        corners = corner_stabilizer.update(corners)

        # Step 5: Oriented homography
        homo = compute_oriented_homography(corners, K)

        # Step 6: Composite overlay
        if overlay_img is not None:
            result = composite_overlay(original, corners, overlay_img, homo,
                                       padding=padding)
        else:
            result = original.copy()

        return dict(corners=corners, homo=homo, result=result, status="ok")

    except RuntimeError as e:
        return dict(corners=None, homo=None, result=original.copy(),
                    status=f"error: {e}")
    except Exception as e:
        return dict(corners=None, homo=None, result=original.copy(),
                    status=f"unexpected: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace):
    """Run the full video overlay pipeline."""

    # Parse resize if given
    resize = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    # Open input videos
    reader_kwargs = dict(
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        stride=args.stride,
        resize=resize,
    )
    print(f"[Pipeline] Opening original: {args.original}")
    reader_orig = VideoReader(args.original, **reader_kwargs)
    print(f"[Pipeline] Opening masked:   {args.masked}")
    reader_mask = VideoReader(args.masked, **reader_kwargs)

    print(f"[Pipeline] Original: {reader_orig.width}x{reader_orig.height} "
          f"@ {reader_orig.fps:.1f} fps, {reader_orig.total_frames} total frames")

    # Load overlay
    overlay = None
    if args.logo:
        overlay = cv2.imread(args.logo, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise RuntimeError(f"Could not read logo: {args.logo}")
        print(f"[Pipeline] Loaded logo: {args.logo} "
              f"({overlay.shape[1]}x{overlay.shape[0]}, "
              f"{'RGBA' if overlay.shape[2] == 4 else 'RGB'})")

    # Determine output frame size
    if resize:
        out_w, out_h = resize
    else:
        out_w, out_h = reader_orig.width, reader_orig.height

    # Camera matrix
    K = estimate_camera_matrix((out_h, out_w),
                               focal_length=args.focal_length)
    print(f"[Pipeline] Camera K: f={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # Temporal stabilizer
    stabilizer = CornerStabilizer(
        alpha=args.smoothing,
        jump_threshold=args.jump_threshold,
    )

    # Open output writer(s)
    writer = VideoWriter(args.output, reader_orig.fps, out_w, out_h)
    debug_writer = None
    if args.debug:
        debug_path = str(Path(args.output).with_suffix("")) + "_debug.mp4"
        debug_writer = VideoWriter(debug_path, reader_orig.fps, out_w, out_h)
        print(f"[Pipeline] Debug video: {debug_path}")

    # Process frames
    print(f"[Pipeline] Processing frames (smoothing={args.smoothing}, "
          f"jump_threshold={args.jump_threshold})...")

    iter_orig = iter(reader_orig)
    iter_mask = iter(reader_mask)

    frames_processed = 0
    frames_ok = 0
    t_start = time.time()

    while True:
        try:
            idx_o, frame_orig = next(iter_orig)
            idx_m, frame_mask = next(iter_mask)
        except StopIteration:
            break

        t_frame = time.time()

        # Process frame
        result = process_frame(
            original=frame_orig,
            masked=frame_mask,
            K=K,
            corner_stabilizer=stabilizer,
            overlay_img=overlay,
            padding=args.padding,
            mask_threshold=args.mask_threshold,
            morph_ksize=args.morph_ksize,
            min_mask_area=args.min_mask_area,
        )

        # Write main output
        writer.write(result["result"])

        # Write debug output
        if debug_writer is not None:
            elapsed = time.time() - t_frame
            fps_val = 1.0 / elapsed if elapsed > 0 else 0
            debug_frame = draw_debug_overlay(
                result["result"], result["corners"],
                idx_o, result["status"], fps_val)
            debug_writer.write(debug_frame)

        frames_processed += 1
        if result["status"] == "ok":
            frames_ok += 1

        # Progress reporting
        if frames_processed % 50 == 0:
            elapsed_total = time.time() - t_start
            avg_fps = frames_processed / elapsed_total if elapsed_total > 0 else 0
            print(f"  Frame {idx_o}: {result['status']} "
                  f"({frames_ok}/{frames_processed} ok, {avg_fps:.1f} avg fps)")

    # Cleanup
    total_time = time.time() - t_start
    avg_fps = frames_processed / total_time if total_time > 0 else 0

    writer.release()
    if debug_writer:
        debug_writer.release()
    reader_orig.release()
    reader_mask.release()

    print(f"\n[Pipeline] Done!")
    print(f"  Frames processed: {frames_processed}")
    print(f"  Frames with successful overlay: {frames_ok} "
          f"({100*frames_ok/max(1,frames_processed):.1f}%)")
    print(f"  Total time: {total_time:.1f}s ({avg_fps:.1f} fps)")
    print(f"  Output: {args.output}")
    if args.debug:
        print(f"  Debug:  {debug_path}")


# ---------------------------------------------------------------------------
# Single-frame mode (for testing without a masked video)
# ---------------------------------------------------------------------------
def run_single_frame(args: argparse.Namespace):
    """Process a single frame pair (images, not videos)."""
    print(f"[SingleFrame] Loading original: {args.original}")
    original = cv2.imread(args.original)
    if original is None:
        raise RuntimeError(f"Cannot read: {args.original}")

    print(f"[SingleFrame] Loading masked: {args.masked}")
    masked = cv2.imread(args.masked)
    if masked is None:
        raise RuntimeError(f"Cannot read: {args.masked}")

    overlay = None
    if args.logo:
        overlay = cv2.imread(args.logo, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise RuntimeError(f"Cannot read logo: {args.logo}")
        print(f"[SingleFrame] Loaded logo: {args.logo}")

    K = estimate_camera_matrix(original.shape, focal_length=args.focal_length)
    stabilizer = CornerStabilizer(alpha=1.0)  # no smoothing for single frame

    result = process_frame(
        original=original,
        masked=masked,
        K=K,
        corner_stabilizer=stabilizer,
        overlay_img=overlay,
        padding=args.padding,
    )

    cv2.imwrite(args.output, result["result"])
    print(f"[SingleFrame] Status: {result['status']}")
    print(f"[SingleFrame] Saved: {args.output}")

    if result["corners"] is not None and args.debug:
        debug = draw_debug_overlay(result["result"], result["corners"], 0,
                                   result["status"], 0)
        debug_path = str(Path(args.output).with_suffix("")) + "_debug.png"
        cv2.imwrite(debug_path, debug)
        print(f"[SingleFrame] Debug: {debug_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Video overlay pipeline: homography-fitting → video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full video pipeline
  python scripts/run_video_overlay.py original.mp4 masked.mp4 output.mp4 \\
      --logo sponsor_logo.png

  # First 100 frames with debug overlay
  python scripts/run_video_overlay.py original.mp4 masked.mp4 output.mp4 \\
      --logo logo.png --max_frames 100 --debug

  # Single image pair (no video)
  python scripts/run_video_overlay.py frame.jpg masked_frame.jpg out.png \\
      --logo logo.png --single

  # Tuning: stronger smoothing, lower resize
  python scripts/run_video_overlay.py original.mp4 masked.mp4 output.mp4 \\
      --logo logo.png --smoothing 0.15 --resize 960x540 --stride 2
        """,
    )

    parser.add_argument("original",
                        help="Original video (or image with --single)")
    parser.add_argument("masked",
                        help="SAM2-masked video (or image with --single)")
    parser.add_argument("output",
                        help="Output video/image path")

    # Overlay
    parser.add_argument("--logo", default=None,
                        help="Image to warp into the detected region (PNG/JPG, "
                             "supports RGBA)")
    parser.add_argument("--padding", type=float, default=0.05,
                        help="Overlay padding fraction (default: 0.05)")

    # Video iteration
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start processing from this frame (default: 0)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames to process (default: all)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (default: 1)")
    parser.add_argument("--resize", type=str, default=None,
                        help="Resize frames to WxH, e.g. 1280x720")

    # Camera
    parser.add_argument("--focal_length", type=float, default=None,
                        help="Camera focal length in pixels (default: max(w,h))")

    # Temporal stabilization
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="Corner EMA alpha: 0=freeze, 1=no smoothing "
                             "(default: 0.3)")
    parser.add_argument("--jump_threshold", type=float, default=80.0,
                        help="Corner jump threshold in pixels for cut "
                             "detection (default: 80)")

    # Mask extraction tuning
    parser.add_argument("--mask_threshold", type=int, default=15,
                        help="Diff threshold for mask extraction (default: 15)")
    parser.add_argument("--morph_ksize", type=int, default=15,
                        help="Morphology kernel size (default: 15)")
    parser.add_argument("--min_mask_area", type=int, default=5000,
                        help="Minimum mask area in pixels (default: 5000)")

    # Mode
    parser.add_argument("--single", action="store_true",
                        help="Single-frame mode (process images, not video)")
    parser.add_argument("--debug", action="store_true",
                        help="Write a debug video/image with corner overlays")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.single:
        run_single_frame(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
