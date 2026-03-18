# -*- coding: utf-8 -*-
"""
run_sam2_overlay.py
-------------------
End-to-end pipeline that DOES NOT require a pre-masked video.
Instead, it uses SAM2's video predictor to:

  1. Let you click seed points on frame 0 to define the ad region
  2. SAM2 propagates masks across all frames
  3. Per-frame: mask → hull → corners → oriented homography → overlay

This is the "batteries-included" version that combines video_masker.py
(from homography-fitting) with the video overlay pipeline.

Usage
-----
  python scripts/run_sam2_overlay.py \\
      input.mp4 output.mp4 \\
      --logo sponsor_logo.png \\
      --point 1:500,300

  # Or interactively click:
  python scripts/run_sam2_overlay.py \\
      input.mp4 output.mp4 \\
      --logo sponsor_logo.png \\
      --interactive
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

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
from video_io import VideoWriter


# ---------------------------------------------------------------------------
# SAM2 integration (adapted from video_masker.py)
# ---------------------------------------------------------------------------
def _detect_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_frames(video_path: str, out_dir: str) -> list[str]:
    """Extract JPEG frames via ffmpeg."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-q:v", "2", "-start_number", "0",
        os.path.join(out_dir, "%05d.jpg"),
        "-y", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    names = sorted(
        [p for p in os.listdir(out_dir) if p.endswith((".jpg", ".jpeg"))],
        key=lambda p: int(Path(p).stem),
    )
    if not names:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return names


def _get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0


def run_sam2_propagation(
    video_path: str,
    prompts: list[dict],
    checkpoint: str,
    model_cfg: str,
) -> tuple[str, dict[int, dict[int, np.ndarray]]]:
    """Run SAM2 video predictor and return frame dir + per-frame masks.

    Parameters
    ----------
    prompts : list of dicts with keys:
        obj_id, points (N,2), labels (N,), frame_idx (default 0)
    """
    import torch
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = _detect_device()
    print(f"[SAM2] Device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Import SAM2
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    # Extract frames
    tmp_dir = tempfile.mkdtemp(prefix="sam2_pipeline_")
    print(f"[SAM2] Extracting frames to {tmp_dir} ...")
    frame_names = _extract_frames(video_path, tmp_dir)
    print(f"[SAM2] {len(frame_names)} frames extracted")

    # Init inference state
    inference_state = predictor.init_state(video_path=tmp_dir)

    # Add prompts
    for prompt in prompts:
        kwargs = dict(
            inference_state=inference_state,
            frame_idx=prompt.get("frame_idx", 0),
            obj_id=prompt["obj_id"],
        )
        if "points" in prompt and prompt["points"] is not None:
            kwargs["points"] = prompt["points"]
            kwargs["labels"] = prompt["labels"]
        if "box" in prompt and prompt["box"] is not None:
            kwargs["box"] = prompt["box"]
        predictor.add_new_points_or_box(**kwargs)
        print(f"[SAM2] Added prompt obj_id={prompt['obj_id']} "
              f"on frame {prompt.get('frame_idx', 0)}")

    # Propagate
    print("[SAM2] Propagating masks ...")
    video_segments: dict[int, dict[int, np.ndarray]] = {}
    for out_idx, out_obj_ids, out_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_idx] = {
            oid: (out_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, oid in enumerate(out_obj_ids)
        }

    print(f"[SAM2] Masks for {len(video_segments)} frames")
    return tmp_dir, video_segments


# ---------------------------------------------------------------------------
# Mask-based corner extraction (per frame)
# ---------------------------------------------------------------------------
def corners_from_mask(
    mask: np.ndarray,
    img_shape: tuple,
    min_area: int = 5000,
) -> np.ndarray | None:
    """Extract 4 corners from a binary mask using hull + vertex classification.

    Returns (4,2) float32 or None if extraction fails.
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    area = np.count_nonzero(mask_uint8)
    if area < min_area:
        return None

    try:
        pts = get_hull_vertices(mask_uint8)
        labels = classify_vertices(pts, img_shape)
        corners = find_corners(pts, labels)
        return corners
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args):
    import time

    # Parse prompts from CLI
    prompts = []
    for token in args.point:
        oid_str, xy_str = token.split(":")
        x, y = map(float, xy_str.split(","))
        oid = int(oid_str)

        # Find existing prompt for this obj_id or create new
        existing = next((p for p in prompts if p["obj_id"] == oid), None)
        if existing:
            existing["points"] = np.vstack([
                existing["points"], [[x, y]]
            ]).astype(np.float32)
            existing["labels"] = np.append(existing["labels"], 1).astype(np.int32)
        else:
            prompts.append(dict(
                obj_id=oid,
                points=np.array([[x, y]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
                frame_idx=0,
            ))

    if not prompts:
        print("Error: provide at least one --point prompt (e.g. --point 1:500,300)")
        return

    # Load overlay
    overlay = None
    if args.logo:
        overlay = cv2.imread(args.logo, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise RuntimeError(f"Cannot read logo: {args.logo}")
        print(f"[Pipeline] Logo: {args.logo} ({overlay.shape[1]}x{overlay.shape[0]})")

    # Run SAM2
    t0 = time.time()
    frame_dir, video_segments = run_sam2_propagation(
        args.input, prompts, args.checkpoint, args.model_cfg
    )
    print(f"[Pipeline] SAM2 took {time.time()-t0:.1f}s")

    # Read frames and process
    frame_names = sorted(
        [p for p in os.listdir(frame_dir) if p.endswith((".jpg", ".jpeg"))],
        key=lambda p: int(Path(p).stem),
    )

    first_frame = cv2.imread(os.path.join(frame_dir, frame_names[0]))
    h, w = first_frame.shape[:2]
    fps = _get_fps(args.input)

    K = estimate_camera_matrix((h, w), focal_length=args.focal_length)
    stabilizer = CornerStabilizer(alpha=args.smoothing,
                                   jump_threshold=args.jump_threshold)

    writer = VideoWriter(args.output, fps, w, h)
    print(f"[Pipeline] Processing {len(frame_names)} frames ...")

    t_start = time.time()
    frames_ok = 0

    for frame_idx, fname in enumerate(frame_names):
        frame = cv2.imread(os.path.join(frame_dir, fname))
        if frame is None:
            continue

        # Get SAM2 mask for this frame (use first object)
        masks_by_obj = video_segments.get(frame_idx, {})
        if not masks_by_obj:
            writer.write(frame)
            continue

        # Combine all object masks (or use the first one)
        obj_id = list(masks_by_obj.keys())[0]
        mask = masks_by_obj[obj_id]

        # Extract corners from mask
        corners = corners_from_mask(mask, frame.shape,
                                     min_area=args.min_mask_area)
        if corners is None:
            writer.write(frame)
            continue

        # Stabilize
        corners = stabilizer.update(corners)

        # Compute homography + composite
        try:
            homo = compute_oriented_homography(corners, K)
            if overlay is not None:
                result = composite_overlay(frame, corners, overlay, homo,
                                            padding=args.padding)
            else:
                result = frame
            frames_ok += 1
        except Exception:
            result = frame

        writer.write(result)

        if frame_idx % 50 == 0:
            elapsed = time.time() - t_start
            avg_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_idx}/{len(frame_names)} "
                  f"({frames_ok} ok, {avg_fps:.1f} fps)")

    writer.release()

    # Cleanup temp frames
    shutil.rmtree(frame_dir, ignore_errors=True)

    total = time.time() - t_start
    print(f"\n[Pipeline] Done!")
    print(f"  Frames: {len(frame_names)}, overlaid: {frames_ok}")
    print(f"  Time: {total:.1f}s ({len(frame_names)/total:.1f} fps)")
    print(f"  Output: {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SAM2-integrated video overlay pipeline (no pre-masked "
                    "video needed)",
    )

    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")

    parser.add_argument("--logo", default=None,
                        help="Overlay image (PNG/JPG, supports RGBA)")
    parser.add_argument("--point", metavar="ID:X,Y", action="append",
                        default=[],
                        help="SAM2 seed point. Format: obj_id:x,y (repeatable)")
    parser.add_argument("--padding", type=float, default=0.05)

    parser.add_argument("--focal_length", type=float, default=None)
    parser.add_argument("--smoothing", type=float, default=0.3)
    parser.add_argument("--jump_threshold", type=float, default=80.0)
    parser.add_argument("--min_mask_area", type=int, default=5000)

    parser.add_argument("--checkpoint",
                        default="sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model-cfg",
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
