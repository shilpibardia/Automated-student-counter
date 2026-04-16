"""
=============================================================
  Automatic Student Counter using Image, Video & Webcam
  Author  : AI/ML Project
  Model   : YOLOv8 (Ultralytics) – COCO pretrained
  Purpose : Detect and count people (students) in real time
=============================================================
"""

import argparse
import sys
from pathlib import Path

# ── local modules ──────────────────────────────────────────
from utils import (
    load_model,
    process_image,
    process_video,
    process_webcam,
    print_banner,
)


# ──────────────────────────────────────────────────────────
#  CLI argument parser
# ──────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """Define all command-line flags the user can pass."""
    parser = argparse.ArgumentParser(
        description="Automatic Student Counter – YOLOv8 powered",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── input source ──────────────────────────────────────
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--image", "-i",
        type=str,
        metavar="PATH",
        help="Path to an image file  (jpg / png / bmp …)",
    )
    source_group.add_argument(
        "--video", "-v",
        type=str,
        metavar="PATH",
        help="Path to a video file  (mp4 / avi / mov …)",
    )
    source_group.add_argument(
        "--webcam", "-w",
        action="store_true",
        help="Use the default webcam (device index 0)",
    )
    source_group.add_argument(
        "--webcam-id",
        type=int,
        metavar="ID",
        help="Use a specific webcam device index (e.g. 1, 2 …)",
    )

    # ── model options ─────────────────────────────────────
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8n.pt",
        metavar="MODEL",
        help=(
            "YOLOv8 model variant:\n"
            "  yolov8n.pt  – nano  (fastest, lowest accuracy)\n"
            "  yolov8s.pt  – small\n"
            "  yolov8m.pt  – medium\n"
            "  yolov8l.pt  – large\n"
            "  yolov8x.pt  – extra-large (slowest, best accuracy)\n"
            "Default: yolov8n.pt"
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        metavar="FLOAT",
        help="Confidence threshold 0–1  (default: 0.40)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        metavar="FLOAT",
        help="IoU threshold for NMS  (default: 0.45)",
    )

    # ── display / output options ───────────────────────────
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Save annotated output to file (image or video)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the OpenCV preview window",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=640,
        metavar="WIDTH",
        help="Resize frame width for faster inference  (default: 640)",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="Enable Region-of-Interest (ROI) selector before counting",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Save per-frame count log to a CSV file (video / webcam only)",
    )

    return parser


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
def main() -> None:
    print_banner()

    parser = build_parser()
    args   = parser.parse_args()

    # ── load model once ────────────────────────────────────
    model = load_model(args.model)

    # ── route to the correct pipeline ─────────────────────
    if args.image:
        path = Path(args.image)
        if not path.exists():
            print(f"[ERROR] Image not found: {path}")
            sys.exit(1)
        process_image(
            model      = model,
            source     = str(path),
            conf       = args.conf,
            iou        = args.iou,
            output     = args.output,
            display    = not args.no_display,
            resize_w   = args.resize,
            use_roi    = args.roi,
        )

    elif args.video:
        path = Path(args.video)
        if not path.exists():
            print(f"[ERROR] Video not found: {path}")
            sys.exit(1)
        process_video(
            model      = model,
            source     = str(path),
            conf       = args.conf,
            iou        = args.iou,
            output     = args.output,
            display    = not args.no_display,
            resize_w   = args.resize,
            use_roi    = args.roi,
            csv_path   = args.csv,
        )

    elif args.webcam or args.webcam_id is not None:
        cam_id = args.webcam_id if args.webcam_id is not None else 0
        process_webcam(
            model      = model,
            cam_id     = cam_id,
            conf       = args.conf,
            iou        = args.iou,
            output     = args.output,
            display    = not args.no_display,
            resize_w   = args.resize,
            use_roi    = args.roi,
            csv_path   = args.csv,
        )


if __name__ == "__main__":
    main()
