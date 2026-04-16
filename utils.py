"""
utils.py – Helper functions for the Automatic Student Counter
=============================================================
Contains:
  • load_model        – download / load YOLOv8 weights
  • detect_persons    – run inference, return person boxes only
  • draw_annotations  – overlay boxes, labels, count badge
  • select_roi        – interactive region-of-interest selector
  • process_image     – single-image pipeline
  • process_video     – video-file pipeline
  • process_webcam    – live-webcam pipeline
  • CSVLogger         – lightweight frame-count logger
  • print_banner      – welcome ASCII art
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── YOLOv8 ────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run:  pip install ultralytics")
    sys.exit(1)

# ── COCO class index for "person" ──────────────────────────
PERSON_CLASS_ID = 0

# ── Visual style constants ─────────────────────────────────
BOX_COLOR        = (0, 220, 120)   # bright green  (BGR)
BOX_THICKNESS    = 2
LABEL_BG_COLOR   = (0, 220, 120)
LABEL_TEXT_COLOR = (0, 0, 0)       # black text on green label
BADGE_BG_COLOR   = (15, 15, 15)    # dark badge
BADGE_TEXT_COLOR = (0, 220, 120)
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_BOX   = 0.55
FONT_SCALE_BADGE = 0.9
FONT_THICKNESS   = 2


# ══════════════════════════════════════════════════════════
#  Banner
# ══════════════════════════════════════════════════════════
def print_banner() -> None:
    """Print a welcome banner to the terminal."""
    banner = r"""
╔══════════════════════════════════════════════════════════╗
║     AUTOMATIC STUDENT COUNTER  –  YOLOv8 + OpenCV       ║
║     Detects & counts people in images, video, webcam     ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


# ══════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════
def load_model(model_name: str = "yolov8n.pt") -> "YOLO":
    """
    Load (or auto-download) a YOLOv8 model.

    Parameters
    ----------
    model_name : str
        Filename of the YOLOv8 weights, e.g. 'yolov8n.pt'.
        Ultralytics will download it automatically on first run.

    Returns
    -------
    YOLO
        Loaded model object ready for inference.
    """
    print(f"[INFO] Loading model: {model_name} …")
    model = YOLO(model_name)
    print(f"[INFO] Model loaded  ✔  (classes: {len(model.names)})")
    return model


# ══════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════
def detect_persons(
    model,
    frame: np.ndarray,
    conf: float = 0.4,
    iou:  float = 0.45,
    roi:  Optional[Tuple[int, int, int, int]] = None,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Run YOLOv8 inference and return bounding boxes for *persons only*.

    Parameters
    ----------
    model   : YOLO model instance
    frame   : BGR image (numpy array)
    conf    : minimum confidence to accept a detection
    iou     : IoU threshold for non-maximum suppression
    roi     : optional (x, y, w, h) tuple; only detections whose centre
              falls inside this rectangle are kept

    Returns
    -------
    List of (x1, y1, x2, y2, confidence) tuples
    """
    # Run inference (verbose=False keeps the terminal clean)
    results = model(frame, conf=conf, iou=iou, verbose=False)[0]

    persons = []
    for box in results.boxes:
        cls_id = int(box.cls[0])               # class index
        if cls_id != PERSON_CLASS_ID:
            continue                            # skip non-person detections

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence      = float(box.conf[0])

        # ── ROI filter ──────────────────────────────────
        if roi is not None:
            rx, ry, rw, rh = roi
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
                continue                        # centre outside ROI

        persons.append((x1, y1, x2, y2, confidence))

    return persons


# ══════════════════════════════════════════════════════════
#  Drawing / Annotation
# ══════════════════════════════════════════════════════════
def draw_annotations(
    frame:   np.ndarray,
    persons: List[Tuple[int, int, int, int, float]],
    roi:     Optional[Tuple[int, int, int, int]] = None,
    elapsed: float = 0.0,
) -> np.ndarray:
    """
    Draw bounding boxes, per-person labels, count badge, and ROI rectangle.

    Parameters
    ----------
    frame   : original BGR frame (will be copied, not modified in place)
    persons : list of (x1,y1,x2,y2,conf) detections
    roi     : optional (x,y,w,h) for drawing the ROI border
    elapsed : inference time in seconds (shown in badge)

    Returns
    -------
    Annotated copy of frame
    """
    annotated = frame.copy()
    count      = len(persons)

    # ── draw each bounding box ────────────────────────────
    for idx, (x1, y1, x2, y2, conf) in enumerate(persons, start=1):
        # bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # label background
        label      = f"#{idx}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE_BOX, 1)
        lx1, ly1   = x1, max(y1 - th - 6, 0)
        lx2, ly2   = x1 + tw + 6, max(y1, th + 6)
        cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), LABEL_BG_COLOR, -1)

        # label text
        cv2.putText(
            annotated, label,
            (x1 + 3, max(y1 - 3, th)),
            FONT, FONT_SCALE_BOX, LABEL_TEXT_COLOR, 1, cv2.LINE_AA,
        )

    # ── ROI border ────────────────────────────────────────
    if roi is not None:
        rx, ry, rw, rh = roi
        cv2.rectangle(
            annotated, (rx, ry), (rx + rw, ry + rh),
            (0, 165, 255), 2,     # orange ROI border
        )
        cv2.putText(
            annotated, "ROI",
            (rx + 4, ry + 18),
            FONT, 0.6, (0, 165, 255), 2, cv2.LINE_AA,
        )

    # ── count badge (top-left) ────────────────────────────
    badge_text = f"Students: {count}"
    (bw, bh), _ = cv2.getTextSize(badge_text, FONT, FONT_SCALE_BADGE, FONT_THICKNESS)
    margin = 12
    # semi-transparent dark background
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (margin - 4, margin - 4),
        (margin + bw + 12, margin + bh + 12),
        BADGE_BG_COLOR, -1,
    )
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    cv2.putText(
        annotated, badge_text,
        (margin + 4, margin + bh),
        FONT, FONT_SCALE_BADGE, BADGE_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA,
    )

    # ── inference time (bottom-left) ─────────────────────
    fps_text = f"{1/elapsed:.1f} FPS" if elapsed > 0 else ""
    if fps_text:
        fh = annotated.shape[0]
        cv2.putText(
            annotated, fps_text,
            (margin + 4, fh - 12),
            FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )

    return annotated


# ══════════════════════════════════════════════════════════
#  ROI Selector
# ══════════════════════════════════════════════════════════
def select_roi(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Show an interactive window for the user to draw a Region-of-Interest.

    Returns (x, y, w, h) or None if the user cancelled.
    """
    print("[INFO] Draw the Region-of-Interest with your mouse, then press ENTER/SPACE.")
    print("       Press ESC or C to cancel ROI.")
    roi = cv2.selectROI(
        "Select ROI – press ENTER to confirm, ESC to skip",
        frame,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyAllWindows()
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("[INFO] ROI selection cancelled – counting whole frame.")
        return None
    print(f"[INFO] ROI selected: x={x}, y={y}, w={w}, h={h}")
    return (x, y, w, h)


# ══════════════════════════════════════════════════════════
#  CSV Logger
# ══════════════════════════════════════════════════════════
class CSVLogger:
    """
    Append per-frame count records to a CSV file.

    CSV columns: timestamp, frame_index, student_count
    """

    def __init__(self, csv_path: str) -> None:
        self.path   = csv_path
        self._file  = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "frame_index", "student_count"])
        print(f"[INFO] CSV logging → {csv_path}")

    def log(self, frame_idx: int, count: int) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._writer.writerow([ts, frame_idx, count])

    def close(self) -> None:
        self._file.close()
        print(f"[INFO] CSV saved   → {self.path}")


# ══════════════════════════════════════════════════════════
#  Resize helper
# ══════════════════════════════════════════════════════════
def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    """
    Proportionally resize frame to target_width.
    If frame is already smaller, return as-is.
    """
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale  = target_width / w
    new_wh = (target_width, int(h * scale))
    return cv2.resize(frame, new_wh, interpolation=cv2.INTER_LINEAR)


# ══════════════════════════════════════════════════════════
#  Image Pipeline
# ══════════════════════════════════════════════════════════
def process_image(
    model,
    source:   str,
    conf:     float = 0.4,
    iou:      float = 0.45,
    output:   Optional[str] = None,
    display:  bool  = True,
    resize_w: int   = 640,
    use_roi:  bool  = False,
) -> int:
    """
    Detect and count students in a single image.

    Returns the final student count.
    """
    print(f"\n[IMAGE] Processing: {source}")
    frame = cv2.imread(source)
    if frame is None:
        print(f"[ERROR] Cannot read image: {source}")
        return 0

    # Resize for faster inference (original stored for output)
    small  = resize_frame(frame, resize_w)
    scale  = frame.shape[1] / small.shape[1]   # used to scale boxes back up

    # Optional ROI selection
    roi = select_roi(small) if use_roi else None

    # Inference
    t0      = time.perf_counter()
    persons = detect_persons(model, small, conf, iou, roi)
    elapsed = time.perf_counter() - t0

    # Scale boxes back to original resolution
    persons_orig = [
        (int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale), c)
        for x1, y1, x2, y2, c in persons
    ]
    roi_orig = (
        (int(roi[0]*scale), int(roi[1]*scale),
         int(roi[2]*scale), int(roi[3]*scale))
        if roi else None
    )

    annotated = draw_annotations(frame, persons_orig, roi_orig, elapsed)
    count     = len(persons)

    print(f"[RESULT] Students detected: {count}")
    print(f"[TIMING] Inference: {elapsed*1000:.1f} ms")

    # Save output
    if output:
        cv2.imwrite(output, annotated)
        print(f"[SAVED]  Annotated image → {output}")

    # Display
    if display:
        win = "Student Counter – press any key to close"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return count


# ══════════════════════════════════════════════════════════
#  Shared Video-loop helper
# ══════════════════════════════════════════════════════════
def _video_loop(
    model,
    cap:      cv2.VideoCapture,
    source_label: str,
    conf:     float,
    iou:      float,
    output:   Optional[str],
    display:  bool,
    resize_w: int,
    use_roi:  bool,
    csv_path: Optional[str],
) -> None:
    """
    Internal loop shared by process_video and process_webcam.
    Reads frames, runs detection, annotates, displays and/or saves.
    """
    # ── get source properties ──────────────────────────────
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # -1 for webcam

    print(f"[INFO]  Source  : {source_label}")
    print(f"[INFO]  Size    : {orig_w} × {orig_h}  |  FPS: {fps_src:.1f}")
    if total > 0:
        print(f"[INFO]  Frames  : {total}")

    # ── read first frame for optional ROI selection ────────
    ok, first = cap.read()
    if not ok:
        print("[ERROR] Cannot read from source.")
        return
    first_small = resize_frame(first, resize_w)
    roi         = select_roi(first_small) if use_roi else None
    # reset video to beginning (webcam stays at current pos)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── video writer setup ─────────────────────────────────
    writer = None
    if output:
        ext    = Path(output).suffix.lower()
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "XVID"))
        writer = cv2.VideoWriter(output, fourcc, fps_src, (orig_w, orig_h))
        print(f"[INFO]  Saving annotated video → {output}")

    # ── CSV logger ─────────────────────────────────────────
    logger = CSVLogger(csv_path) if csv_path else None

    # ── main loop ─────────────────────────────────────────
    frame_idx  = 0
    count_hist: List[int] = []

    print("[INFO]  Running … press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break                               # end of video or camera lost

        small   = resize_frame(frame, resize_w)
        scale_x = orig_w  / small.shape[1]
        scale_y = orig_h  / small.shape[0]

        # ── scale ROI to inference size ────────────────────
        roi_small = None
        if roi:
            rx, ry, rw, rh = roi
            roi_small = (
                int(rx / scale_x), int(ry / scale_y),
                int(rw / scale_x), int(rh / scale_y),
            )

        # Detect
        t0      = time.perf_counter()
        persons = detect_persons(model, small, conf, iou, roi_small)
        elapsed = time.perf_counter() - t0

        # Scale boxes back
        persons_full = [
            (int(x1*scale_x), int(y1*scale_y),
             int(x2*scale_x), int(y2*scale_y), c)
            for x1, y1, x2, y2, c in persons
        ]
        roi_full = (
            (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
            if roi else None
        )

        count = len(persons_full)
        count_hist.append(count)

        annotated = draw_annotations(frame, persons_full, roi_full, elapsed)

        # Write frame
        if writer:
            writer.write(annotated)

        # Log to CSV
        if logger:
            logger.log(frame_idx, count)

        # Display
        if display:
            cv2.imshow("Student Counter  –  press Q to quit", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("[INFO]  Quit requested.")
                break

        frame_idx += 1

        # Progress for video files
        if total > 0 and frame_idx % 30 == 0:
            pct = 100 * frame_idx / total
            print(f"  … frame {frame_idx}/{total}  ({pct:.0f}%)  "
                  f"current count={count}")

    # ── cleanup ───────────────────────────────────────────
    if writer:
        writer.release()
        print(f"[SAVED] Video → {output}")
    if logger:
        logger.close()
    if display:
        cv2.destroyAllWindows()

    if count_hist:
        avg = sum(count_hist) / len(count_hist)
        mx  = max(count_hist)
        print(f"\n[SUMMARY] Frames processed : {frame_idx}")
        print(f"[SUMMARY] Avg students/frame: {avg:.1f}")
        print(f"[SUMMARY] Peak count        : {mx}")


# ══════════════════════════════════════════════════════════
#  Video Pipeline
# ══════════════════════════════════════════════════════════
def process_video(
    model,
    source:   str,
    conf:     float = 0.4,
    iou:      float = 0.45,
    output:   Optional[str] = None,
    display:  bool  = True,
    resize_w: int   = 640,
    use_roi:  bool  = False,
    csv_path: Optional[str] = None,
) -> None:
    """
    Run the student counter on every frame of a video file.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {source}")
        return
    _video_loop(
        model, cap, source, conf, iou,
        output, display, resize_w, use_roi, csv_path,
    )
    cap.release()


# ══════════════════════════════════════════════════════════
#  Webcam Pipeline
# ══════════════════════════════════════════════════════════
def process_webcam(
    model,
    cam_id:   int   = 0,
    conf:     float = 0.4,
    iou:      float = 0.45,
    output:   Optional[str] = None,
    display:  bool  = True,
    resize_w: int   = 640,
    use_roi:  bool  = False,
    csv_path: Optional[str] = None,
) -> None:
    """
    Run the student counter on a live webcam feed.
    """
    print(f"[INFO]  Opening webcam (device {cam_id}) …")
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam id={cam_id}. "
              "Check connection or try a different --webcam-id.")
        return
    # Suggest a reasonable capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    _video_loop(
        model, cap, f"webcam:{cam_id}", conf, iou,
        output, display, resize_w, use_roi, csv_path,
    )
    cap.release()
