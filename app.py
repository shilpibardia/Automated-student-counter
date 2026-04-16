"""
app.py – Streamlit Web Interface for Automatic Student Counter
==============================================================
Run with:
    streamlit run app.py

Features:
  • Upload an image or video and get annotated results instantly
  • Adjustable confidence and IoU sliders
  • Student count displayed as a metric
  • Download annotated outputs
"""

import io
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── local modules ──────────────────────────────────────────
from utils import (
    detect_persons,
    draw_annotations,
    load_model,
    resize_frame,
)

# ══════════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Student Counter – YOLOv8",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
#  Custom CSS  –  clean dark academic theme
# ══════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── sidebar ─────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0f172a;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider   label { color: #94a3b8; }

    /* ── main background ─────────────────── */
    .main .block-container { padding-top: 1.5rem; }

    /* ── metric cards ─────────────────────── */
    div[data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="metric-container"] label  { color: #94a3b8 !important; }
    div[data-testid="metric-container"] div    { color: #00dc82 !important; font-size: 2rem !important; }

    /* ── file uploader ────────────────────── */
    .stFileUploader { border: 2px dashed #334155; border-radius: 12px; }

    /* ── headings ─────────────────────────── */
    h1, h2, h3 { font-weight: 700; }
    h1 { color: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════
#  Sidebar – settings
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    model_choice = st.selectbox(
        "YOLOv8 Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="Larger models are slower but more accurate.",
    )
    conf_thresh = st.slider(
        "Confidence Threshold", 0.10, 0.95, 0.40, 0.05,
        help="Lower → detects more people (may include false positives).",
    )
    iou_thresh = st.slider(
        "IoU Threshold (NMS)", 0.10, 0.90, 0.45, 0.05,
        help="Controls overlap suppression.",
    )
    resize_w = st.slider(
        "Inference Width (px)", 320, 1280, 640, 64,
        help="Wider = more accurate but slower.",
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "Detects and counts **students (people)** using YOLOv8 "
        "trained on the COCO dataset.\n\n"
        "Built with Python · OpenCV · Ultralytics"
    )

# ══════════════════════════════════════════════════════════
#  Load model (cached across re-runs)
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading YOLOv8 model …")
def get_model(name: str):
    return load_model(name)


model = get_model(model_choice)

# ══════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════
st.markdown("# 🎓 Automatic Student Counter")
st.markdown(
    "Upload an **image** or **video** to detect and count all students present."
)
st.markdown("---")

# ══════════════════════════════════════════════════════════
#  Tab layout: Image | Video
# ══════════════════════════════════════════════════════════
tab_img, tab_vid = st.tabs(["📷  Image", "🎬  Video"])

# ──────────────────────────────────────────────────────────
#  IMAGE TAB
# ──────────────────────────────────────────────────────────
with tab_img:
    uploaded_img = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
    )

    if uploaded_img:
        # Decode to OpenCV BGR
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Could not decode the image. Please upload a valid file.")
        else:
            with st.spinner("Running detection …"):
                small   = resize_frame(frame, resize_w)
                scale   = frame.shape[1] / small.shape[1]

                t0      = time.perf_counter()
                persons = detect_persons(model, small, conf_thresh, iou_thresh)
                elapsed = time.perf_counter() - t0

                persons_full = [
                    (int(x1*scale), int(y1*scale),
                     int(x2*scale), int(y2*scale), c)
                    for x1, y1, x2, y2, c in persons
                ]
                annotated = draw_annotations(frame, persons_full, elapsed=elapsed)

            count = len(persons)

            # ── metrics row ───────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("👥 Students Detected", count)
            m2.metric("⚡ Inference Time",    f"{elapsed*1000:.0f} ms")
            m3.metric("📐 Image Size",        f"{frame.shape[1]}×{frame.shape[0]}")

            # ── side-by-side images ───────────────────────
            col_orig, col_ann = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(orig_rgb, use_container_width=True)
            with col_ann:
                st.markdown(f"**Annotated  –  {count} student(s)**")
                ann_rgb  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(ann_rgb, use_container_width=True)

            # ── download button ───────────────────────────
            _, enc = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
            st.download_button(
                "⬇️  Download Annotated Image",
                data        = enc.tobytes(),
                file_name   = f"counted_{uploaded_img.name}",
                mime        = "image/jpeg",
            )

# ──────────────────────────────────────────────────────────
#  VIDEO TAB
# ──────────────────────────────────────────────────────────
with tab_vid:
    uploaded_vid = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
    )

    max_frames = st.slider(
        "Max frames to process (0 = all)", 0, 500, 100,
        help="Limit processing for large videos in the web app.",
    )

    if uploaded_vid:
        # Save to temp file (OpenCV needs a real path)
        suffix = Path(uploaded_vid.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("Cannot open the video file.")
        else:
            fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            limit   = max_frames if max_frames > 0 else total

            st.markdown(
                f"Video: **{orig_w}×{orig_h}**  |  "
                f"**{fps:.1f} FPS**  |  "
                f"**{total} frames**"
            )

            progress   = st.progress(0, text="Processing …")
            count_log: list[int] = []
            frame_idx  = 0
            out_frames = []           # store annotated frames for preview gif

            with st.spinner("Detecting students …"):
                while frame_idx < limit:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    small   = resize_frame(frame, resize_w)
                    scale_x = orig_w  / small.shape[1]
                    scale_y = orig_h  / small.shape[0]

                    persons = detect_persons(
                        model, small, conf_thresh, iou_thresh
                    )
                    persons_full = [
                        (int(x1*scale_x), int(y1*scale_y),
                         int(x2*scale_x), int(y2*scale_y), c)
                        for x1, y1, x2, y2, c in persons
                    ]
                    annotated = draw_annotations(frame, persons_full)
                    count_log.append(len(persons_full))

                    # Keep 1 in every 10 frames for preview
                    if frame_idx % 10 == 0:
                        preview_rgb = cv2.cvtColor(
                            resize_frame(annotated, 480),
                            cv2.COLOR_BGR2RGB,
                        )
                        out_frames.append(Image.fromarray(preview_rgb))

                    frame_idx += 1
                    progress.progress(
                        min(frame_idx / limit, 1.0),
                        text=f"Frame {frame_idx}/{limit}  –  count={count_log[-1]}",
                    )

            cap.release()
            os.unlink(tmp_path)
            progress.empty()

            # ── summary metrics ────────────────────────────
            if count_log:
                m1, m2, m3 = st.columns(3)
                m1.metric("👥 Peak Count",         max(count_log))
                m2.metric("📊 Average / Frame",    f"{sum(count_log)/len(count_log):.1f}")
                m3.metric("🎞️  Frames Processed",  frame_idx)

                # ── sparkline chart ────────────────────────
                st.markdown("#### Count per Frame")
                st.line_chart(count_log)

                # ── preview strip ──────────────────────────
                if out_frames:
                    st.markdown("#### Annotated Preview Frames")
                    cols = st.columns(min(5, len(out_frames)))
                    for col, img in zip(cols, out_frames[:5]):
                        col.image(img, use_container_width=True)

                # ── CSV download ───────────────────────────
                csv_buf = io.StringIO()
                csv_buf.write("frame_index,student_count\n")
                for i, c in enumerate(count_log):
                    csv_buf.write(f"{i},{c}\n")
                st.download_button(
                    "⬇️  Download Count CSV",
                    data      = csv_buf.getvalue(),
                    file_name = "student_counts.csv",
                    mime      = "text/csv",
                )
