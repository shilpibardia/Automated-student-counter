"""
app.py – Streamlit Web Interface for Automatic Student Counter
Cloud-safe: uses Pillow only — no cv2 import at module level.
Works on Python 3.11, 3.12, 3.13, 3.14+
"""

import io
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    st.error("ultralytics not installed. Add it to requirements.txt")
    st.stop()

PERSON_CLASS_ID = 0
BOX_COLOR       = (0, 220, 120)
BADGE_BG        = (15, 15, 15, 180)
BADGE_TEXT      = (0, 220, 120)

st.set_page_config(
    page_title="Student Counter – YOLOv8",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background: #0f172a; color: #e2e8f0; }
section[data-testid="stSidebar"] label { color: #94a3b8; }
div[data-testid="metric-container"] {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #94a3b8 !important; }
div[data-testid="metric-container"] div   { color: #00dc82 !important; font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")
    model_choice = st.selectbox("YOLOv8 Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    conf_thresh  = st.slider("Confidence Threshold", 0.10, 0.95, 0.40, 0.05)
    iou_thresh   = st.slider("IoU Threshold (NMS)",  0.10, 0.90, 0.45, 0.05)
    resize_w     = st.slider("Inference Width (px)",  320, 1280, 640, 64)
    st.markdown("---")
    st.markdown("Built with Python · Pillow · Ultralytics")


def resize_pil(img, target_width):
    w, h = img.size
    if w <= target_width:
        return img
    return img.resize((target_width, int(h * target_width / w)), Image.LANCZOS)


def detect_persons(model, img, conf, iou):
    results = model(img, conf=conf, iou=iou, verbose=False)[0]
    persons = []
    for box in results.boxes:
        if int(box.cls[0]) != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        persons.append((x1, y1, x2, y2, float(box.conf[0])))
    return persons


def draw_annotations(img, persons):
    out  = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    try:
        font_box   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font_box = font_badge = ImageFont.load_default()

    for idx, (x1, y1, x2, y2, conf) in enumerate(persons, 1):
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=3)
        label = f"#{idx}  {conf:.0%}"
        bb = draw.textbbox((0, 0), label, font=font_box)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        draw.rectangle([x1, max(y1-th-8,0), x1+tw+10, max(y1,th+8)], fill=BOX_COLOR)
        draw.text((x1+5, max(y1-th-4,0)), label, fill=(0,0,0), font=font_box)

    badge = f"Students: {len(persons)}"
    bb = draw.textbbox((0,0), badge, font=font_badge)
    bw, bh = bb[2]-bb[0], bb[3]-bb[1]
    overlay = Image.new("RGBA", out.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([8, 8, bw+24, bh+24], fill=BADGE_BG)
    out = Image.alpha_composite(out, overlay)
    ImageDraw.Draw(out).text((16,12), badge, fill=BADGE_TEXT, font=font_badge)
    return out.convert("RGB")


@st.cache_resource(show_spinner="Loading YOLOv8 model ...")
def get_model(name):
    return YOLO(name)


model    = get_model(model_choice)
IS_CLOUD = os.path.exists("/mount/src")

st.markdown("# Automatic Student Counter")
st.markdown("Upload an **image** or **video** to detect and count all students present.")
st.markdown("---")

tab_img, tab_vid, tab_cam = st.tabs(["Image", "Video", "Webcam"])

# IMAGE TAB
with tab_img:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"], key="img")
    if uploaded_img:
        img_pil        = Image.open(uploaded_img).convert("RGB")
        orig_w, orig_h = img_pil.size

        with st.spinner("Running detection ..."):
            small   = resize_pil(img_pil, resize_w)
            scale_x = orig_w / small.width
            scale_y = orig_h / small.height
            t0      = time.perf_counter()
            persons = detect_persons(model, small, conf_thresh, iou_thresh)
            elapsed = time.perf_counter() - t0
            persons_full = [
                (int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y), c)
                for x1, y1, x2, y2, c in persons
            ]
            annotated = draw_annotations(img_pil, persons_full)

        count = len(persons_full)
        m1, m2, m3 = st.columns(3)
        m1.metric("Students Detected", count)
        m2.metric("Inference Time",    f"{elapsed*1000:.0f} ms")
        m3.metric("Image Size",        f"{orig_w}x{orig_h}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.image(img_pil, use_container_width=True)
        with c2:
            st.markdown(f"**Annotated — {count} student(s)**")
            st.image(annotated, use_container_width=True)

        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=95)
        st.download_button("Download Annotated Image", buf.getvalue(),
                           f"counted_{uploaded_img.name}", "image/jpeg")

# VIDEO TAB
with tab_vid:
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"], key="vid")
    max_frames   = st.slider("Max frames to process (0 = all)", 0, 300, 60)

    if uploaded_vid:
        try:
            import cv2 as _cv2
            HAS_CV2 = True
        except Exception:
            HAS_CV2 = False

        if not HAS_CV2:
            st.error(
                "Video processing requires OpenCV which could not load on this Python version. "
                "Please use the Image tab, or run locally with: streamlit run app.py"
            )
        else:
            suffix = Path(uploaded_vid.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            cap   = _cv2.VideoCapture(tmp_path)
            fps   = cap.get(_cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            vw    = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            vh    = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            limit = max_frames if max_frames > 0 else total

            st.markdown(f"Video: **{vw}x{vh}** | **{fps:.1f} FPS** | **{total} frames**")
            progress  = st.progress(0, text="Processing ...")
            count_log = []
            previews  = []
            frame_idx = 0

            with st.spinner("Detecting students ..."):
                while frame_idx < limit:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_pil = Image.fromarray(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
                    small     = resize_pil(frame_pil, resize_w)
                    scale_x   = frame_pil.width  / small.width
                    scale_y   = frame_pil.height / small.height
                    persons   = detect_persons(model, small, conf_thresh, iou_thresh)
                    persons_f = [
                        (int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y), c)
                        for x1, y1, x2, y2, c in persons
                    ]
                    annotated = draw_annotations(frame_pil, persons_f)
                    count_log.append(len(persons_f))
                    if frame_idx % 10 == 0:
                        previews.append(resize_pil(annotated, 480))
                    frame_idx += 1
                    progress.progress(min(frame_idx/limit, 1.0), text=f"Frame {frame_idx}/{limit}")

            cap.release()
            os.unlink(tmp_path)
            progress.empty()

            if count_log:
                m1, m2, m3 = st.columns(3)
                m1.metric("Peak Count",       max(count_log))
                m2.metric("Average / Frame",  f"{sum(count_log)/len(count_log):.1f}")
                m3.metric("Frames Processed", frame_idx)
                st.markdown("#### Count per Frame")
                st.line_chart(count_log)
                if previews:
                    st.markdown("#### Preview Frames")
                    cols = st.columns(min(5, len(previews)))
                    for col, frm in zip(cols, previews[:5]):
                        col.image(frm, use_container_width=True)
                csv_buf = io.StringIO()
                csv_buf.write("frame_index,student_count\n")
                for i, c in enumerate(count_log):
                    csv_buf.write(f"{i},{c}\n")
                st.download_button("Download Count CSV", csv_buf.getvalue(),
                                   "student_counts.csv", "text/csv")

# WEBCAM TAB
with tab_cam:
    st.markdown("### Live Webcam Student Counter")
    if IS_CLOUD:
        st.warning("Webcam is not available on the cloud server. Run locally: streamlit run app.py", icon="📷")
        st.info("Image and Video tabs work fully on the cloud — upload a photo or video to count students.", icon="✅")
    else:
        st.info("Allow camera access when your browser asks.", icon="ℹ️")
        auto_mode = st.toggle("Continuous Mode", value=False)
        if auto_mode:
            refresh_interval = st.slider("Refresh interval (seconds)", 1, 10, 2)

        cam_frame = st.camera_input("Capture", label_visibility="collapsed")
        if cam_frame is not None:
            img_pil        = Image.open(cam_frame).convert("RGB")
            orig_w, orig_h = img_pil.size
            with st.spinner("Detecting ..."):
                small   = resize_pil(img_pil, resize_w)
                scale_x = orig_w / small.width
                scale_y = orig_h / small.height
                t0      = time.perf_counter()
                persons = detect_persons(model, small, conf_thresh, iou_thresh)
                elapsed = time.perf_counter() - t0
                persons_full = [
                    (int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y), c)
                    for x1, y1, x2, y2, c in persons
                ]
                annotated = draw_annotations(img_pil, persons_full)
            count = len(persons_full)
            if count == 0:
                st.warning("No students detected. Try lowering the confidence threshold.")
            else:
                st.success(f"{count} student{'s' if count != 1 else ''} detected")
            m1, m2, m3 = st.columns(3)
            m1.metric("Students Detected", count)
            m2.metric("Inference Time",    f"{elapsed*1000:.0f} ms")
            m3.metric("Frame Size",        f"{orig_w}x{orig_h}")
            st.image(annotated, caption=f"{count} student(s)", use_container_width=True)
            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=95)
            st.download_button("Download Annotated Frame", buf.getvalue(), "webcam_count.jpg", "image/jpeg")
            if auto_mode:
                time.sleep(refresh_interval)
                st.rerun()
