import os
import sys
import time
import random

import cv2
import streamlit as st

# Make local imports work when running `streamlit run app.py`.
sys.path.append(os.path.dirname(__file__))

from detector import PersonDetector  # noqa: E402
from predictor import CrowdPredictor  # noqa: E402
from utils import FlowCounter, classify_risk, compute_zone_stats, ensure_sample_video  # noqa: E402


st.set_page_config(page_title="RailYatra Predict – Crowd Chaos Buster", layout="wide")

st.title("🚆 RailYatra Predict – AI Crowd Chaos Buster")

BASE_DIR = os.path.dirname(__file__)
VIDEO_PATH_DEFAULT = os.path.join(BASE_DIR, "sample_video.mp4")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Ensure there's always some input video so the prototype runs immediately.
ensure_sample_video(VIDEO_PATH_DEFAULT)

if "video_path" not in st.session_state:
    st.session_state.video_path = VIDEO_PATH_DEFAULT
    st.session_state.video_source_key = None

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)


with st.sidebar:
    st.header("Detection & Performance")
    model_path = st.text_input("YOLO model", value="yolov8n.pt")
    conf = st.slider("Confidence", min_value=0.05, max_value=0.8, value=0.25, step=0.05)
    iou = st.slider("IOU", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    img_size = st.slider("Inference size (imgsz)", min_value=320, max_value=1280, value=640, step=64)
    enable_fallback = st.checkbox("Enable synthetic fallback (if YOLO finds nothing)", value=True)
    process_every_n_frames = st.slider("Process every Nth frame", min_value=1, max_value=6, value=2, step=1)
    ui_sleep_ms = st.slider("UI throttle (ms)", min_value=0, max_value=60, value=10, step=5)

    st.divider()
    st.header("Visualization")
    horizon_minutes = st.slider("Prediction horizon (minutes)", min_value=5, max_value=60, value=30, step=5)
    show_heatmap = st.checkbox("Density heatmap overlay", value=False)
    show_virtual_line = st.checkbox("Show entry/exit virtual line", value=True)
    show_zone_divider = st.checkbox("Show platform zone divider", value=True)
    chart_update_every_n_frames = st.slider("Chart update interval (frames)", 1, 30, 8)

    st.divider()
    st.header("🚆 Train Arrival Simulation")
    simulate_spike = st.checkbox("Auto random arrivals (background)", value=False)
    spike_interval_sec_min = st.slider("Auto arrival interval (min sec)", 60, 300, 180, step=15)
    spike_interval_sec_max = st.slider("Auto arrival interval (max sec)", 60, 600, 300, step=15)
    spike_duration_sec = st.slider("Arrival duration (sec)", 5, 90, 25, step=5)
    spike_boost_min = st.slider("Crowd surge boost (people min)", 20, 80, 25, step=5)
    spike_boost_max = st.slider("Crowd surge boost (people max)", 30, 140, 50, step=5)
    arrival_inflow_boost_per_min = st.slider("Inflow surge (people/min)", 0, 60, 18, step=2)

    simulate_train_now = st.button("Simulate Train Arrival", use_container_width=True, type="primary")

    st.divider()
    st.header("CCTV Video Input")
    uploaded_file = st.file_uploader(
        "Upload a video for testing (mp4/avi/mov/mkv)",
        type=["mp4", "avi", "mov", "mkv"],
    )
    if uploaded_file is not None:
        source_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if source_key != st.session_state.video_source_key:
            # Persist uploaded content to disk so OpenCV can read it.
            # (Streamlit provides a temporary handle, not a stable filesystem path.)
            safe_name = os.path.basename(uploaded_file.name).replace(" ", "_")
            save_path = os.path.join(UPLOAD_DIR, f"uploaded_{int(time.time())}_{safe_name}")
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.video_path = save_path
            st.session_state.video_source_key = source_key


if "predictor" not in st.session_state:
    st.session_state.predictor = CrowdPredictor(
        history_points=400,
        ma_window=10,
        trend_window=30,
        horizon_minutes=horizon_minutes,
        clamp_min=0.0,
        clamp_max=200.0,
    )

if "detector" not in st.session_state:
    with st.spinner("Loading YOLOv8 model (CPU) ... this may download weights first time"):
        st.session_state.detector = PersonDetector(
            model_path=model_path,
            conf=conf,
            iou=iou,
            device="cpu",
            img_size=img_size,
            enable_fallback=enable_fallback,
        )

if "spike_state" not in st.session_state:
    st.session_state.spike_state = {
        "next_spike_at": time.time() + random.uniform(spike_interval_sec_min, spike_interval_sec_max),
        "spike_until": 0.0,
        "spike_boost": random.randint(spike_boost_min, spike_boost_max),
    }

if "flow_counter" not in st.session_state:
    st.session_state.flow_counter = FlowCounter(window_sec=60.0, cooldown_sec=1.0)

if "train_arrival" not in st.session_state:
    st.session_state.train_arrival = {"until": 0.0, "boost": 0, "inflow_boost": 0.0}


main_cols = st.columns([1.55, 1.0], gap="large")
with main_cols[0]:
    frame_placeholder = st.empty()
    chart_placeholder = st.empty()

with main_cols[1]:
    metrics_cols = st.columns(2)
    raw_count_text = metrics_cols[0].empty()
    predicted_text = metrics_cols[1].empty()

    # Use placeholders so UI updates in-place inside the infinite loop.
    risk_title = st.empty()
    risk_body = st.empty()
    risk_alert = st.empty()
    flow_cols = st.columns(2)
    inflow_text = flow_cols[0].empty()
    outflow_text = flow_cols[1].empty()

    zone_title = st.empty()
    zone_body = st.empty()
    method_text = st.empty()

VIDEO_PATH = st.session_state.video_path
video_label = os.path.basename(VIDEO_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    st.error(f"Failed to open video: {VIDEO_PATH}")
    st.stop()
video_source_caption = st.caption(f"Video source: {video_label}")
st.caption("Powered by YOLOv8 + Tracking + Predictive AI")

frame_idx = 0

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        # Loop video continuously.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Use video timestamps (from the file) so "30 minutes" extrapolation is consistent.
    timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    det: object = st.session_state.detector
    predictor: CrowdPredictor = st.session_state.predictor
    flow_counter: FlowCounter = st.session_state.flow_counter

    # If user changes horizon in the sidebar, refresh predictor horizon.
    predictor.horizon_minutes = float(horizon_minutes)

    do_process = (frame_idx % int(process_every_n_frames)) == 0
    detection = None
    if do_process:
        detection = det.detect_people(frame_bgr, show_heatmap=show_heatmap)
        st.session_state.last_detection = detection
    else:
        detection = st.session_state.get("last_detection", None)

    if detection is None:
        frame_idx += 1
        time.sleep(float(ui_sleep_ms) / 1000.0)
        continue

    crowd_count_raw = int(detection.count)
    detection_method = detection.method
    annotated = detection.annotated_frame_bgr.copy()
    tracks = detection.tracks or []
    h, w = annotated.shape[:2]

    # Optional fake spike logic.
    spike_active = False
    spike_boost = 0
    now_wall = time.time()

    # Manual "Train Arrival" trigger (preferred UX).
    if simulate_train_now:
        st.session_state.train_arrival = {
            "until": now_wall + float(spike_duration_sec),
            "boost": int(random.randint(int(spike_boost_min), int(spike_boost_max))),
            "inflow_boost": float(arrival_inflow_boost_per_min),
        }

    # Optional background auto-arrivals.
    if simulate_spike:
        ss = st.session_state.spike_state
        if now_wall >= ss["next_spike_at"] and now_wall >= ss["spike_until"]:
            ss["spike_until"] = now_wall + float(spike_duration_sec)
            ss["spike_boost"] = random.randint(int(spike_boost_min), int(spike_boost_max))
            ss["next_spike_at"] = now_wall + random.uniform(
                float(spike_interval_sec_min), float(spike_interval_sec_max)
            )

    arrival = st.session_state.train_arrival
    if now_wall < float(arrival.get("until", 0.0)):
        spike_active = True
        spike_boost = int(arrival.get("boost", 0))
    elif simulate_spike:
        ss = st.session_state.spike_state
        if now_wall < ss["spike_until"]:
            spike_active = True
            spike_boost = int(ss["spike_boost"])

    # Flow update (line crossing) on processed frames only.
    if do_process:
        flow_counter.update(tracks=tracks, timestamp_sec=float(timestamp_sec), frame_w=int(w))
    inflow_per_min, outflow_per_min = flow_counter.rates_per_min(float(timestamp_sec))
    if spike_active:
        inflow_per_min += float(arrival.get("inflow_boost", 0.0))

    # History is updated using raw counts plus spike boost (so the chart reflects the simulation).
    crowd_count_for_model = crowd_count_raw + spike_boost
    crowd_count_for_model = float(min(200.0, max(0.0, crowd_count_for_model)))

    if do_process or "last_pred" not in st.session_state:
        pred = predictor.update_and_predict(
            crowd_count=crowd_count_for_model,
            timestamp_sec=float(timestamp_sec),
            inflow_per_min=float(inflow_per_min),
            outflow_per_min=float(outflow_per_min),
        )
        st.session_state.last_pred = pred
        st.session_state.last_inflow = float(inflow_per_min)
        st.session_state.last_outflow = float(outflow_per_min)
    else:
        pred = st.session_state.last_pred
        inflow_per_min = float(st.session_state.get("last_inflow", inflow_per_min))
        outflow_per_min = float(st.session_state.get("last_outflow", outflow_per_min))

    # For display: show predicted count with spike impact (already in predictor),
    # but make spike visibility explicit.
    predicted_display = float(pred.predicted_count)
    risk = classify_risk(predicted_display)

    # Zones (A/B) from tracks.
    zone_stats = compute_zone_stats(tracks=tracks, frame_w=int(w))

    # Visual overlays: line + zone divider.
    if show_virtual_line:
        xmid = int(w // 2)
        cv2.line(annotated, (xmid, 0), (xmid, h), (255, 255, 255), 2)
        cv2.putText(
            annotated,
            "ENTRY  ->",
            (xmid + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "<-  EXIT",
            (max(10, xmid - 140), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if show_zone_divider:
        xmid = int(w // 2)
        cv2.line(annotated, (xmid, 0), (xmid, h), (180, 180, 180), 2)
        cv2.putText(
            annotated,
            "Zone A",
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "Zone B",
            (xmid + 20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Update UI.
    frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

    raw_count_text.metric(label="Current crowd", value=f"{crowd_count_raw}")

    # Large risk in color.
    risk_color = risk["color"]
    risk_label = risk["label"]
    spike_note = " (Train Arriving Active)" if spike_active else ""

    predicted_text.metric(
        label=f"Predicted (Next {int(horizon_minutes)} min){spike_note}",
        value=f"{predicted_display:.0f}",
    )

    risk_title.subheader("Risk level")
    risk_body.markdown(
        f"""
        <div style="
            font-size: 34px;
            font-weight: 900;
            color: white;
            background: {risk_color};
            padding: 14px 18px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        ">
            {risk_label.upper()}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if spike_active:
        risk_alert.warning("🚆 Train arriving → Crowd surge expected")
    else:
        risk_alert.empty()

    inflow_text.metric(label="Inflow (people/min)", value=f"{inflow_per_min:.1f}")
    outflow_text.metric(label="Outflow (people/min)", value=f"{outflow_per_min:.1f}")

    zone_title.subheader("Platform zones")
    za = zone_stats["A"]
    zb = zone_stats["B"]
    zone_body.markdown(
        f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap;">
          <div style="flex:1; min-width:220px; border-radius:12px; padding:12px 14px; background:{za.risk_color}; color:white;">
            <div style="font-size:14px; opacity:0.9;">Platform A</div>
            <div style="font-size:22px; font-weight:900;">{za.risk_label} ({za.count} people)</div>
          </div>
          <div style="flex:1; min-width:220px; border-radius:12px; padding:12px 14px; background:{zb.risk_color}; color:white;">
            <div style="font-size:14px; opacity:0.9;">Platform B</div>
            <div style="font-size:22px; font-weight:900;">{zb.risk_label} ({zb.count} people)</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    method_text.info(
        f"{detection_method} | trend: {float(getattr(pred, 'trend_per_min', 0.0)):+.2f}/min | net flow: {float(getattr(pred, 'net_flow_per_min', 0.0)):+.1f}/min"
    )

    if frame_idx % int(chart_update_every_n_frames) == 0:
        df = predictor.history_as_dataframe()
        if len(df) > 1:
            chart_placeholder.line_chart(df.set_index("time_min")[["count"]])
        else:
            chart_placeholder.write("Collecting history for the prediction curve...")

    frame_idx += 1

    # Slow down a bit so Streamlit can keep up (hackathon-friendly CPU usage).
    time.sleep(float(ui_sleep_ms) / 1000.0)

