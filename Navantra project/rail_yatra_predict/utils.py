import os
import cv2
import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Iterable, Tuple, Any


def classify_risk(predicted_count: float):
    """
    Risk buckets (as requested):
    - Green: count < 30
    - Yellow: 30–70
    - Red: > 70
    """
    try:
        c = float(predicted_count)
    except (TypeError, ValueError):
        c = 0.0

    if c < 30:
        return {"label": "Green", "color": "#2ecc71"}
    if c <= 70:
        return {"label": "Yellow", "color": "#f1c40f"}
    return {"label": "Red", "color": "#e74c3c"}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _synthetic_person_boxes_by_color(frame_bgr, debug=False):
    """
    Fallback detector for the synthetic demo video.

    The synthetic video draws filled yellow rectangles for "people".
    If YOLO finds nothing for a while, this helps keep the dashboard functional.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Yellow-ish range for the synthetic rectangles.
    lower = np.array([15, 120, 120], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up mask.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = frame_bgr.shape[:2]

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < (0.0025 * w * h):
            continue

        # Filter out tiny blobs.
        aspect = bh / (bw + 1e-6)
        if aspect < 1.4:
            continue

        x1 = int(clamp(x, 0, w - 1))
        y1 = int(clamp(y, 0, h - 1))
        x2 = int(clamp(x + bw, 0, w - 1))
        y2 = int(clamp(y + bh, 0, h - 1))
        boxes.append((x1, y1, x2, y2))

    if debug:
        return boxes, mask
    return boxes


def ensure_sample_video(video_path: str, width=640, height=360, fps=15, duration_sec=25):
    """
    Create a small synthetic CCTV-like mp4 if `sample_video.mp4` is missing.

    Important: YOLOv8 may not detect synthetic rectangles as "person".
    Therefore, detector.py includes a fallback that recovers boxes by color.
    """
    video_path = os.path.abspath(video_path)
    if os.path.exists(video_path) and os.path.getsize(video_path) > 1024 * 200:
        return

    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video at: {video_path}")

    rng = np.random.default_rng(42)
    total_frames = int(fps * duration_sec)

    # Simple "people tracks" that move around.
    num_tracks = 12
    tracks = []
    for i in range(num_tracks):
        x = rng.uniform(0.1 * width, 0.9 * width)
        y = rng.uniform(0.25 * height, 0.95 * height)
        vx = rng.uniform(-1.2, 1.2)
        vy = rng.uniform(-0.6, 0.6)
        w = rng.uniform(18, 28)
        h = rng.uniform(48, 85)
        tracks.append({"x": x, "y": y, "vx": vx, "vy": vy, "w": w, "h": h})

    # Background grid.
    for t in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (25, 25, 28)

        # Subtle "CCTV" vignette / grid lines.
        for gx in range(0, width, 40):
            cv2.line(frame, (gx, 0), (gx, height), (30, 30, 35), 1)
        for gy in range(0, height, 45):
            cv2.line(frame, (0, gy), (width, gy), (30, 30, 35), 1)

        # Add a little camera jitter.
        jitter_x = int(rng.integers(-2, 3))
        jitter_y = int(rng.integers(-2, 3))

        for tr in tracks:
            tr["x"] += tr["vx"]
            tr["y"] += tr["vy"]

            # Bounce off borders.
            if tr["x"] < 0.05 * width or tr["x"] > 0.95 * width:
                tr["vx"] *= -1
            if tr["y"] < 0.15 * height or tr["y"] > 0.98 * height:
                tr["vy"] *= -1

            w = tr["w"]
            h = tr["h"]
            x1 = int(tr["x"] - w / 2) + jitter_x
            x2 = int(tr["x"] + w / 2) + jitter_x
            y1 = int(tr["y"] - h / 2) + jitter_y
            y2 = int(tr["y"] + h / 2) + jitter_y

            x1 = int(clamp(x1, 0, width - 1))
            x2 = int(clamp(x2, 0, width - 1))
            y1 = int(clamp(y1, 0, height - 1))
            y2 = int(clamp(y2, 0, height - 1))

            # Draw filled yellow rectangles + head circle to resemble a person silhouette.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), thickness=-1)
            head_r = max(7, int((y2 - y1) * 0.12))
            head_cx = int((x1 + x2) / 2)
            head_cy = int(y1 + head_r)
            cv2.circle(frame, (head_cx, head_cy), head_r, (0, 200, 255), thickness=-1)

        # Occasional "crowd surge" in the synthetic input (purely for visuals).
        if (t // (fps * 6)) % 2 == 1:
            surge = 4
            for _ in range(surge):
                sx = int(rng.uniform(0.1 * width, 0.9 * width))
                sy = int(rng.uniform(0.25 * height, 0.95 * height))
                sw = int(rng.uniform(18, 28))
                sh = int(rng.uniform(55, 85))
                cv2.rectangle(frame, (sx, sy - sh), (sx + sw, sy), (0, 255, 255), thickness=-1)
                cv2.circle(frame, (sx + sw // 2, sy - sh + 8), max(7, sw // 3), (0, 200, 255), -1)

        writer.write(frame)

    writer.release()


def synthetic_person_boxes(frame_bgr):
    """Public wrapper for the synthetic demo fallback detector."""
    return _synthetic_person_boxes_by_color(frame_bgr)


def _zone_for_x(cx: int, frame_w: int) -> str:
    return "A" if cx < (frame_w // 2) else "B"


@dataclass
class ZoneStats:
    zone: str
    count: int
    risk_label: str
    risk_color: str


def compute_zone_stats(
    tracks: Iterable[Any], frame_w: int, risk_from: str = "count"
) -> Dict[str, ZoneStats]:
    """
    Split the platform into two zones:
    - Zone A: left half
    - Zone B: right half

    risk_from:
      - "count": risk derived from zone count using classify_risk()
    """
    counts = {"A": 0, "B": 0}
    for tr in tracks:
        cx = int(getattr(tr, "center_xy")[0])
        z = _zone_for_x(cx, int(frame_w))
        counts[z] += 1

    out: Dict[str, ZoneStats] = {}
    for z in ("A", "B"):
        risk = classify_risk(float(counts[z]) if risk_from == "count" else float(counts[z]))
        out[z] = ZoneStats(
            zone=z,
            count=int(counts[z]),
            risk_label=str(risk["label"]),
            risk_color=str(risk["color"]),
        )
    return out


class FlowCounter:
    """
    Counts entry/exit by line-crossing using tracked IDs.

    - Virtual line at x = frame_w/2
    - Left -> Right : entry (inflow)
    - Right -> Left : exit (outflow)

    Maintains rolling per-minute rates over a time window (seconds).
    """

    def __init__(self, window_sec: float = 60.0, cooldown_sec: float = 1.0):
        self.window_sec = float(window_sec)
        self.cooldown_sec = float(cooldown_sec)

        self._last_side: Dict[int, str] = {}
        self._last_event_t: Dict[Tuple[int, str], float] = {}
        self._entry_events: Deque[float] = deque()
        self._exit_events: Deque[float] = deque()

    def reset(self):
        self._last_side = {}
        self._last_event_t = {}
        self._entry_events = deque()
        self._exit_events = deque()

    def update(self, tracks: Iterable[Any], timestamp_sec: float, frame_w: int) -> Tuple[int, int]:
        """
        Returns (entries, exits) counted *for this update*.
        """
        ts = float(timestamp_sec)
        mid = float(frame_w) / 2.0

        entries = 0
        exits = 0

        active_ids = set()
        for tr in tracks:
            tid = int(getattr(tr, "track_id"))
            cx = float(getattr(tr, "center_xy")[0])
            side = "L" if cx < mid else "R"
            active_ids.add(tid)

            prev = self._last_side.get(tid)
            self._last_side[tid] = side
            if prev is None or prev == side:
                continue

            # Debounce repeated flip-flops.
            key_entry = (tid, "entry")
            key_exit = (tid, "exit")

            if prev == "L" and side == "R":
                last_t = self._last_event_t.get(key_entry, -1e9)
                if ts - last_t >= self.cooldown_sec:
                    entries += 1
                    self._entry_events.append(ts)
                    self._last_event_t[key_entry] = ts
            elif prev == "R" and side == "L":
                last_t = self._last_event_t.get(key_exit, -1e9)
                if ts - last_t >= self.cooldown_sec:
                    exits += 1
                    self._exit_events.append(ts)
                    self._last_event_t[key_exit] = ts

        # Prune stale sides for disappeared tracks (keeps dict small).
        for tid in list(self._last_side.keys()):
            if tid not in active_ids:
                # keep for a short while could be helpful, but simplicity wins here
                del self._last_side[tid]

        # Prune old events for rolling window.
        self._prune(ts)
        return entries, exits

    def _prune(self, now_sec: float):
        cutoff = float(now_sec) - self.window_sec
        while self._entry_events and self._entry_events[0] < cutoff:
            self._entry_events.popleft()
        while self._exit_events and self._exit_events[0] < cutoff:
            self._exit_events.popleft()

    def rates_per_min(self, now_sec: float) -> Tuple[float, float]:
        """
        Returns (inflow_per_min, outflow_per_min) over the rolling window.
        """
        self._prune(float(now_sec))
        window_min = max(1e-6, self.window_sec / 60.0)
        inflow = float(len(self._entry_events)) / window_min
        outflow = float(len(self._exit_events)) / window_min
        return inflow, outflow

