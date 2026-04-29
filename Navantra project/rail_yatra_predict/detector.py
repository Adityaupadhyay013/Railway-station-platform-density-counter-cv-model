from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os

import cv2
import numpy as np


@dataclass
class TrackedPerson:
    track_id: int
    box_xyxy: Tuple[int, int, int, int]
    confidence: float
    center_xy: Tuple[int, int]


@dataclass
class DetectionResult:
    boxes_xyxy: List[Tuple[int, int, int, int]]
    count: int
    annotated_frame_bgr: np.ndarray
    method: str  # e.g. "yolov8" or "synthetic-fallback"
    confidences: Optional[List[float]] = None
    tracks: Optional[List[TrackedPerson]] = None


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)

    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def _center_xy(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


class _LightTracker:
    """
    Lightweight IoU + centroid tracker (CPU-friendly, dependency-free).

    - Assigns stable integer IDs across frames
    - Uses greedy matching by IoU first, then distance gating
    - Drops tracks after a short "max_age" if not matched
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 20, max_distance_px: int = 120):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.max_distance_px = int(max_distance_px)

        self._next_id = 1
        self._tracks: Dict[int, Dict[str, object]] = {}

    def reset(self):
        self._next_id = 1
        self._tracks = {}

    def update(
        self, boxes_xyxy: List[Tuple[int, int, int, int]], confidences: List[float]
    ) -> List[TrackedPerson]:
        detections = [{"box": b, "conf": float(confidences[i])} for i, b in enumerate(boxes_xyxy)]
        det_centers = [_center_xy(d["box"]) for d in detections]

        # Age existing tracks.
        for tid in list(self._tracks.keys()):
            self._tracks[tid]["age"] = int(self._tracks[tid].get("age", 0)) + 1

        # Build candidate matches (track_id, det_idx, score).
        candidates: List[Tuple[int, int, float]] = []
        for tid, tr in self._tracks.items():
            tbox = tr["box"]  # type: ignore[assignment]
            tcx, tcy = tr["center"]  # type: ignore[assignment]
            for di, det in enumerate(detections):
                iou = _iou_xyxy(tbox, det["box"])  # type: ignore[arg-type]
                dcx, dcy = det_centers[di]
                dist = float(((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5)
                if iou >= self.iou_threshold or dist <= float(self.max_distance_px):
                    # Prefer IoU, then closer distance.
                    score = iou - (dist / (self.max_distance_px + 1e-6)) * 0.05
                    candidates.append((tid, di, score))

        # Greedy best-first assignment.
        candidates.sort(key=lambda x: x[2], reverse=True)
        matched_tracks: Dict[int, int] = {}
        matched_dets: Dict[int, int] = {}
        for tid, di, _ in candidates:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks[tid] = di
            matched_dets[di] = tid

        # Update matched tracks.
        for tid, di in matched_tracks.items():
            box = detections[di]["box"]  # type: ignore[assignment]
            conf = float(detections[di]["conf"])
            self._tracks[tid] = {
                "box": box,
                "conf": conf,
                "center": _center_xy(box),
                "age": 0,
            }

        # Create new tracks for unmatched detections.
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            tid = int(self._next_id)
            self._next_id += 1
            box = det["box"]  # type: ignore[assignment]
            self._tracks[tid] = {
                "box": box,
                "conf": float(det["conf"]),
                "center": _center_xy(box),
                "age": 0,
            }

        # Prune old tracks.
        for tid in list(self._tracks.keys()):
            if int(self._tracks[tid].get("age", 0)) > self.max_age:
                del self._tracks[tid]

        # Emit current tracked persons.
        out: List[TrackedPerson] = []
        for tid, tr in self._tracks.items():
            box = tr["box"]  # type: ignore[assignment]
            conf = float(tr["conf"])
            cx, cy = tr["center"]  # type: ignore[assignment]
            out.append(TrackedPerson(track_id=int(tid), box_xyxy=box, confidence=conf, center_xy=(int(cx), int(cy))))
        return out


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        img_size: int = 640,
        enable_fallback: bool = True,
        fallback_min_area_ratio: float = 0.0025,
    ):
        self.model_path = model_path
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.img_size = int(img_size)
        self.enable_fallback = enable_fallback
        self.fallback_min_area_ratio = float(fallback_min_area_ratio)

        # Heatmap overlay state (optional).
        self._heatmap: Optional[np.ndarray] = None
        self._heatmap_decay = 0.94

        # Lightweight tracker for stable IDs (CPU friendly).
        self._tracker = _LightTracker(iou_threshold=0.25, max_age=18, max_distance_px=140)

        # Import ultralytics lazily so the app can start even before model download.
        from ultralytics import YOLO  # type: ignore

        # Resolve model path if needed (ultralytics supports names like "yolov8n.pt").
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(model_path)

    def _update_heatmap(self, frame_bgr: np.ndarray, boxes_xyxy: List[Tuple[int, int, int, int]]):
        h, w = frame_bgr.shape[:2]
        if self._heatmap is None or self._heatmap.shape[:2] != (h, w):
            self._heatmap = np.zeros((h, w), dtype=np.float32)

        # Decay previous heat.
        self._heatmap *= self._heatmap_decay

        for (x1, y1, x2, y2) in boxes_xyxy:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            # Draw a small "hot" blob at the person center.
            cv2.circle(self._heatmap, (cx, cy), radius=10, color=1.0, thickness=-1)

        # Blur to create a smoother density look.
        heat = cv2.GaussianBlur(self._heatmap, (0, 0), sigmaX=15, sigmaY=15)
        heat_norm = heat / (heat.max() + 1e-6)
        heat_u8 = (heat_norm * 255).astype(np.uint8)
        return heat_u8

    def _annotate_frame(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: List[Tuple[int, int, int, int]],
        method: str,
        confidences: Optional[List[float]] = None,
        tracks: Optional[List[TrackedPerson]] = None,
        show_heatmap: bool = False,
    ):
        annotated = frame_bgr.copy()

        if show_heatmap and len(boxes_xyxy) > 0:
            heat_u8 = self._update_heatmap(frame_bgr, boxes_xyxy)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            annotated = cv2.addWeighted(annotated, 0.70, heat_color, 0.30, 0)

        if tracks is not None and len(tracks) > 0:
            for tr in tracks:
                x1, y1, x2, y2 = tr.box_xyxy
                label = f"ID: {tr.track_id}  conf: {tr.confidence:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        else:
            for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
                conf = confidences[i] if (confidences is not None and i < len(confidences)) else None
                label = f"person {conf:.2f}" if conf is not None else "person"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            annotated,
            f"det: {method}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def detect_people(self, frame_bgr: np.ndarray, show_heatmap: bool = False) -> DetectionResult:
        """
        Returns YOLO "person" detections + count, with optional heatmap overlay.

        If YOLO finds no persons and `enable_fallback=True`, uses a synthetic fallback
        detector (color-based) to keep the demo functional.
        """
        # YOLO person class id in COCO is 0.
        person_class_id = 0

        method = "yolov8"
        boxes_xyxy: List[Tuple[int, int, int, int]] = []
        confidences: List[float] = []

        try:
            # Ultralytics accepts BGR numpy frames; it will handle preprocessing.
            results = self.model.predict(
                source=frame_bgr,
                verbose=False,
                imgsz=self.img_size,
                conf=self.conf,
                iou=self.iou,
                classes=[person_class_id],
                device=self.device,
            )
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    for b in boxes:
                        xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                        boxes_xyxy.append((x1, y1, x2, y2))
                        confidences.append(float(b.conf[0]))
        except Exception:
            # If YOLO errors (e.g., download issues), we can still show a working demo.
            boxes_xyxy = []
            confidences = []

        # Fallback for the synthetic demo video.
        if self.enable_fallback and len(boxes_xyxy) == 0:
            method = "synthetic-fallback"
            # Import here to avoid circular imports.
            from utils import synthetic_person_boxes

            # boxes from color thresholding
            boxes_xyxy = synthetic_person_boxes(frame_bgr)
            # Provide "fake" confidences for consistent annotation.
            confidences = [0.55 for _ in boxes_xyxy]

        # Update tracker (even for fallback boxes).
        tracks = self._tracker.update(boxes_xyxy=boxes_xyxy, confidences=confidences) if len(boxes_xyxy) > 0 else []

        count = len(boxes_xyxy)
        annotated = self._annotate_frame(
            frame_bgr=frame_bgr,
            boxes_xyxy=boxes_xyxy,
            confidences=confidences,
            tracks=tracks,
            method=method,
            show_heatmap=show_heatmap,
        )
        return DetectionResult(
            boxes_xyxy=boxes_xyxy,
            count=count,
            annotated_frame_bgr=annotated,
            method=method,
            confidences=confidences,
            tracks=tracks,
        )

