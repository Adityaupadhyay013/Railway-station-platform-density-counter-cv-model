from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque

import numpy as np


@dataclass
class CrowdPrediction:
    predicted_count: float
    inflow_per_min: float = 0.0
    outflow_per_min: float = 0.0
    trend_per_min: float = 0.0
    net_flow_per_min: float = 0.0


class CrowdPredictor:
    """
    Simple predictive model:
    - Keep a history of (timestamp_sec, crowd_count)
    - Use moving average as a stabilizer
    - Use linear trend (polyfit) to extrapolate to a fixed horizon (30 min)
    """

    def __init__(
        self,
        history_points: int = 300,
        ma_window: int = 10,
        trend_window: int = 30,
        horizon_minutes: float = 30.0,
        clamp_min: float = 0.0,
        clamp_max: float = 200.0,
        flow_gain: float = 0.8,
        flow_horizon_cap_min: float = 15.0,
    ):
        self.history: Deque[Tuple[float, float]] = deque(maxlen=history_points)
        self.ma_window = int(ma_window)
        self.trend_window = int(trend_window)
        self.horizon_minutes = float(horizon_minutes)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # How strongly net flow (in-out) influences prediction.
        self.flow_gain = float(flow_gain)
        # Prevent long horizons from over-amplifying flow-based deltas.
        self.flow_horizon_cap_min = float(flow_horizon_cap_min)

    def update_and_predict(
        self,
        crowd_count: float,
        timestamp_sec: float,
        inflow_per_min: float = 0.0,
        outflow_per_min: float = 0.0,
    ) -> CrowdPrediction:
        self.history.append((float(timestamp_sec), float(crowd_count)))
        predicted, trend_per_min, net_flow = self.predict_next(
            inflow_per_min=float(inflow_per_min),
            outflow_per_min=float(outflow_per_min),
        )
        return CrowdPrediction(
            predicted_count=float(predicted),
            inflow_per_min=float(inflow_per_min),
            outflow_per_min=float(outflow_per_min),
            trend_per_min=float(trend_per_min),
            net_flow_per_min=float(net_flow),
        )

    def _moving_average(self, counts: List[float]) -> float:
        if not counts:
            return 0.0
        w = max(1, min(len(counts), self.ma_window))
        arr = np.array(counts[-w:], dtype=np.float32)
        return float(arr.mean())

    def predict_next(self, inflow_per_min: float = 0.0, outflow_per_min: float = 0.0) -> Tuple[float, float, float]:
        if len(self.history) == 0:
            return 0.0, 0.0, float(inflow_per_min - outflow_per_min)

        times = np.array([t for (t, _) in self.history], dtype=np.float64)
        counts = np.array([c for (_, c) in self.history], dtype=np.float64)

        ma = self._moving_average(list(counts))

        # If insufficient history for a meaningful trend, return moving average.
        if len(self.history) < max(4, self.trend_window // 2):
            base = float(np.clip(ma, self.clamp_min, self.clamp_max))
            net_flow = float(inflow_per_min - outflow_per_min)
            flow_minutes = float(min(self.horizon_minutes, self.flow_horizon_cap_min))
            flow_delta = self.flow_gain * net_flow * flow_minutes
            out = float(np.clip(base + flow_delta, self.clamp_min, self.clamp_max))
            return out, 0.0, net_flow

        # Trend window.
        w = max(4, min(len(counts), self.trend_window))
        times_w = times[-w:]
        counts_w = counts[-w:]

        # Convert to minutes relative to the start of the window to reduce numeric magnitude.
        t0 = times_w[0]
        t_minutes = (times_w - t0) / 60.0

        # Linear regression: count ~= slope * t + intercept
        try:
            slope, intercept = np.polyfit(t_minutes, counts_w, deg=1)
            future_t = t_minutes[-1] + self.horizon_minutes
            pred = slope * future_t + intercept
        except Exception:
            pred = ma

        # Blend with moving average to avoid wild extrapolation.
        if len(self.history) < self.trend_window:
            blend = 0.7
        else:
            blend = 0.5

        blended = blend * ma + (1.0 - blend) * float(pred)

        # Flow adjustment (production-like heuristic):
        # expected_change ~= (inflow - outflow) * horizon_minutes
        net_flow = float(inflow_per_min - outflow_per_min)
        flow_minutes = float(min(self.horizon_minutes, self.flow_horizon_cap_min))
        flow_delta = self.flow_gain * net_flow * flow_minutes

        out = float(np.clip(blended + flow_delta, self.clamp_min, self.clamp_max))
        # slope is in "count per minute" because t_minutes is in minutes.
        trend_per_min = float(slope) if "slope" in locals() else 0.0
        return out, trend_per_min, net_flow

    def history_as_dataframe(self):
        """
        Returns a minimal structure suitable for Streamlit line_chart.
        """
        # Import locally to keep predictor.py lightweight.
        import pandas as pd

        if len(self.history) == 0:
            return pd.DataFrame({"time_min": [], "count": []})

        t0 = self.history[0][0]
        rows = [{"time_min": (t - t0) / 60.0, "count": c} for (t, c) in self.history]
        return pd.DataFrame(rows)

