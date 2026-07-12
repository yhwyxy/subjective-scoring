"""可解释的相似度分数校准组件。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class ScoreCalibrator(Protocol):
    """把后端相关度映射为可用于评分的 0..1 覆盖度。"""

    def calibrate(self, score: float) -> float: ...


@dataclass(frozen=True)
class PiecewiseLinearCalibrator:
    """在单调控制点之间做线性插值。"""

    points: tuple[tuple[float, float], ...]
    name: str = "piecewise_linear"

    def __init__(
        self,
        points: Sequence[tuple[float, float]],
        *,
        name: str = "piecewise_linear",
    ) -> None:
        normalized = tuple((float(x), float(y)) for x, y in points)
        if len(normalized) < 2:
            raise ValueError("calibration points 至少需要两个控制点")
        previous_x = previous_y = -1.0
        for x, y in normalized:
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError("calibration points 必须位于 0..1")
            if x <= previous_x or y < previous_y:
                raise ValueError("calibration points 必须按 x 严格递增且 y 单调不减")
            previous_x, previous_y = x, y
        object.__setattr__(self, "points", normalized)
        object.__setattr__(self, "name", name)

    def calibrate(self, score: float) -> float:
        value = max(0.0, min(1.0, float(score)))
        if value <= self.points[0][0]:
            return self.points[0][1]
        for (x0, y0), (x1, y1) in zip(self.points, self.points[1:]):
            if value <= x1:
                ratio = (value - x0) / (x1 - x0)
                return max(0.0, min(1.0, y0 + ratio * (y1 - y0)))
        return self.points[-1][1]


IDENTITY_CALIBRATOR = PiecewiseLinearCalibrator(
    ((0.0, 0.0), (1.0, 1.0)),
    name="identity",
)
LOCAL_CROSS_ENCODER_CALIBRATOR = PiecewiseLinearCalibrator(
    ((0.0, 0.0), (0.20, 0.20), (0.50, 0.65), (0.75, 0.90), (1.0, 1.0)),
    name="local_cross_encoder_v1",
)
REMOTE_RERANKER_CALIBRATOR = PiecewiseLinearCalibrator(
    ((0.0, 0.0), (0.02, 0.10), (0.08, 0.55), (0.20, 0.80), (0.45, 1.0), (1.0, 1.0)),
    name="remote_reranker_v1",
)


def default_calibrator_for_backend(backend_name: str) -> ScoreCalibrator:
    """按解析后的后端名称选择默认校准曲线。"""

    if backend_name.startswith("cohere:"):
        return REMOTE_RERANKER_CALIBRATOR
    if backend_name in {"injected", "lexical_fallback"}:
        return IDENTITY_CALIBRATOR
    return LOCAL_CROSS_ENCODER_CALIBRATOR


__all__ = [
    "IDENTITY_CALIBRATOR",
    "LOCAL_CROSS_ENCODER_CALIBRATOR",
    "REMOTE_RERANKER_CALIBRATOR",
    "PiecewiseLinearCalibrator",
    "ScoreCalibrator",
    "default_calibrator_for_backend",
]
