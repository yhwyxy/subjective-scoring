from __future__ import annotations

import pytest

from subjective_scoring import PiecewiseLinearCalibrator
from subjective_scoring.engines.calibration import (
    calibrator_for_backend,
    default_calibrator_for_backend,
)


def test_piecewise_calibrator_interpolates_monotonically():
    calibrator = PiecewiseLinearCalibrator(((0.0, 0.0), (0.5, 0.8), (1.0, 1.0)))
    assert calibrator.calibrate(0.25) == pytest.approx(0.4)
    assert calibrator.calibrate(0.75) == pytest.approx(0.9)


def test_calibrator_rejects_non_monotonic_points():
    with pytest.raises(ValueError):
        PiecewiseLinearCalibrator(((0.0, 0.0), (0.5, 0.8), (0.7, 0.6)))


def test_remote_and_injected_backends_use_distinct_profiles():
    remote = default_calibrator_for_backend("cohere:model")
    injected = default_calibrator_for_backend("injected")
    assert remote.calibrate(0.2) != injected.calibrate(0.2)


def test_remote_profile_uses_exam_system_validated_curve():
    remote = default_calibrator_for_backend("cohere:model")

    assert remote.points == ((0.0, 0.0), (0.9, 0.5), (1.0, 0.85))
    assert remote.calibrate(0.9) == pytest.approx(0.5)
    assert remote.calibrate(1.0) == pytest.approx(0.85)


def test_request_points_override_backend_profile_without_mutating_default():
    original = default_calibrator_for_backend("cohere:model")
    request_calibrator = calibrator_for_backend(
        "cohere:model",
        ((0.0, 0.0), (0.5, 0.2), (1.0, 1.0)),
    )

    assert request_calibrator.calibrate(0.5) == pytest.approx(0.2)
    assert default_calibrator_for_backend("cohere:model") is original
