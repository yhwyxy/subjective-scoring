from __future__ import annotations

import pytest

from subjective_scoring import PiecewiseLinearCalibrator
from subjective_scoring.engines.calibration import default_calibrator_for_backend


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
    assert remote.calibrate(0.2) > injected.calibrate(0.2)
