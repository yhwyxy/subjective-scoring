from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    expected_total: float
    actual_total: float
    net_error: float
    net_error_rate: float
    absolute_error: float
    absolute_error_rate: float
    mean_absolute_error: float
    large_over_score_count: int
    large_under_score_count: int


def run_cases(
    cases: Iterable[Mapping[str, Any]],
    score_fn: Callable[[Mapping[str, Any]], float],
) -> BenchmarkSummary:
    expected_scores: list[float] = []
    actual_scores: list[float] = []
    max_scores: list[float] = []
    absolute_errors: list[float] = []
    large_over_score_count = 0
    large_under_score_count = 0

    for case in cases:
        expected = float(case["expected_score"])
        actual = float(score_fn(case))
        max_score = float(case["max_score"])
        delta = actual - expected
        severe_error_threshold = max(2.0, max_score * 0.4)

        expected_scores.append(expected)
        actual_scores.append(actual)
        max_scores.append(max_score)
        absolute_errors.append(abs(delta))
        if delta >= severe_error_threshold:
            large_over_score_count += 1
        elif -delta >= severe_error_threshold:
            large_under_score_count += 1

    expected_total = fsum(expected_scores)
    actual_total = fsum(actual_scores)
    max_score_total = fsum(max_scores)
    absolute_error = fsum(absolute_errors)
    case_count = len(expected_scores)
    net_error = actual_total - expected_total

    return BenchmarkSummary(
        expected_total=expected_total,
        actual_total=actual_total,
        net_error=net_error,
        net_error_rate=net_error / expected_total if expected_total else 0.0,
        absolute_error=absolute_error,
        absolute_error_rate=(
            absolute_error / max_score_total if max_score_total else 0.0
        ),
        mean_absolute_error=absolute_error / case_count if case_count else 0.0,
        large_over_score_count=large_over_score_count,
        large_under_score_count=large_under_score_count,
    )
