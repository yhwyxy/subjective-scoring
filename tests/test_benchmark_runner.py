from __future__ import annotations

from subjective_scoring.engines import TextRerankerScorer
from tests.benchmarks.runner import calculate_metrics, load_cases, run_cases


def test_text_benchmark_uses_shared_case_files_and_meets_ordering_target():
    cases = load_cases("text")

    def scorer_factory(case):
        scores = case["pair_scores"]
        point_ids = {point["text"]: point["id"] for point in case["request"]["scoring_points"]}
        return TextRerankerScorer(
            pair_scorer=lambda student, point: scores[point_ids[point]],
            allow_model_load=False,
        )

    records, metrics = run_cases(cases, scorer_factory)
    assert len(records) == 4
    assert metrics["band_hit_rate"] >= 0.75
    assert metrics["ordering_accuracy"] >= 0.95
    assert metrics["mae"] <= 1.5


def test_metrics_are_reported_per_expected_contract():
    metrics = calculate_metrics(
        [
            {"quality": "complete", "ordering_group": "g", "actual_score": 9, "absolute_error": 0.5, "within_band": True},
            {"quality": "wrong", "ordering_group": "g", "actual_score": 1, "absolute_error": 0.5, "within_band": True},
        ]
    )
    assert metrics == {
        "mae": 0.5,
        "band_hit_rate": 1.0,
        "ordering_accuracy": 1.0,
        "record_count": 2,
    }
