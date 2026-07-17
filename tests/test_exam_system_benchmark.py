from __future__ import annotations

import json
from pathlib import Path

from tests.benchmarks.exam_system_runner import run_cases


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "exam_system_subjective_cases.json"


def test_exam_system_fixture_matches_reference_corpus() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["source"] == "examSystem-18-papers-subjective-only"
    assert len(payload["cases"]) == 61
    assert sum(case["expected_score"] for case in payload["cases"]) == 307

    serialized = json.dumps(payload, ensure_ascii=False).lower()
    assert "http://" not in serialized
    assert "https://" not in serialized
    assert "authorization" not in serialized
    assert "api_key" not in serialized


def test_run_cases_does_not_hide_offsetting_errors() -> None:
    cases = [
        {"question_id": "q1", "expected_score": 0.0, "max_score": 10.0},
        {"question_id": "q2", "expected_score": 10.0, "max_score": 10.0},
    ]
    actual = {"q1": 10.0, "q2": 0.0}

    summary = run_cases(cases, lambda case: actual[case["question_id"]])

    assert summary.net_error == 0.0
    assert summary.absolute_error == 20.0
    assert summary.mean_absolute_error == 10.0
    assert summary.absolute_error_rate == 1.0
    assert summary.large_over_score_count == 1
    assert summary.large_under_score_count == 1
