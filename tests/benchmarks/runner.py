from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from subjective_scoring import ScoringRequest


ROOT = Path(__file__).parent
QUALITY_RANK = {"wrong": 0, "partial": 1, "paraphrase": 2, "complete": 3}


def load_cases(mode: str | None = None) -> list[dict[str, Any]]:
    paths = [ROOT / f"{mode}_cases.json"] if mode else sorted(ROOT.glob("*_cases.json"))
    cases: list[dict[str, Any]] = []
    for path in paths:
        cases.extend(json.loads(path.read_text(encoding="utf-8")))
    return cases


def run_cases(
    cases: list[dict[str, Any]],
    scorer_factory: Callable[[dict[str, Any]], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        request = ScoringRequest.model_validate(case["request"])
        result = scorer_factory(case).score(request)
        expected_min, expected_max = case["expected_band"]
        midpoint = (expected_min + expected_max) / 2
        records.append(
            {
                "id": case["id"],
                "mode": case["mode"],
                "quality": case["quality"],
                "ordering_group": case.get("ordering_group"),
                "expected_min": expected_min,
                "expected_max": expected_max,
                "actual_score": result.score,
                "absolute_error": abs(result.score - midpoint),
                "within_band": expected_min <= result.score <= expected_max,
            }
        )
    return records, calculate_metrics(records)


def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"mae": 0.0, "band_hit_rate": 0.0, "ordering_accuracy": 0.0}
    mae = sum(float(record["absolute_error"]) for record in records) / len(records)
    band_hit_rate = sum(bool(record["within_band"]) for record in records) / len(records)
    comparisons = correct = 0
    groups = {record.get("ordering_group") for record in records if record.get("ordering_group")}
    for group in groups:
        grouped = [record for record in records if record.get("ordering_group") == group]
        for left, right in combinations(grouped, 2):
            expected_order = QUALITY_RANK[left["quality"]] - QUALITY_RANK[right["quality"]]
            if expected_order == 0:
                continue
            comparisons += 1
            actual_order = float(left["actual_score"]) - float(right["actual_score"])
            if actual_order * expected_order > 0:
                correct += 1
    return {
        "mae": round(mae, 4),
        "band_hit_rate": round(band_hit_rate, 4),
        "ordering_accuracy": round(correct / comparisons, 4) if comparisons else 1.0,
        "record_count": len(records),
    }
