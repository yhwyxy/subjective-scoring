from subjective_scoring import CalculationScorer, ScoringMode, ScoringRequest, SubjectiveScoringService


def _request(answer: str) -> ScoringRequest:
    return ScoringRequest.model_validate(
        {
            "question_id": "calc-1",
            "max_score": 20,
            "scoring_mode": "calculation",
            "student_answer": answer,
            "scoring_config": {
                "calculation": {
                    "require_working": True,
                    "final_only_score_cap": 8,
                    "steps": [
                        {"id": "mn_delta", "description": "锰目标增量", "expected": 0.45, "tolerance": 0.005, "unit": "%", "score": 4, "keywords": ["增量"]},
                        {"id": "simn_amount", "description": "硅锰合金用量", "expected": 752.5, "tolerance": 1, "unit": "kg", "score": 8, "keywords": ["硅锰"]},
                    ],
                    "final_answers": [
                        {"id": "final_simn", "description": "最终硅锰合金答案", "expected": 753, "tolerance": 1, "unit": "kg", "score": 4},
                        {"id": "final_sife", "description": "最终硅铁答案", "expected": 135, "tolerance": 1, "unit": "kg", "score": 4},
                    ],
                }
            },
        }
    )


def test_calculation_steps_and_final_answers_use_tolerance_and_units():
    result = CalculationScorer().score(
        _request("锰目标增量 = 45%\n硅锰合金 = 753 kg\n最终：753 kg，135 kg")
    )

    assert result.scoring_mode is ScoringMode.CALCULATION
    assert result.score == 20.0
    assert result.force_manual_review is False
    assert result.metadata["decision_reason"] == "all_configured_values_matched"


def test_calculation_without_working_is_capped_at_configured_final_score():
    result = CalculationScorer().score(_request("753 kg, 135 kg"))

    assert result.score == 8.0
    assert result.force_manual_review is False
    assert any("计算过程" in warning for warning in result.warnings)


def test_service_routes_calculation_and_preserves_units():
    result = SubjectiveScoringService(allow_model_load=False).score(
        {
            "question_id": "calc-service",
            "max_score": 2,
            "scoring_mode": "calculation",
            "student_answer": "结果 = 10 kg",
            "scoring_config": {
                "calculation": {
                    "final_answers": [
                        {"id": "final", "description": "结果", "expected": 10, "unit": "kg", "score": 2}
                    ]
                }
            },
        }
    )
    assert result.scoring_mode is ScoringMode.CALCULATION
    assert result.track == "CalculationScorer"
    assert result.score == 2.0
