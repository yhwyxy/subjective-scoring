from subjective_scoring import CodeHybridScorer, ScoringRequest


def _request(answer: str, profile: str, language: str = "python") -> ScoringRequest:
    return ScoringRequest.model_validate(
        {
            "question_id": "code-static",
            "max_score": 10 if profile == "nested_loop_static" else 15,
            "scoring_mode": "code",
            "code_language": language,
            "code_scoring_profile": profile,
            "student_answer": answer,
        }
    )


def test_nested_loop_profile_requires_actual_nesting():
    result = CodeHybridScorer(allow_model_load=False).score(
        _request(
            "def f(n):\n    for i in range(n):\n        for j in range(n):\n            pass\n",
            "nested_loop_static",
        )
    )
    assert result.score == 10.0
    assert result.force_manual_review is False
    assert result.metadata["checks"]["nested_loop"] is True

    sequential = CodeHybridScorer(allow_model_load=False).score(
        _request(
            "def f(n):\n    for i in range(n):\n        pass\n    for j in range(n):\n        pass\n",
            "nested_loop_static",
        )
    )
    assert sequential.score < 10.0
    assert sequential.metadata["checks"]["nested_loop"] is False


def test_find_index_profile_accepts_python_manual_loop_without_execution():
    result = CodeHybridScorer(allow_model_load=False).score(
        _request(
            "def find_index(array, item):\n"
            "    for index, element in enumerate(array):\n"
            "        if element == item:\n"
            "            return index\n",
            "find_index_static",
        )
    )
    assert result.score == 15.0
    assert result.force_manual_review is False
    assert result.metadata["assessment_type"] == "static_profile"
