"""QuestionTypeRouter 单元测试。"""

from __future__ import annotations

from subjective_scoring import (
    IntermediateScoreResult,
    QuestionTypeRouter,
    ScoringMode,
    ScoringRequest,
)


class _FakeScorer:
    def __init__(self, name: str, mode: ScoringMode):
        self.name = name
        self.mode = mode

    def score(self, request: ScoringRequest) -> IntermediateScoreResult:
        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=self.mode,
            score=1.0,
            max_score=request.max_score,
            confidence=0.9,
            metadata={},
        )


def _router() -> QuestionTypeRouter:
    return QuestionTypeRouter(
        scorers={
            ScoringMode.TEXT: _FakeScorer("TextRerankerScorer", ScoringMode.TEXT),
            ScoringMode.SQL: _FakeScorer("SQLStructureScorer", ScoringMode.SQL),
            ScoringMode.CODE: _FakeScorer("CodeHybridScorer", ScoringMode.CODE),
        }
    )


def test_explicit_scoring_mode_wins():
    r = _router()
    d = r.resolve(
        ScoringRequest(
            question_id="q",
            max_score=10,
            scoring_mode=ScoringMode.SQL,
            code_language="python",  # 应被忽略
            course_type="programming",
        )
    )
    assert d.mode is ScoringMode.SQL
    assert d.scorer_name == "SQLStructureScorer"
    assert "scoring_mode" in d.reason or d.reason.startswith("显式")


def test_question_type_code():
    r = _router()
    d = r.resolve(
        ScoringRequest(
            question_id="q",
            max_score=10,
            question_type="programming",
        )
    )
    assert d.mode is ScoringMode.CODE


def test_code_language_sql():
    r = _router()
    d = r.resolve(
        ScoringRequest(
            question_id="q",
            max_score=10,
            code_language="SQL",
        )
    )
    assert d.mode is ScoringMode.SQL
    assert d.reason == "code_language"


def test_code_language_python():
    r = _router()
    d = r.resolve(
        ScoringRequest(
            question_id="q",
            max_score=10,
            code_language="python",
        )
    )
    assert d.mode is ScoringMode.CODE


def test_course_type_database():
    r = _router()
    d = r.resolve(
        ScoringRequest(
            question_id="q",
            max_score=10,
            course_type="database",
            question_type="subjective",  # 泛化 text 类型不压过 course_type
        )
    )
    assert d.mode is ScoringMode.SQL
    assert d.reason == "course_type"


def test_course_type_when_question_type_unknown():
    r = _router()
    d = r.resolve(
        ScoringRequest(
            question_id="q",
            max_score=10,
            question_type="unknown_custom",
            course_type="database",
        )
    )
    assert d.mode is ScoringMode.SQL
    assert d.reason == "course_type"


def test_reference_answer_sql_fallback():
    r = _router()
    d = r.resolve(
        ScoringRequest(
            question_id="q",
            max_score=10,
            question_type="custom",
            reference_answer="SELECT name FROM student WHERE id = 1",
        )
    )
    assert d.mode is ScoringMode.SQL


def test_default_text():
    r = _router()
    d = r.resolve(
        ScoringRequest(
            question_id="q",
            max_score=10,
            student_answer="索引可以提高查询效率",
        )
    )
    assert d.mode is ScoringMode.TEXT
    assert d.scorer_name == "TextRerankerScorer"


def test_route_invokes_registered_scorer():
    r = _router()
    result = r.route(
        ScoringRequest(
            question_id="q",
            max_score=10,
            scoring_mode=ScoringMode.CODE,
            code_language="python",
        )
    )
    assert result.scorer == "CodeHybridScorer"
    assert result.metadata["route_mode"] == "code"
    assert "route_reason" in result.metadata


def test_select_scorer_missing_raises():
    r = QuestionTypeRouter(scorers={})
    try:
        r.select_scorer(
            ScoringRequest(question_id="q", max_score=1, scoring_mode=ScoringMode.TEXT)
        )
        assert False, "expected KeyError"
    except KeyError as e:
        assert "TextRerankerScorer" in str(e)
