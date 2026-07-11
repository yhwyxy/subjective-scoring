"""SubjectiveScoringService 端到端单元测试。"""

from __future__ import annotations

from subjective_scoring import (
    IntermediateScoreResult,
    ReviewLevel,
    ScoringMode,
    ScoringRequest,
    ScoringResult,
    ScoringServiceResult,
    SubjectiveScoringService,
    create_default_service,
)


class _FixedScorer:
    def __init__(self, name: str, mode: ScoringMode, score: float, confidence: float, **kwargs):
        self.name = name
        self.mode = mode
        self._score = score
        self._confidence = confidence
        self.kwargs = kwargs
        self.calls: list[ScoringRequest] = []

    def score(self, request: ScoringRequest) -> IntermediateScoreResult:
        self.calls.append(request)
        capped = min(self._score, request.max_score)
        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=self.mode,
            score=capped,
            max_score=request.max_score,
            confidence=self._confidence,
            matched_evidence=[
                {
                    "point_id": "p1",
                    "score": capped,
                    "max_score": request.max_score,
                    "reason": "fixed",
                    "similarity": self._confidence,
                }
            ],
            warnings=list(self.kwargs.get("warnings") or []),
            force_manual_review=bool(self.kwargs.get("force_manual_review", False)),
            metadata={},
        )


def _service(**engine_scores) -> SubjectiveScoringService:
    text = _FixedScorer(
        "TextRerankerScorer",
        ScoringMode.TEXT,
        engine_scores.get("text_score", 8.0),
        engine_scores.get("text_conf", 0.9),
    )
    sql = _FixedScorer(
        "SQLStructureScorer",
        ScoringMode.SQL,
        engine_scores.get("sql_score", 10.0),
        engine_scores.get("sql_conf", 0.95),
    )
    code = _FixedScorer(
        "CodeHybridScorer",
        ScoringMode.CODE,
        engine_scores.get("code_score", 7.0),
        engine_scores.get("code_conf", 0.7),
        force_manual_review=engine_scores.get("code_force", False),
    )
    svc = SubjectiveScoringService(
        scorers={
            ScoringMode.TEXT: text,
            ScoringMode.SQL: sql,
            ScoringMode.CODE: code,
        },
        allow_model_load=False,
    )
    svc._test_text = text  # type: ignore[attr-defined]
    svc._test_sql = sql  # type: ignore[attr-defined]
    svc._test_code = code  # type: ignore[attr-defined]
    return svc


def test_text_pipeline_end_to_end():
    svc = _service()
    result = svc.score(
        {
            "question_id": "q001",
            "max_score": 10,
            "scoring_mode": "text",
            "student_answer": "索引可以让数据库查得更快。",
            "reference_answer": "索引可以提高查询效率。",
            "scoring_points": [
                {"id": "p1", "text": "提高查询效率", "score": 10},
            ],
        }
    )
    assert isinstance(result, ScoringResult)
    assert result.question_id == "q001"
    assert result.scoring_mode is ScoringMode.TEXT
    assert result.track == "TextRerankerScorer"
    assert result.score == 8.0
    assert result.confidence == 0.9
    assert result.need_manual_review is False
    assert result.review_level is ReviewLevel.AUTO_PASS
    assert result.matched_points[0].point_id == "p1"
    # normalizer should have been applied (punctuation etc.) before engine
    assert len(svc._test_text.calls) == 1  # type: ignore[attr-defined]


def test_routes_to_sql_by_mode():
    svc = _service()
    result = svc.score(
        ScoringRequest(
            question_id="s1",
            max_score=10,
            scoring_mode=ScoringMode.SQL,
            student_answer="select name from student",
            reference_answer="SELECT name FROM student",
        )
    )
    assert result.track == "SQLStructureScorer"
    assert result.scoring_mode is ScoringMode.SQL
    assert result.score == 10.0
    assert len(svc._test_sql.calls) == 1  # type: ignore[attr-defined]


def test_routes_by_code_language_when_mode_missing():
    svc = _service()
    result = svc.score(
        {
            "question_id": "c1",
            "max_score": 10,
            "code_language": "python",
            "student_answer": "def f():\n    return 1\n",
            "reference_answer": "def f():\n    return 1\n",
        }
    )
    assert result.scoring_mode is ScoringMode.CODE
    assert result.track == "CodeHybridScorer"


def test_force_manual_review_propagates():
    svc = _service(code_force=True, code_conf=0.95, code_score=9.0)
    result = svc.score(
        {
            "question_id": "c1",
            "max_score": 10,
            "scoring_mode": "code",
            "code_language": "python",
            "student_answer": "x=1",
            "reference_answer": "x=1",
        }
    )
    assert result.need_manual_review is True
    assert result.review_level is ReviewLevel.MANUAL_REQUIRED


def test_score_with_trace_ok():
    svc = _service(text_score=4.0, text_conf=0.88)
    wrapped = svc.score_with_trace(
        {
            "question_id": "q1",
            "max_score": 5,
            "scoring_mode": "text",
            "student_answer": "hello",
        }
    )
    assert isinstance(wrapped, ScoringServiceResult)
    assert wrapped.result.score == 4.0
    assert wrapped.trace is not None
    assert wrapped.trace.route.mode is ScoringMode.TEXT
    assert wrapped.trace.intermediate.scorer == "TextRerankerScorer"
    assert wrapped.trace.normalized_request.scoring_mode is ScoringMode.TEXT
    # convenience proxy
    assert wrapped.confidence == 0.88


def test_low_confidence_suggested_review():
    svc = _service(text_score=5.0, text_conf=0.72)
    result = svc.score(
        {
            "question_id": "q1",
            "max_score": 10,
            "scoring_mode": "text",
            "student_answer": "x",
        }
    )
    assert result.review_level is ReviewLevel.SUGGESTED_REVIEW
    assert result.need_manual_review is True


def test_engine_exception_degrades_gracefully():
    class Boom:
        def score(self, request):
            raise RuntimeError("model OOM")

    svc = SubjectiveScoringService(
        scorers={
            ScoringMode.TEXT: Boom(),
            ScoringMode.SQL: _FixedScorer("SQLStructureScorer", ScoringMode.SQL, 1, 1),
            ScoringMode.CODE: _FixedScorer("CodeHybridScorer", ScoringMode.CODE, 1, 1),
        },
        allow_model_load=False,
    )
    result = svc.score(
        {
            "question_id": "q1",
            "max_score": 10,
            "scoring_mode": "text",
            "student_answer": "a",
        }
    )
    assert result.score == 0.0
    assert result.need_manual_review is True
    assert result.review_level is ReviewLevel.MANUAL_REQUIRED
    assert any("异常" in w for w in result.warnings)


def test_score_many():
    svc = _service(text_score=3.0, text_conf=0.9)
    results = svc.score_many(
        [
            {"question_id": "a", "max_score": 10, "scoring_mode": "text", "student_answer": "1"},
            {"question_id": "b", "max_score": 10, "scoring_mode": "text", "student_answer": "2"},
        ]
    )
    assert len(results) == 2
    assert all(isinstance(r, ScoringResult) for r in results)
    assert results[0].question_id == "a"


def test_create_default_service_factory():
    svc = create_default_service(allow_model_load=False)
    assert isinstance(svc, SubjectiveScoringService)


def test_callable_interface():
    svc = _service(text_score=2.0, text_conf=0.5)
    result = svc(
        {"question_id": "q", "max_score": 10, "scoring_mode": "text", "student_answer": "x"}
    )
    assert result.score == 2.0
    assert result.review_level is ReviewLevel.MANUAL_REQUIRED
