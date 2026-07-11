"""ScoreAggregatorComponent 单元测试。"""

from __future__ import annotations

from typing import Any

import pytest

from subjective_scoring import (
    EvidenceItem,
    IntermediateScoreResult,
    ManualReviewThresholds,
    ReviewLevel,
    ScoringMode,
    ScoringOptions,
    ScoringRequest,
)
from subjective_scoring.components import ScoreAggregatorComponent


@pytest.fixture
def scoring_request() -> ScoringRequest:
    return ScoringRequest(
        question_id="q1",
        max_score=10,
        scoring_mode=ScoringMode.TEXT,
        scoring_config=ScoringOptions(
            manual_review_thresholds=ManualReviewThresholds(
                auto_pass=0.85,
                review=0.6,
            ),
            score_precision=1,
        ),
    )


def intermediate(
    *,
    scorer: str,
    score: float,
    confidence: float,
    max_score: float = 10,
    scoring_mode: ScoringMode = ScoringMode.TEXT,
    matched: list[dict[str, Any]] | None = None,
    missed: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    force_manual_review: bool = False,
) -> IntermediateScoreResult:
    return IntermediateScoreResult(
        scorer=scorer,
        scoring_mode=scoring_mode,
        score=score,
        max_score=max_score,
        confidence=confidence,
        matched_evidence=[EvidenceItem.model_validate(item) for item in matched or []],
        missed_evidence=[EvidenceItem.model_validate(item) for item in missed or []],
        warnings=warnings or [],
        force_manual_review=force_manual_review,
    )


def test_aggregates_score_confidence_evidence_and_review(scoring_request):
    result = ScoreAggregatorComponent().aggregate(
        scoring_request,
        intermediate(
            scorer="TextRerankerScorer",
            score=8,
            confidence=0.72,
            matched=[
                {
                    "point_id": "p1",
                    "score": 4,
                    "max_score": 5,
                    "evidence": "查询更快",
                    "reason": "命中效率评分点",
                    "similarity": 0.9,
                }
            ],
            missed=[
                {
                    "point_id": "p2",
                    "max_score": 5,
                    "reason": "未提到减少全表扫描",
                }
            ],
            warnings=["答案较短"],
        ),
    )

    assert result.question_id == "q1"
    assert result.score == 8
    assert result.confidence == 0.72
    assert result.review_level is ReviewLevel.SUGGESTED_REVIEW
    assert result.need_manual_review is True
    assert result.track == "TextRerankerScorer"
    assert result.matched_points[0].evidence == "查询更快"
    assert result.missed_points[0].point_id == "p2"
    assert result.warnings == ["答案较短"]


def test_aggregates_partial_results_and_caps_score(scoring_request):
    """分项 max_score 之和 <= 题目满分时允许相加，并在超满分时封顶。

    分项各自 score <= 自身 max_score；合计 score 仍可能因浮点/边界略超题目满分，
    此处构造 6+4 满分分项但得分 6+4.5 不可能（受中间校验）。改为分项合计 max=10，
    得分合计 6+4=10 不封顶；另起用例验证封顶。
    """
    result = ScoreAggregatorComponent()(
        scoring_request,
        [
            intermediate(
                scorer="PartialA",
                score=6,
                confidence=0.9,
                max_score=6,
            ),
            intermediate(
                scorer="PartialB",
                score=4,
                confidence=0.8,
                max_score=4,
                warnings=["规则冲突"],
            ),
        ],
    )

    assert result.score == 10
    assert result.confidence == pytest.approx(0.85)
    assert result.review_level is ReviewLevel.AUTO_PASS
    assert result.need_manual_review is False
    assert result.track == "PartialA + PartialB"
    assert result.warnings == ["规则冲突"]


def test_caps_score_when_intermediate_exceeds_question_max(scoring_request):
    """中间结果自身满分可大于题目满分（异常输入）；Aggregator 仍按题目满分截断。"""
    result = ScoreAggregatorComponent().aggregate(
        scoring_request,
        intermediate(
            scorer="TextRerankerScorer",
            score=12,
            confidence=0.9,
            max_score=20,
        ),
    )
    assert result.score == 10
    assert any("已封顶" in w for w in result.warnings)


def test_rejects_double_counting_full_score_engines(scoring_request):
    with pytest.raises(ValueError, match="重复计分|按题目满分"):
        ScoreAggregatorComponent().aggregate(
            scoring_request,
            [
                intermediate(scorer="TextRerankerScorer", score=7, confidence=0.9),
                intermediate(
                    scorer="RuleInterceptor",
                    score=6,
                    confidence=0.8,
                    warnings=["规则冲突"],
                ),
            ],
        )


def test_rejects_max_score_sum_exceeding_question(scoring_request):
    with pytest.raises(ValueError, match="max_score 合计"):
        ScoreAggregatorComponent().aggregate(
            scoring_request,
            [
                intermediate(
                    scorer="A", score=5, confidence=0.9, max_score=7
                ),
                intermediate(
                    scorer="B", score=5, confidence=0.9, max_score=5
                ),
            ],
        )


def test_confidence_below_review_threshold_requires_manual_review(scoring_request):
    result = ScoreAggregatorComponent().aggregate(
        scoring_request,
        intermediate(scorer="TextRerankerScorer", score=3, confidence=0.59),
    )

    assert result.review_level is ReviewLevel.MANUAL_REQUIRED
    assert result.need_manual_review is True


def test_force_manual_review_overrides_high_confidence(scoring_request):
    result = ScoreAggregatorComponent().aggregate(
        scoring_request,
        intermediate(
            scorer="SQLStructureScorer",
            score=9,
            confidence=0.95,
            scoring_mode=ScoringMode.SQL,
            force_manual_review=True,
            warnings=["SQL 解析失败，已回退空 AST"],
        ),
    )

    assert result.review_level is ReviewLevel.MANUAL_REQUIRED
    assert result.need_manual_review is True
    assert result.confidence == 0.95


def test_synthesizes_point_id_for_sql_code_evidence(scoring_request):
    req = scoring_request.model_copy(update={"scoring_mode": ScoringMode.SQL})
    result = ScoreAggregatorComponent().aggregate(
        req,
        intermediate(
            scorer="SQLStructureScorer",
            score=8,
            confidence=0.9,
            scoring_mode=ScoringMode.SQL,
            matched=[
                {
                    "score": 4,
                    "max_score": 5,
                    "evidence": "SELECT name",
                    "reason": "SELECT 字段匹配",
                }
            ],
            missed=[
                {
                    "score": 0,
                    "max_score": 5,
                    "reason": "WHERE 条件方向错误",
                }
            ],
        ),
    )

    assert result.matched_points[0].point_id == "SQLStructureScorer:matched:0"
    assert result.missed_points[0].point_id == "SQLStructureScorer:missed:0"
    assert any("合成为" in w for w in result.warnings)
    assert result.review_level is ReviewLevel.AUTO_PASS


def test_scoring_mode_mismatch_warns_and_uses_request(scoring_request):
    result = ScoreAggregatorComponent().aggregate(
        scoring_request,
        intermediate(
            scorer="SQLStructureScorer",
            score=5,
            confidence=0.9,
            scoring_mode=ScoringMode.SQL,
        ),
    )

    assert result.scoring_mode is ScoringMode.TEXT
    assert any("不一致" in w for w in result.warnings)


def test_rejects_empty_results(scoring_request):
    with pytest.raises(ValueError, match="至少需要一个"):
        ScoreAggregatorComponent().aggregate(scoring_request, [])


def test_score_rounded_to_precision(scoring_request):
    req = scoring_request.model_copy(
        update={
            "scoring_config": ScoringOptions(
                score_precision=1,
                manual_review_thresholds=ManualReviewThresholds(
                    auto_pass=0.85, review=0.6
                ),
            )
        }
    )
    result = ScoreAggregatorComponent().aggregate(
        req,
        intermediate(scorer="TextRerankerScorer", score=7.66, confidence=0.91),
    )
    assert result.score == 7.7
