"""TextRerankerScorer 单元测试（注入相似度，不加载模型）。"""

from __future__ import annotations

from subjective_scoring import ScoringMode, ScoringRequest
from subjective_scoring.engines import RuleInterceptor, TextRerankerScorer


def _req(**kwargs) -> ScoringRequest:
    base = dict(
        question_id="q1",
        max_score=10,
        scoring_mode=ScoringMode.TEXT,
        scoring_points=[
            {"id": "p1", "text": "提高查询效率", "score": 5, "required": True},
            {"id": "p2", "text": "减少全表扫描", "score": 5, "required": False},
        ],
        student_answer="索引可以让数据库查得更快。",
        reference_answer="索引可以提高查询效率，减少全表扫描。",
    )
    base.update(kwargs)
    return ScoringRequest.model_validate(base)


def test_scores_points_with_injected_similarity():
    def sim(student: str, point: str) -> float:
        mapping = {"提高查询效率": 0.9, "减少全表扫描": 0.2}
        return mapping.get(point, 0.0)

    scorer = TextRerankerScorer(pair_scorer=sim, allow_model_load=False)
    result = scorer.score(_req())

    assert result.scorer == "TextRerankerScorer"
    assert result.scoring_mode is ScoringMode.TEXT
    assert result.score == 5.5  # 0.9*5 + 0.2*5
    assert len(result.matched_evidence) == 1
    assert result.matched_evidence[0].point_id == "p1"
    assert result.missed_evidence[0].point_id == "p2"
    assert result.force_manual_review is False
    assert result.metadata["model"] == "injected"


def test_negation_conflict_zeros_point_and_forces_review():
    def sim(student: str, point: str) -> float:
        return 0.95

    scorer = TextRerankerScorer(pair_scorer=sim, allow_model_load=False)
    result = scorer.score(
        _req(
            scoring_points=[{"id": "p1", "text": "索引可以提高查询效率", "score": 10}],
            student_answer="索引不能提高查询效率",
        )
    )

    assert result.score == 0.0
    assert result.force_manual_review is True
    assert any("否定" in w for w in result.warnings)


def test_full_reference_fallback_forces_review():
    scorer = TextRerankerScorer(
        pair_scorer=lambda s, p: 0.8,
        allow_model_load=False,
    )
    result = scorer.score(
        _req(
            scoring_points=[],
            reference_answer="完整标准答案内容",
            student_answer="完整标准答案内容",
        )
    )
    assert result.force_manual_review is True
    assert any("全文兜底" in w for w in result.warnings)
    assert result.score == 8.0


def test_empty_student_answer():
    scorer = TextRerankerScorer(pair_scorer=lambda s, p: 1.0, allow_model_load=False)
    result = scorer.score(_req(student_answer=""))
    assert result.score == 0.0
    assert result.force_manual_review is True


def test_rule_interceptor_number_mismatch():
    ri = RuleInterceptor()
    hit = ri.check("缓存时间应为 30 秒", "缓存时间应为 60 秒", "p1")
    assert any(h.kind == "number" for h in hit.hits)
