"""TextRerankerScorer 单元测试（注入相似度，不加载模型）。"""

from __future__ import annotations

from subjective_scoring import PointRelation, ScoringMode, ScoringRequest
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
    assert result.score == 4.5  # 低于支持阈值的原子评分点不再贡献残余分
    assert len(result.matched_evidence) == 1
    assert result.matched_evidence[0].point_id == "p1"
    assert result.missed_evidence[0].point_id == "p2"
    assert result.matched_evidence[0].relation is PointRelation.SUPPORTED
    assert result.missed_evidence[0].relation is PointRelation.UNKNOWN
    assert result.missed_evidence[0].score == 0.0
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
    assert result.missed_evidence[0].relation is PointRelation.CONTRADICTED
    assert any("否定" in w for w in result.warnings)
    diagnostic = result.metadata["point_diagnostics"][0]
    assert diagnostic["raw_similarity"] == 0.95
    assert diagnostic["rule_hits"][0]["evidence"] == "索引不能提高查询效率"


def test_stateless_domain_phrases_do_not_trigger_negation_conflict():
    scorer = TextRerankerScorer(
        pair_scorer=lambda student, point: 0.9,
        allow_model_load=False,
    )
    for student_answer in (
        "REST 通信是无状态的",
        "每次请求自包含，服务端不保存客户端会话状态",
    ):
        result = scorer.score(
            _req(
                scoring_points=[
                    {
                        "id": "stateless",
                        "text": "客户端与服务端通信保持无状态",
                        "score": 10,
                    }
                ],
                student_answer=student_answer,
            )
        )
        assert not any("否定词冲突" in warning for warning in result.warnings)
        assert result.score >= 8.0


def test_stateful_answer_conflicts_with_stateless_point():
    scorer = TextRerankerScorer(
        pair_scorer=lambda student, point: 0.95,
        allow_model_load=False,
    )
    result = scorer.score(
        _req(
            scoring_points=[
                {"id": "stateless", "text": "REST 通信保持无状态", "score": 10}
            ],
            student_answer="REST 是有状态的",
        )
    )
    assert result.force_manual_review is True
    assert result.score == 0.0
    assert any("否定词冲突" in warning for warning in result.warnings)


def test_supported_atomic_point_keeps_partial_credit_when_another_point_conflicts():
    def sim(student: str, point: str) -> float:
        return {"REST 以资源为中心": 0.9, "REST 通信保持无状态": 0.95}[point]

    scorer = TextRerankerScorer(pair_scorer=sim, allow_model_load=False)
    result = scorer.score(
        _req(
            scoring_points=[
                {"id": "resource", "text": "REST 以资源为中心", "score": 5},
                {"id": "stateless", "text": "REST 通信保持无状态", "score": 5},
            ],
            student_answer="REST 以资源为中心，但 REST 是有状态的",
        )
    )

    assert result.score == 4.5
    assert [item.point_id for item in result.matched_evidence] == ["resource"]
    assert result.missed_evidence[0].relation is PointRelation.CONTRADICTED
    assert result.force_manual_review is True


def test_critical_conflict_can_cap_total_score():
    scorer = TextRerankerScorer(
        pair_scorer=lambda student, point: 0.95,
        allow_model_load=False,
    )
    result = scorer.score(
        _req(
            scoring_points=[
                {"id": "resource", "text": "REST 以资源为中心", "score": 6},
                {
                    "id": "stateless",
                    "text": "REST 通信保持无状态",
                    "score": 4,
                    "critical": True,
                    "conflict_policy": "cap_total",
                    "conflict_score_cap_ratio": 0.4,
                },
            ],
            student_answer="REST 以资源为中心，但 REST 是有状态的",
        )
    )

    assert result.score == 4.0
    assert result.metadata["applied_caps"] == ["critical_conflict:0.4"]
    assert result.metadata["decision_reason"] == "critical_point_conflict_cap"


def test_no_supported_points_with_unknown_relation_rejects_auto_scoring():
    scorer = TextRerankerScorer(
        pair_scorer=lambda student, point: 0.3,
        allow_model_load=False,
    )
    result = scorer.score(
        _req(
            scoring_points=[
                {"id": "resource", "text": "REST 以资源为中心", "score": 10}
            ],
            student_answer="REST 是一种架构风格",
        )
    )

    assert result.score == 0.0
    assert result.missed_evidence[0].relation is PointRelation.UNKNOWN
    assert result.force_manual_review is True
    assert result.metadata["decision_reason"] == "no_supported_points_uncertain"
    assert any("拒绝自动定分" in warning for warning in result.warnings)


def test_negation_only_applies_to_the_local_uniform_interface_clause():
    scorer = TextRerankerScorer(
        pair_scorer=lambda student, point: 0.95,
        allow_model_load=False,
    )
    result = scorer.score(
        _req(
            scoring_points=[
                {"id": "uniform", "text": "REST 要求统一接口", "score": 10}
            ],
            student_answer="REST 不要求统一接口，但服务端不保存会话状态",
        )
    )
    assert result.force_manual_review is True
    assert any("uniform" in warning for warning in result.warnings)


def test_rule_interceptor_accepts_extensible_domain_polarity_rules():
    interceptor = RuleInterceptor(
        polarity_rules=[("read_only", r"只读", r"可写|允许写入")]
    )
    result = interceptor.check("接口必须只读", "接口允许写入数据", "readonly")
    assert any(hit.kind == "negation" for hit in result.hits)


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
