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
        if point == "REST 以资源为中心":
            return 0.9 if "资源" in student else 0.1
        return 0.95 if "有状态" in student or "无状态" in student else 0.1

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
    def sim(student: str, point: str) -> float:
        if point == "REST 以资源为中心":
            return 0.95 if "资源" in student else 0.1
        return 0.95 if "有状态" in student or "无状态" in student else 0.1

    scorer = TextRerankerScorer(
        pair_scorer=sim,
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
            reference_answer="REST 以资源为中心，REST 通信保持无状态",
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
    assert result.provisional_score == 3.0
    assert result.missed_evidence[0].provisional_score == 3.0
    assert result.missed_evidence[0].relation is PointRelation.UNKNOWN
    assert result.force_manual_review is True
    assert result.metadata["decision_reason"] == "no_supported_points_uncertain"
    assert any("拒绝自动定分" in warning for warning in result.warnings)


def test_rubric_self_check_rejects_when_reference_misses_required_point():
    def sim(student: str, point: str) -> float:
        return 0.9 if "REST" in student else 0.1

    scorer = TextRerankerScorer(pair_scorer=sim, allow_model_load=False)
    result = scorer.score(
        _req(
            scoring_points=[
                {
                    "id": "resource",
                    "text": "REST 以资源为中心",
                    "score": 10,
                    "required": True,
                }
            ],
            student_answer="REST 以资源为中心",
            reference_answer="标准答案没有描述该评分点",
        )
    )

    assert result.score == 9.0
    assert result.force_manual_review is True
    assert result.metadata["decision_reason"] == "rubric_self_check_failed"
    assert result.metadata["rubric_validation"][0]["supported"] is False


def test_complete_specialist_answers_use_local_evidence_without_false_negation():
    cases = [
        {
            "id": "text-1",
            "answer": (
                "幂等表示同一请求执行一次或多次对服务器资源产生的预期效果相同。"
                "GET 只读取资源，通常幂等；PUT 按指定状态整体创建或替换资源，"
                "重复执行结果相同，通常幂等；POST 通常创建新资源或触发操作，"
                "重复提交可能产生多个结果，因此通常不幂等。"
            ),
            "points": [
                {"id": "p1", "text": "幂等是重复执行与执行一次的资源效果相同", "score": 5, "required": True},
                {"id": "p2", "text": "GET 通常幂等且用于读取", "score": 5},
                {"id": "p3", "text": "PUT 通常幂等且按目标状态创建或替换", "score": 5},
                {"id": "p4", "text": "POST 通常不幂等且重复提交可能产生多个结果", "score": 5},
            ],
        },
        {
            "id": "text-3",
            "answer": (
                "常用流程是先更新数据库，再删除缓存，并通过重试或消息队列补偿删除失败。"
                "穿透是查询不存在的数据持续绕过缓存，可用布隆过滤器或空值缓存；"
                "击穿是热点键失效瞬间大量请求访问数据库，可用互斥锁或逻辑过期；"
                "雪崩是大量键同时失效，可用随机过期时间、限流和多级缓存。"
            ),
            "points": [
                {"id": "p1", "text": "更新数据库后删除缓存并对失败进行补偿", "score": 5, "required": True},
                {"id": "p2", "text": "穿透针对不存在数据并可用布隆过滤器或空值缓存", "score": 5},
                {"id": "p3", "text": "击穿针对热点键失效并可用互斥锁或逻辑过期", "score": 5},
                {"id": "p4", "text": "雪崩针对大量键同时失效并可用随机过期或限流", "score": 5},
            ],
        },
    ]
    scorer = TextRerankerScorer(allow_model_load=False)

    for case in cases:
        result = scorer.score(
            ScoringRequest.model_validate(
                {
                    "question_id": case["id"],
                    "max_score": 20,
                    "scoring_mode": "text",
                    "reference_answer": case["answer"],
                    "student_answer": case["answer"],
                    "scoring_points": case["points"],
                }
            )
        )
        assert result.score > 0
        assert result.metadata["relation_counts"]["contradicted"] == 0
        assert not any("否定词冲突" in warning for warning in result.warnings)


def test_unrelated_specialist_answer_stays_zero_with_small_provisional_score():
    scorer = TextRerankerScorer(allow_model_load=False)
    result = scorer.score(
        _req(
            scoring_points=[
                {"id": "p1", "text": "幂等是重复执行效果相同", "score": 10, "required": True}
            ],
            reference_answer="幂等表示重复执行与执行一次效果相同。",
            student_answer="这个问题只需要增加服务器内存即可解决。",
        )
    )

    assert result.score == 0.0
    assert result.provisional_score is not None
    assert result.provisional_score <= 1.0


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
