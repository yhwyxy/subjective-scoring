"""有界修正（实体门槛 / 数值保底 / 短答案保底）单元测试（注入相似度，不加载模型）。"""

from __future__ import annotations

from subjective_scoring import (
    ScoringMode,
    ScoringOptions,
    ScoringRequest,
    TextBoundedCorrections,
)
from subjective_scoring.engines import TextRerankerScorer
from subjective_scoring.engines.entities import build_point_entity_profile


def _req(**kwargs) -> ScoringRequest:
    base = dict(
        question_id="q1",
        max_score=10,
        scoring_mode=ScoringMode.TEXT,
        student_answer="",
        reference_answer="",
    )
    base.update(kwargs)
    return ScoringRequest.model_validate(base)


def _scorer(sim_value: float) -> TextRerankerScorer:
    return TextRerankerScorer(
        pair_scorer=lambda student, point: sim_value,
        allow_model_load=False,
    )


# ---------------------------------------------------------------------------
# ① 实体门槛
# ---------------------------------------------------------------------------


def test_entity_gate_deflates_generic_answer():
    """套话答案语义相似度虚高时，实体零命中 -> 得分 ×0.2 且置信度受压。"""
    result = _scorer(0.9).score(
        _req(
            scoring_points=[
                {
                    "id": "p1",
                    "text": "废液倒入专用废液桶，分类收集，贴好标签",
                    "score": 10,
                }
            ],
            student_answer="按环保规范来处理就行",
            reference_answer="废液应倒入专用废液桶分类收集并贴标签",
        )
    )

    assert result.score == 2.0  # 10 * 0.2
    assert result.metadata["bounded_corrections"]["gated_points"] == ["p1"]
    diag = result.metadata["point_diagnostics"][0]
    assert diag["entity_gate"] is True
    assert diag["entity_hit_count"] == 0
    assert any("实体门槛" in w for w in result.warnings)
    # 门槛触发说明模型与规则冲突，置信度必须被压低
    assert result.confidence <= 0.5


def test_entity_gate_respects_equivalence_table():
    """同义表述（纯水≈蒸馏水、标定≈校准）命中实体 -> 不触发门槛。"""
    result = _scorer(0.9).score(
        _req(
            scoring_points=[
                {"id": "p1", "text": "使用前用蒸馏水清洗并校准", "score": 10}
            ],
            student_answer="先用纯水冲洗，再标定仪器",
        )
    )

    assert result.score == 10.0
    assert result.metadata["bounded_corrections"]["gated_points"] == []


def test_entity_gate_skips_single_entity_points():
    """实体数 < entity_gate_min_entities 的点不启用门槛（防误伤）。"""
    result = _scorer(0.9).score(
        _req(
            scoring_points=[{"id": "p1", "text": "提高查询效率", "score": 10}],
            student_answer="索引可以让数据库查得更快。",
        )
    )

    assert result.score == 10.0
    assert result.metadata["bounded_corrections"]["gated_points"] == []


def test_entity_gate_spares_paraphrase_with_cross_point_hits():
    """改述型答案在别的评分点命中过实体 -> 整卷非零命中，任何点都不压分。

    套话签名是"整卷零命中"：只要答案在某处命中过关键实体（如 HTTP），
    说明是实质作答，个别点零命中按同义改述处理，交给语义相似度判断。
    """
    result = _scorer(0.9).score(
        _req(
            scoring_points=[
                {"id": "p1", "text": "使用 HTTP 方法表达资源操作", "score": 5},
                {"id": "p2", "text": "通信保持无状态", "score": 5},
            ],
            student_answer="标准 HTTP 动词表达行为，单次调用自包含且不依赖历史上下文。",
        )
    )

    assert result.score == 10.0
    assert result.metadata["bounded_corrections"]["gated_points"] == []


def test_entity_gate_can_be_disabled():
    options = ScoringOptions(
        text_bounded_corrections=TextBoundedCorrections(enable_entity_gate=False)
    )
    result = _scorer(0.9).score(
        _req(
            scoring_points=[
                {
                    "id": "p1",
                    "text": "废液倒入专用废液桶，分类收集，贴好标签",
                    "score": 10,
                }
            ],
            student_answer="按环保规范来处理就行",
            scoring_config=options,
        )
    )

    assert result.score == 10.0


# ---------------------------------------------------------------------------
# ② 数值保底
# ---------------------------------------------------------------------------


def test_numeric_floor_lifts_correct_numeric_answer():
    """数值全对、非题干抄写的简洁答案：低相似度也保底到满分区间。"""
    result = _scorer(0.2).score(
        _req(
            question="滴定管的量程和最小分度分别是多少？",
            scoring_points=[
                {"id": "p1", "text": "量程为0~25mL，最小分度0.1mL", "score": 10}
            ],
            student_answer="0-25mL，最小分度0.1mL",
        )
    )

    assert result.score == 10.0  # 保底相似度 0.9 >= 0.85 -> 整点满分
    assert result.metadata["bounded_corrections"]["floored_points"] == [
        {"point_id": "p1", "kind": "numeric_floor"}
    ]


def test_numeric_floor_ignores_numbers_copied_from_stem():
    """答案数字全部来自题干原文 -> 无信息量，不保底。"""
    result = _scorer(0.2).score(
        _req(
            question="滴定管量程为0~25mL、最小分度0.1mL，使用时注意什么？",
            scoring_points=[
                {"id": "p1", "text": "量程为0~25mL，最小分度0.1mL", "score": 10}
            ],
            student_answer="0-25mL，0.1mL",
        )
    )

    assert result.metadata["bounded_corrections"]["floored_points"] == []
    assert result.score < 10.0


def test_numeric_floor_blocked_by_wrong_numbers():
    """关键数字答错（25、0.1 未命中）时不保底，维持原有低分流程。"""
    result = _scorer(0.2).score(
        _req(
            question="滴定管的量程和最小分度分别是多少？",
            scoring_points=[
                {"id": "p1", "text": "量程为0~25mL，最小分度0.1mL", "score": 10}
            ],
            student_answer="量程0-50mL，最小分度0.5mL",
        )
    )

    assert result.metadata["bounded_corrections"]["floored_points"] == []
    assert result.score == 2.0  # UNKNOWN 关系的 provisional 估分，未被抬升


def test_numeric_floor_skips_explanatory_points():
    """长解释型评分点即使含数字也不算“以数值为主”，不触发保底。"""
    profile = build_point_entity_profile(
        "在pH为7时溶液呈中性，酚酞不变色，因此需要继续滴定至微红色出现并保持30秒",
        "",
    )
    assert profile.primarily_numeric is False


# ---------------------------------------------------------------------------
# ③ 短答案保底
# ---------------------------------------------------------------------------


def test_short_answer_floor_lifts_concise_correct_answer():
    """简洁正确答案实体全命中 -> 相似度保底 0.7。"""
    result = _scorer(0.3).score(
        _req(
            scoring_points=[{"id": "p1", "text": "使用前需要检漏", "score": 10}],
            student_answer="先检漏",
        )
    )

    assert result.score == 7.0  # 0.7 * 10
    assert result.metadata["bounded_corrections"]["floored_points"] == [
        {"point_id": "p1", "kind": "short_answer_floor"}
    ]


def test_short_answer_floor_requires_entity_coverage():
    """实体未覆盖的短答案（套话）不被抬分。"""
    result = _scorer(0.3).score(
        _req(
            scoring_points=[{"id": "p1", "text": "使用前需要检漏", "score": 10}],
            student_answer="按要求操作",
        )
    )

    assert result.metadata["bounded_corrections"]["floored_points"] == []
    assert result.score < 7.0


def test_floor_never_lowers_existing_score():
    """保底是 max()：相似度本来就高时结果不变。"""
    high = _scorer(0.95).score(
        _req(
            scoring_points=[{"id": "p1", "text": "使用前需要检漏", "score": 10}],
            student_answer="使用前需要检漏并润洗",
        )
    )
    assert high.score == 10.0
    assert high.metadata["bounded_corrections"]["floored_points"] == []


# ---------------------------------------------------------------------------
# 互斥性：门槛与保底不可能同时触发
# ---------------------------------------------------------------------------


def test_gate_and_floor_are_mutually_exclusive():
    """任何触发了保底的点必然实体命中，不可能再被门槛压分。"""
    result = _scorer(0.2).score(
        _req(
            question="滴定管的量程和最小分度分别是多少？",
            scoring_points=[
                {"id": "p1", "text": "量程为0~25mL，最小分度0.1mL", "score": 10}
            ],
            student_answer="0-25mL，最小分度0.1mL",
        )
    )

    corrections = result.metadata["bounded_corrections"]
    floored = {item["point_id"] for item in corrections["floored_points"]}
    assert floored.isdisjoint(set(corrections["gated_points"]))
    assert corrections["floored_points"]
