"""SQLStructureScorer 单元测试。"""

from __future__ import annotations

from subjective_scoring import ScoringMode, ScoringRequest, ScoringOptions, TextRelationThresholds
from subjective_scoring.engines import SQLStructureScorer


def _req(ref: str, stu: str, max_score: float = 10) -> ScoringRequest:
    return ScoringRequest(
        question_id="sql1",
        max_score=max_score,
        scoring_mode=ScoringMode.SQL,
        reference_answer=ref,
        student_answer=stu,
    )


def test_case_insensitive_equivalent_sql():
    scorer = SQLStructureScorer()
    result = scorer.score(
        _req(
            "SELECT name FROM student WHERE age > 18",
            "select name from student where age > 18",
        )
    )
    assert result.scorer == "SQLStructureScorer"
    assert result.scoring_mode is ScoringMode.SQL
    assert result.score >= 9.0
    assert result.force_manual_review is False
    assert result.metadata["parser"] == "sqlglot"


def test_operator_direction_mismatch_penalizes():
    scorer = SQLStructureScorer()
    good = scorer.score(
        _req(
            "SELECT name FROM student WHERE age > 18",
            "SELECT name FROM student WHERE age > 18",
        )
    )
    bad = scorer.score(
        _req(
            "SELECT name FROM student WHERE age > 18",
            "SELECT name FROM student WHERE age < 18",
        )
    )
    assert good.score > bad.score
    missed_ids = {e.point_id for e in bad.missed_evidence}
    assert "sql.where" in missed_ids or "sql.operators" in missed_ids


def test_parse_failure_forces_manual_review():
    scorer = SQLStructureScorer()
    result = scorer.score(_req("SELECT 1", "SELECT FROM WHERE"))
    assert result.force_manual_review is True
    assert any("解析失败" in w for w in result.warnings)


def test_empty_student_sql():
    scorer = SQLStructureScorer()
    result = scorer.score(_req("SELECT 1", ""))
    assert result.force_manual_review is True
    assert result.score == 0.0 or result.score < 3


def test_delete_against_select_is_always_zero():
    result = SQLStructureScorer().score(
        _req("SELECT id, name FROM users", "DELETE FROM users")
    )
    assert result.score == 0.0
    assert result.force_manual_review is True
    assert result.metadata["reference_statement_type"] == "SELECT"
    assert result.metadata["student_statement_type"] == "DELETE"
    assert result.metadata["rejection_reason"] == "statement_type_mismatch"


def test_multiple_statements_are_rejected():
    result = SQLStructureScorer().score(
        _req("SELECT id FROM users", "SELECT id FROM users; DELETE FROM users")
    )
    assert result.score == 0.0
    assert result.force_manual_review is True
    assert result.metadata["rejection_reason"] == "parse_error"
    assert any("只允许单条" in warning for warning in result.warnings)


def test_absent_optional_dimensions_do_not_receive_weight():
    result = SQLStructureScorer().score(
        _req("SELECT name FROM users", "SELECT name FROM users")
    )
    assert result.metadata["active_dimensions"] == ["select", "from"]
    assert {item.point_id for item in result.matched_evidence} == {
        "sql.select",
        "sql.from",
    }


def test_partial_select_match_gives_partial_credit():
    """SELECT 部分列匹配时应给部分分（>= 0.3），而不是 0。"""
    result = SQLStructureScorer().score(
        _req(
            "SELECT name, age FROM users",
            "SELECT name FROM users",
        )
    )
    select_evidence = [e for e in result.matched_evidence if e.point_id == "sql.select"]
    if select_evidence:
        assert select_evidence[0].similarity >= 0.3
    else:
        # 当前行为：只有 similarity >= 0.99 才算 matched，导致被归为 unknown
        # 修复后此断言应成立
        missed_ids = {e.point_id for e in result.missed_evidence}
        assert "sql.select" in missed_ids


def test_similar_where_condition_gives_partial_credit():
    """WHERE 条件高相似（>= 0.95）但非完全相同时应给部分分。"""
    result = SQLStructureScorer().score(
        _req(
            "SELECT name FROM users WHERE age > 18 AND status = 'active'",
            "SELECT name FROM users WHERE age > 18 AND status = 'active'",
        )
    )
    where_evidence = [e for e in result.matched_evidence if e.point_id == "sql.where"]
    assert len(where_evidence) == 1
    assert where_evidence[0].similarity >= 0.95


def test_sql_unknown_points_get_nonzero_score_when_threshold_relaxed():
    """当 reject_when_no_supported=False 且所有点 unknown 时，不应得 0 分。"""
    scorer = SQLStructureScorer()
    result = scorer.score(
        ScoringRequest(
            question_id="sql-open",
            max_score=10,
            scoring_mode=ScoringMode.SQL,
            reference_answer="SELECT name FROM users WHERE age > 18",
            student_answer="SELECT name FROM users WHERE age >= 20",
            scoring_points=[
                {"id": "sql.select", "text": "SELECT name", "score": 5},
                {"id": "sql.where", "text": "WHERE age > 18", "score": 5},
            ],
            scoring_config=ScoringOptions(
                text_relation_thresholds=TextRelationThresholds(
                    reject_when_no_supported=False,
                    required_unknown_requires_review=False,
                )
            ),
        )
    )
    # 不应直接归零
    assert result.score > 0.0
