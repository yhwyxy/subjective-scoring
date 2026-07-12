"""SQLStructureScorer 单元测试。"""

from __future__ import annotations

from subjective_scoring import ScoringMode, ScoringRequest
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
