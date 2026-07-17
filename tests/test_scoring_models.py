"""主观题评分数据结构（ScoringRequest / ScoringResult）单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from subjective_scoring import (
    CodeScoreWeights,
    IntermediateScoreResult,
    ManualReviewThresholds,
    MatchedPoint,
    MissedPoint,
    PointConflictPolicy,
    PointRelation,
    ReviewLevel,
    ScoringMode,
    ScoringOptions,
    ScoringPoint,
    ScoringRequest,
    ScoringResult,
    TextRelationThresholds,
)


def _sample_request_payload(**overrides):
    base = {
        "question_id": "q001",
        "paper_id": "p001",
        "question_type": "subjective",
        "scoring_mode": "text",
        "code_language": None,
        "course_type": "database",
        "max_score": 10,
        "question": "索引的作用是什么？",
        "reference_answer": "索引可以提高查询效率，减少全表扫描。",
        "scoring_points": [
            {"id": "p1", "text": "提高查询效率", "score": 5, "required": True},
            {"id": "p2", "text": "减少全表扫描", "score": 5, "required": False},
        ],
        "student_answer": "索引可以让数据库查得更快。",
        "scoring_config": {
            "manual_review_thresholds": {"auto_pass": 0.85, "review": 0.6},
            "code_score_weights": {"semantic": 0.7, "structure": 0.3},
            "allow_auto_scoring_point_generation": False,
        },
    }
    base.update(overrides)
    return base


class TestScoringRequest:
    def test_design_doc_example_parses(self):
        req = ScoringRequest.model_validate(_sample_request_payload())
        assert req.question_id == "q001"
        assert req.scoring_mode is ScoringMode.TEXT
        assert req.max_score == 10
        assert len(req.scoring_points) == 2
        assert req.scoring_points[0].required is True
        assert req.scoring_points[0].conflict_policy is PointConflictPolicy.POINT_ZERO
        assert req.scoring_config.allow_auto_scoring_point_generation is False
        assert req.scoring_config.manual_review_thresholds.auto_pass == 0.85
        assert req.scoring_config.code_score_weights.semantic == 0.7

    def test_defaults_when_optional_omitted(self):
        req = ScoringRequest(
            question_id="q1",
            max_score=5,
            student_answer="hello",
        )
        assert req.scoring_mode is None
        assert req.scoring_points == []
        assert req.scoring_config.allow_auto_scoring_point_generation is False
        assert req.question_type == "subjective"

    def test_code_language_normalized(self):
        req = ScoringRequest(
            question_id="q1",
            max_score=5,
            code_language="  Python ",
            scoring_mode=ScoringMode.CODE,
        )
        assert req.code_language == "python"

    def test_scoring_points_cannot_exceed_max_score(self):
        with pytest.raises(ValidationError, match="scoring_points"):
            ScoringRequest.model_validate(
                _sample_request_payload(
                    max_score=5,
                    scoring_points=[
                        {"id": "p1", "text": "a", "score": 3},
                        {"id": "p2", "text": "b", "score": 3},
                    ],
                )
            )

    def test_atomic_scoring_point_ids_must_be_unique(self):
        with pytest.raises(ValidationError, match="必须唯一"):
            ScoringRequest.model_validate(
                _sample_request_payload(
                    scoring_points=[
                        {"id": "same", "text": "a", "score": 3},
                        {"id": "same", "text": "b", "score": 3},
                    ]
                )
            )

    def test_rejects_negative_max_score(self):
        with pytest.raises(ValidationError):
            ScoringRequest(question_id="q1", max_score=-1)

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            ScoringRequest.model_validate(
                _sample_request_payload(unknown_field="x")
            )


class TestScoringOptions:
    def test_threshold_order_validated(self):
        with pytest.raises(ValidationError, match="review"):
            ManualReviewThresholds(auto_pass=0.5, review=0.8)

    def test_code_weights_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="code_score_weights"):
            CodeScoreWeights(semantic=0.5, structure=0.3)

    def test_default_options(self):
        opts = ScoringOptions()
        assert opts.manual_review_thresholds.auto_pass == 0.85
        assert opts.code_score_weights.structure == 0.3
        assert opts.score_precision == 1
        assert opts.text_relation_thresholds.support == 0.55
        assert opts.text_relation_thresholds.reject_when_no_supported is True

    def test_text_relation_thresholds_are_bounded(self):
        with pytest.raises(ValidationError):
            TextRelationThresholds(support=1.1)

    def test_accepts_monotonic_calibration_points(self):
        options = ScoringOptions(
            calibration_points=((0.0, 0.0), (0.08, 0.25), (0.3, 0.8), (1.0, 1.0))
        )

        assert options.calibration_points[1] == (0.08, 0.25)
        assert isinstance(options.calibration_points, tuple)

    @pytest.mark.parametrize(
        "points",
        [
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.5), (1.0, 0.4)),
            ((-0.1, 0.0), (1.0, 1.0)),
            ((0.0, 0.0),),
        ],
    )
    def test_rejects_invalid_calibration_points(self, points):
        with pytest.raises(ValueError):
            ScoringOptions(calibration_points=points)


class TestIntermediateScoreResult:
    def test_valid_intermediate(self):
        mid = IntermediateScoreResult(
            scorer="TextRerankerScorer",
            scoring_mode=ScoringMode.TEXT,
            score=8.5,
            provisional_score=9.0,
            max_score=10,
            confidence=0.9,
            matched_evidence=[
                {
                    "point_id": "p1",
                    "score": 3,
                    "provisional_score": 3,
                    "max_score": 3,
                    "evidence": "查得更快",
                    "reason": "命中评分点：提高查询效率",
                    "similarity": 0.92,
                    "relation": "supported",
                    "relation_confidence": 0.92,
                }
            ],
            missed_evidence=[
                {
                    "point_id": "p2",
                    "score": 0,
                    "max_score": 2,
                    "reason": "未明确表达减少全表扫描",
                }
            ],
            metadata={"model": "BAAI/bge-reranker-base", "parser": None},
            force_manual_review=False,
        )
        assert mid.score == 8.5
        assert mid.provisional_score == 9.0
        assert mid.force_manual_review is False
        assert len(mid.matched_evidence) == 1
        assert mid.matched_evidence[0].relation is PointRelation.SUPPORTED
        assert mid.metadata["model"] == "BAAI/bge-reranker-base"

    def test_score_cannot_exceed_max(self):
        with pytest.raises(ValidationError, match="max_score"):
            IntermediateScoreResult(
                scorer="SQLStructureScorer",
                scoring_mode=ScoringMode.SQL,
                score=11,
                max_score=10,
                confidence=0.5,
            )


class TestScoringResult:
    def test_design_doc_example_parses(self):
        result = ScoringResult(
            question_id="q001",
            score=7.7,
            provisional_score=8.0,
            max_score=10,
            scoring_mode=ScoringMode.TEXT,
            track="TextRerankerScorer",
            confidence=0.86,
            need_manual_review=False,
            review_level=ReviewLevel.AUTO_PASS,
            matched_points=[
                MatchedPoint(
                    point_id="p1",
                    score=3.0,
                    max_score=3.0,
                    similarity=0.9,
                    evidence="学生答案中对应知识点 A 的表达",
                    reason="命中评分点：掌握知识点 A",
                )
            ],
            missed_points=[
                MissedPoint(
                    point_id="p2",
                    score=0,
                    max_score=3.0,
                    reason="未明确表达知识点 B",
                )
            ],
            warnings=[],
        )
        assert result.review_level is ReviewLevel.AUTO_PASS
        assert result.provisional_score == 8.0
        assert result.need_manual_review is False
        assert result.matched_points[0].similarity == 0.9

    def test_need_manual_review_must_match_level(self):
        with pytest.raises(ValidationError, match="need_manual_review"):
            ScoringResult(
                question_id="q1",
                score=5,
                max_score=10,
                scoring_mode=ScoringMode.TEXT,
                track="TextRerankerScorer",
                confidence=0.5,
                need_manual_review=False,
                review_level=ReviewLevel.MANUAL_REQUIRED,
            )

    def test_suggested_review_requires_flag(self):
        result = ScoringResult(
            question_id="q1",
            score=6,
            max_score=10,
            scoring_mode=ScoringMode.SQL,
            track="SQLStructureScorer",
            confidence=0.72,
            need_manual_review=True,
            review_level=ReviewLevel.SUGGESTED_REVIEW,
        )
        assert result.need_manual_review is True

    def test_round_trip_json(self):
        req = ScoringRequest.model_validate(_sample_request_payload())
        restored = ScoringRequest.model_validate_json(req.model_dump_json())
        assert restored.question_id == req.question_id
        assert restored.scoring_points[1].text == "减少全表扫描"
