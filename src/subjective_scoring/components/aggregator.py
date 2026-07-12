"""将评分引擎的中间结果聚合为统一最终结果。"""

from __future__ import annotations

from collections.abc import Iterable

from subjective_scoring.models import (
    IntermediateScoreResult,
    MatchedPoint,
    MissedPoint,
    REVIEW_LEVEL_RANK,
    ReviewLevel,
    ScoringDecision,
    ScoringMode,
    ScoringRequest,
    ScoringResult,
)

_SCORE_EPS = 1e-6


class ScoreAggregatorComponent:
    """只负责合并中间结果，不执行任何额外的智能评分。

    第一版约定（与设计文档 Router 单路径一致）：
    - 通常只有 1 个 IntermediateScoreResult。
    - 若传入多个，视为分项得分：各 result.max_score 之和不得超过题目满分，
      禁止每个引擎都按题目满分输出后再相加（重复计分）。
    - 语义/结构融合、规则扣分应在各 Scorer 内部完成后再交给本组件。
    """

    def aggregate(
        self,
        request: ScoringRequest,
        intermediate_results: (
            IntermediateScoreResult
            | Iterable[IntermediateScoreResult]
        ),
    ) -> ScoringResult:
        """聚合一个或多个中间结果并生成最终评分结果。

        - 分数：分项相加后按题目满分封顶，并按 score_precision 四舍五入。
        - 置信度：算术平均后保留 4 位小数。
        - 复核等级：取「置信度阈值等级」与「引擎 force_manual_review」中更严者。
        - 证据：按输入顺序合并；缺 point_id 时合成稳定 ID（供 SQL/Code 结构证据）。
        """
        results = self._normalize_results(intermediate_results)
        self._assert_no_double_counting(request, results)

        scoring_mode, mode_warnings = self._resolve_scoring_mode(request, results)
        matched_points, missed_points, evidence_warnings = self._merge_evidence(
            results
        )
        warnings = [
            warning
            for result in results
            for warning in result.warnings
        ]
        warnings.extend(mode_warnings)
        warnings.extend(evidence_warnings)

        raw_score = sum(result.score for result in results)
        capped = min(max(raw_score, 0.0), request.max_score)
        if raw_score > request.max_score + _SCORE_EPS:
            warnings.append(
                f"聚合总分 ({raw_score}) 超过题目满分 ({request.max_score})，已封顶"
            )
        precision = request.scoring_config.score_precision
        score = round(capped, precision)

        confidence = round(
            sum(result.confidence for result in results) / len(results),
            4,
        )
        review_level = self._strictest_review_level(
            self._review_level_from_confidence(
                confidence,
                request.scoring_config.manual_review_thresholds,
            ),
            self._review_level_from_engines(results),
        )
        decision_reasons = [
            str(result.metadata["decision_reason"])
            for result in results
            if result.metadata.get("decision_reason")
        ]
        if review_level != ReviewLevel.AUTO_PASS:
            decision = ScoringDecision.MANUAL_REVIEW
            decision_reason = decision_reasons[0] if decision_reasons else "review_required"
        elif score <= _SCORE_EPS:
            decision = ScoringDecision.AUTO_ZERO
            decision_reason = decision_reasons[0] if decision_reasons else "zero_score"
        else:
            decision = ScoringDecision.AUTO_SCORE
            decision_reason = decision_reasons[0] if decision_reasons else "supported_points"

        return ScoringResult(
            question_id=request.question_id,
            score=score,
            max_score=request.max_score,
            scoring_mode=scoring_mode,
            track=" + ".join(dict.fromkeys(result.scorer for result in results)),
            confidence=confidence,
            need_manual_review=review_level != ReviewLevel.AUTO_PASS,
            review_level=review_level,
            matched_points=matched_points,
            missed_points=missed_points,
            warnings=warnings,
            decision=decision,
            decision_reason=decision_reason,
        )

    def __call__(
        self,
        request: ScoringRequest,
        intermediate_results: (
            IntermediateScoreResult
            | Iterable[IntermediateScoreResult]
        ),
    ) -> ScoringResult:
        """允许组件像函数一样调用。"""
        return self.aggregate(request, intermediate_results)

    @staticmethod
    def _normalize_results(
        intermediate_results: (
            IntermediateScoreResult
            | Iterable[IntermediateScoreResult]
        ),
    ) -> list[IntermediateScoreResult]:
        if isinstance(intermediate_results, IntermediateScoreResult):
            results = [intermediate_results]
        else:
            results = list(intermediate_results)

        if not results:
            raise ValueError("至少需要一个 IntermediateScoreResult")
        if not all(isinstance(result, IntermediateScoreResult) for result in results):
            raise TypeError("intermediate_results 必须全部为 IntermediateScoreResult")
        return results

    @staticmethod
    def _assert_no_double_counting(
        request: ScoringRequest,
        results: list[IntermediateScoreResult],
    ) -> None:
        """多结果时校验 max_score 合计，防止各引擎均按题目满分输出后相加。"""
        if len(results) <= 1:
            return

        claimed_max = sum(result.max_score for result in results)
        if claimed_max > request.max_score + _SCORE_EPS:
            raise ValueError(
                f"多个中间结果 max_score 合计 ({claimed_max}) 超过题目满分 "
                f"({request.max_score})，疑似重复计分；"
                "第一版 Router 单路径应只产出一个 IntermediateScoreResult，"
                "或分项 max_score 之和不超过题目满分"
            )

        full_score_engines = [
            result.scorer
            for result in results
            if abs(result.max_score - request.max_score) <= _SCORE_EPS
        ]
        if len(full_score_engines) > 1:
            names = ", ".join(full_score_engines)
            raise ValueError(
                f"多个中间结果均按题目满分计分（{names}），禁止直接相加；"
                "请在 Scorer 内部完成融合后再交给 Aggregator"
            )

    @staticmethod
    def _resolve_scoring_mode(
        request: ScoringRequest,
        results: list[IntermediateScoreResult],
    ) -> tuple[ScoringMode, list[str]]:
        result_modes = {result.scoring_mode for result in results}
        warnings: list[str] = []

        if len(result_modes) > 1:
            modes = ", ".join(sorted(m.value for m in result_modes))
            raise ValueError(
                f"多个中间结果的 scoring_mode 不一致（{modes}），"
                "请在 request 中显式指定且保证引擎输出一致"
            )

        result_mode = results[0].scoring_mode
        if request.scoring_mode is None:
            return result_mode, warnings

        if request.scoring_mode != result_mode:
            warnings.append(
                f"request.scoring_mode={request.scoring_mode.value} 与中间结果 "
                f"scoring_mode={result_mode.value} 不一致，以 request 为准"
            )
        return request.scoring_mode, warnings

    @staticmethod
    def _merge_evidence(
        results: list[IntermediateScoreResult],
    ) -> tuple[list[MatchedPoint], list[MissedPoint], list[str]]:
        matched_points: list[MatchedPoint] = []
        missed_points: list[MissedPoint] = []
        warnings: list[str] = []

        for result in results:
            for index, evidence in enumerate(result.matched_evidence):
                point_id = evidence.point_id
                if point_id is None:
                    point_id = f"{result.scorer}:matched:{index}"
                    warnings.append(
                        f"{result.scorer} matched_evidence[{index}] 缺少 point_id，"
                        f"已合成为 {point_id}"
                    )
                matched_points.append(
                    MatchedPoint(
                        point_id=point_id,
                        score=evidence.score,
                        max_score=evidence.max_score,
                        similarity=evidence.similarity,
                        evidence=evidence.evidence,
                        reason=evidence.reason,
                        relation=evidence.relation,
                        relation_confidence=evidence.relation_confidence,
                    )
                )

            for index, evidence in enumerate(result.missed_evidence):
                point_id = evidence.point_id
                if point_id is None:
                    point_id = f"{result.scorer}:missed:{index}"
                    warnings.append(
                        f"{result.scorer} missed_evidence[{index}] 缺少 point_id，"
                        f"已合成为 {point_id}"
                    )
                missed_points.append(
                    MissedPoint(
                        point_id=point_id,
                        score=evidence.score,
                        max_score=evidence.max_score,
                        reason=evidence.reason,
                        similarity=evidence.similarity,
                        evidence=evidence.evidence,
                        relation=evidence.relation,
                        relation_confidence=evidence.relation_confidence,
                    )
                )

        return matched_points, missed_points, warnings

    @staticmethod
    def _review_level_from_confidence(confidence: float, thresholds) -> ReviewLevel:
        if confidence >= thresholds.auto_pass:
            return ReviewLevel.AUTO_PASS
        if confidence >= thresholds.review:
            return ReviewLevel.SUGGESTED_REVIEW
        return ReviewLevel.MANUAL_REQUIRED

    @staticmethod
    def _review_level_from_engines(
        results: list[IntermediateScoreResult],
    ) -> ReviewLevel:
        if any(result.force_manual_review for result in results):
            return ReviewLevel.MANUAL_REQUIRED
        return ReviewLevel.AUTO_PASS

    @staticmethod
    def _strictest_review_level(*levels: ReviewLevel) -> ReviewLevel:
        return max(levels, key=lambda level: REVIEW_LEVEL_RANK[level])


__all__ = ["ScoreAggregatorComponent"]
