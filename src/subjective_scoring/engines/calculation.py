"""固定计算题的确定性步骤评分。

该引擎只核验配置中的数值、单位和可选步骤标签，不执行学生提交的表达式。
"""

from __future__ import annotations

import re

from subjective_scoring.models import (
    CalculationItem,
    EvidenceItem,
    IntermediateScoreResult,
    ScoringMode,
    ScoringRequest,
)

_NUMBER_RE = re.compile(
    r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:[%％]|kg|公斤|千克|吨|t|g|克|mg|毫克|m|米|cm|厘米|s|秒|分钟|小时)?",
    re.IGNORECASE,
)
_WORKING_RE = re.compile(r"(?:=|≈|≃|~=|÷|×|\*|/|\b等于\b|\b计算\b)")
_UNIT_RE = re.compile(r"[%％]|kg|公斤|吨|g|克|mg|毫克|m|米|cm|厘米|s|秒|分钟|小时", re.IGNORECASE)


def _unit(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("％", "%")
    return {
        "公斤": "kg",
        "千克": "kg",
        "吨": "t",
        "克": "g",
        "毫克": "mg",
        "厘米": "cm",
        "分钟": "min",
        "小时": "h",
        "秒": "s",
    }.get(normalized, normalized)


def _numbers(text: str) -> list[tuple[float, str | None, str]]:
    result: list[tuple[float, str | None, str]] = []
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group(0).replace(",", "").strip()
        unit_match = _UNIT_RE.search(raw)
        unit = _unit(unit_match.group(0) if unit_match else None)
        number_match = re.match(r"-?[\d,]+(?:\.\d+)?", raw)
        if number_match is None:
            continue
        number = number_match.group(0).replace(",", "")
        try:
            result.append((float(number), unit, match.group(0)))
        except ValueError:
            continue
    return result


def _matches(value: float, candidate: float, tolerance: float, unit: str | None) -> bool:
    expected = value
    actual = candidate
    normalized_unit = _unit(unit)
    if normalized_unit == "%":
        # 配置通常使用比例（0.45），学生也可能写成 45%。
        if actual > 1.0 and expected <= 1.0:
            actual /= 100.0
    return abs(expected - actual) <= max(tolerance, 1e-9)


class CalculationScorer:
    """按题目配置的步骤和最终答案进行容错匹配。"""

    name = "CalculationScorer"

    @staticmethod
    def _candidate_lines(answer: str, item: CalculationItem, require_working: bool) -> list[str]:
        lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
        if not lines:
            return []
        keywords = tuple(k.casefold() for k in item.keywords if k.strip())
        if keywords:
            lines = [line for line in lines if any(k in line.casefold() for k in keywords)]
        if require_working:
            lines = [line for line in lines if _WORKING_RE.search(line)]
        return lines

    @classmethod
    def _find(cls, answer: str, item: CalculationItem, *, require_working: bool) -> str | None:
        for line in cls._candidate_lines(answer, item, require_working):
            for value, candidate_unit, raw in _numbers(line):
                if item.unit and _unit(candidate_unit) != _unit(item.unit):
                    continue
                if _matches(item.expected, value, item.tolerance, item.unit):
                    return raw
        return None

    def score(self, request: ScoringRequest) -> IntermediateScoreResult:
        config = request.scoring_config.calculation
        items = [*config.steps, *config.final_answers]
        if not items:
            return IntermediateScoreResult(
                scorer=self.name,
                scoring_mode=ScoringMode.CALCULATION,
                score=0.0,
                max_score=request.max_score,
                confidence=0.0,
                warnings=["未配置 calculation.steps 或 calculation.final_answers"],
                force_manual_review=True,
                metadata={"parser": "deterministic_values", "decision_reason": "missing_calculation_config"},
            )

        answer = request.student_answer or ""
        has_working = bool(_WORKING_RE.search(answer))
        matched: list[EvidenceItem] = []
        missed: list[EvidenceItem] = []
        score = 0.0
        warnings: list[str] = []
        force_review = False

        for item in items:
            is_step = item in config.steps
            require_working = config.require_working and is_step
            evidence = self._find(answer, item, require_working=require_working)
            if evidence is not None:
                score += item.score
                matched.append(
                    EvidenceItem(
                        point_id=item.id,
                        score=item.score,
                        max_score=item.score,
                        evidence=evidence,
                        reason=f"{item.description}: 数值在容许误差内",
                        similarity=1.0,
                    )
                )
            else:
                missed.append(
                    EvidenceItem(
                        point_id=item.id,
                        score=0.0,
                        max_score=item.score,
                        reason=f"{item.description}: 未找到符合数值/单位要求的答案",
                        similarity=0.0,
                    )
                )
                if is_step and config.require_working and not has_working:
                    warnings.append("要求展示计算过程，但未检测到公式或等式结构")
                force_review = True

        if config.require_working and not has_working:
            final_score = sum(item.score for item in config.final_answers if self._find(answer, item, require_working=False))
            if config.final_only_score_cap is not None:
                final_score = min(final_score, config.final_only_score_cap)
            score = final_score
            # 这是明确配置的最终答案封顶，不需要把所有只写答案的情况强制送审。
            force_review = any(
                item.score > 0 and self._find(answer, item, require_working=False) is None
                for item in config.final_answers
            )
        score = min(max(score, 0.0), request.max_score)
        all_items_matched = len(matched) == len(items)
        if all_items_matched:
            force_review = False
        confidence = 1.0 if all_items_matched else (0.55 if force_review else 0.85)
        decision_reason = "all_configured_values_matched" if all_items_matched else "calculation_values_partial"
        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=ScoringMode.CALCULATION,
            score=round(score, request.scoring_config.score_precision),
            max_score=request.max_score,
            confidence=confidence,
            matched_evidence=matched,
            missed_evidence=missed,
            warnings=list(dict.fromkeys(warnings)),
            force_manual_review=force_review,
            metadata={
                "parser": "deterministic_values",
                "decision_reason": decision_reason,
                "require_working": config.require_working,
                "working_detected": has_working,
            },
        )

    def __call__(self, request: ScoringRequest) -> IntermediateScoreResult:
        return self.score(request)


__all__ = ["CalculationScorer"]
