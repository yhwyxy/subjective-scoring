"""固定代码题的静态评分模板。

模板验证题目要求的 AST/文本特征，不把参考代码字符串当作唯一答案，也不执行学生代码。
"""

from __future__ import annotations

import re

from subjective_scoring.components.normalizer import CodeNormalizer
from subjective_scoring.engines.code_hybrid import TreeSitterAstExtractor
from subjective_scoring.models import EvidenceItem, IntermediateScoreResult, ScoringMode, ScoringRequest


class CodeStaticScorer:
    name = "CodeStaticScorer"

    _ALLOWED_PROFILES = {"nested_loop_static", "find_index_static"}
    _ARRAY_ACCESS = re.compile(
        r"\b\w+\s*\[[^\]]+\]|\benumerate\s*\(|\.index(?:of)?\s*\(|array\.indexof\s*\(",
        re.IGNORECASE,
    )
    _COMPARISON = re.compile(r"===|==|\.equals\s*\(", re.IGNORECASE)
    _RETURN_INDEX = re.compile(
        r"\breturn\s+(?:i|idx|index|position)\b|\breturn\s+[^;\n]*\.index(?:of)?\s*\(|\breturn\s+Array\.IndexOf\s*\(",
        re.IGNORECASE,
    )

    def __init__(self, *, strip_comments: bool = True) -> None:
        self.normalizer = CodeNormalizer(strip_comments=strip_comments)
        self.extractor = TreeSitterAstExtractor()

    @staticmethod
    def _item(
        point_id: str,
        score: float,
        maximum: float,
        ok: bool,
        reason: str,
    ) -> EvidenceItem:
        return EvidenceItem(
            point_id=point_id,
            score=score if ok else 0.0,
            max_score=maximum,
            reason=reason,
            similarity=1.0 if ok else 0.0,
        )

    def _score_nested_loop(self, request: ScoringRequest, code: str) -> IntermediateScoreResult:
        features = self.extractor.extract(code, request.code_language or "python")
        checks = [
            ("language", 1.0, bool(request.code_language), "已提供代码语言"),
            ("parse", 1.0, features.parse_ok, "代码 AST 解析成功"),
            ("outer_loop", 2.0, features.flags.get("loop", False), "检测到循环结构"),
            (
                "nested_loop",
                5.0,
                features.loop_depth >= 2,
                f"循环最大嵌套深度为 {features.loop_depth}",
            ),
            ("complete_structure", 1.0, features.parse_ok and features.flags.get("function", True), "代码结构基本完整"),
        ]
        return self._build_result(request, "nested_loop_static", checks, features.parse_ok)

    def _score_find_index(self, request: ScoringRequest, code: str) -> IntermediateScoreResult:
        features = self.extractor.extract(code, request.code_language or "python")
        has_lookup_api = bool(re.search(r"\.index(?:Of)?\s*\(|Array\.IndexOf\s*\(", code, re.IGNORECASE))
        checks = [
            ("language", 1.0, bool(request.code_language), "已提供代码语言"),
            ("parse", 1.0, features.parse_ok, "代码 AST 解析成功"),
            ("array_access", 3.0, bool(self._ARRAY_ACCESS.search(code)), "检测到数组/列表访问或查找 API"),
            ("item_comparison", 3.0, bool(self._COMPARISON.search(code)) or has_lookup_api, "检测到元素比较或标准查找 API"),
            ("return_index", 5.0, bool(self._RETURN_INDEX.search(code)), "检测到返回匹配位置的逻辑"),
            ("complete_structure", 2.0, features.parse_ok and bool(features.flags.get("function", True)), "代码结构基本完整"),
        ]
        return self._build_result(request, "find_index_static", checks, features.parse_ok)

    def _build_result(
        self,
        request: ScoringRequest,
        profile: str,
        checks: list[tuple[str, float, bool, str]],
        parse_ok: bool,
    ) -> IntermediateScoreResult:
        matched: list[EvidenceItem] = []
        missed: list[EvidenceItem] = []
        score = 0.0
        for point_id, maximum, ok, reason in checks:
            item = self._item(point_id, maximum, maximum, ok, reason)
            (matched if ok else missed).append(item)
            score += item.score
        all_ok = not missed
        force_review = not parse_ok or not all_ok
        confidence = 0.9 if profile == "nested_loop_static" and all_ok else 0.8 if all_ok else 0.55
        warnings = ["代码题采用静态规则评分，未编译或执行学生代码"]
        if not parse_ok:
            warnings.append("学生代码无法解析，静态证据不足")
        if profile == "find_index_static":
            warnings.append("数组查找的运行时边界行为未执行验证")
        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=ScoringMode.CODE,
            score=round(min(score, request.max_score), request.scoring_config.score_precision),
            max_score=request.max_score,
            confidence=confidence,
            matched_evidence=matched,
            missed_evidence=missed,
            warnings=warnings,
            force_manual_review=force_review,
            metadata={
                "parser": "tree-sitter",
                "assessment_type": "static_profile",
                "code_scoring_profile": profile,
                "checks": {point_id: ok for point_id, _maximum, ok, _reason in checks},
            },
        )

    def score(self, request: ScoringRequest) -> IntermediateScoreResult:
        profile = (request.code_scoring_profile or "").strip().lower()
        if profile not in self._ALLOWED_PROFILES:
            return IntermediateScoreResult(
                scorer=self.name,
                scoring_mode=ScoringMode.CODE,
                score=0.0,
                max_score=request.max_score,
                confidence=0.0,
                warnings=[f"未知代码静态评分模板: {profile or '<empty>'}"],
                force_manual_review=True,
                metadata={"assessment_type": "static_profile"},
            )
        language = request.code_language or "python"
        code = self.normalizer.normalize(request.student_answer, language)
        if profile == "nested_loop_static":
            return self._score_nested_loop(request, code)
        return self._score_find_index(request, code)

    def __call__(self, request: ScoringRequest) -> IntermediateScoreResult:
        return self.score(request)


__all__ = ["CodeStaticScorer"]
