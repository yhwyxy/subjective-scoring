"""题型路由：按元数据优先级选择评分引擎。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from subjective_scoring.models import IntermediateScoreResult, ScoringMode, ScoringRequest

_SQL_LANGS = frozenset({"sql"})
_CODE_LANGS = frozenset(
    {
        "python",
        "py",
        "java",
        "javascript",
        "js",
        "typescript",
        "ts",
        "cpp",
        "c++",
        "cc",
        "cxx",
        "c",
        "go",
        "golang",
        "rust",
        "ruby",
        "php",
        "csharp",
        "c#",
        "kotlin",
        "swift",
    }
)

_SQL_QUESTION_TYPES = frozenset(
    {"sql", "sql_question", "database_sql", "db_sql"}
)
_CODE_QUESTION_TYPES = frozenset(
    {
        "code",
        "programming",
        "coding",
        "algorithm",
        "program",
        "code_completion",
    }
)
_TEXT_QUESTION_TYPES = frozenset(
    {
        "subjective",
        "short_answer",
        "essay",
        "text",
        "open",
        "fill_blank_subjective",
    }
)

_SQL_COURSE_HINTS = frozenset(
    {"database", "db", "sql", "mysql", "postgresql", "oracle", "sqlite"}
)
_CODE_COURSE_HINTS = frozenset(
    {
        "programming",
        "coding",
        "algorithm",
        "java",
        "python",
        "cpp",
        "software",
        "cs",
        "computer",
    }
)

_SQL_LOOKS_RE = re.compile(
    r"\b(select|insert|update|delete|with|create|drop|alter)\b",
    re.IGNORECASE,
)
_CODE_LOOKS_RE = re.compile(
    r"(^\s*(def|class|function|public\s+|private\s+|#include|import\s+|package\s+)"
    r"|[{;]\s*$|=>|::)",
    re.MULTILINE,
)


class ScorerProtocol(Protocol):
    def score(self, request: ScoringRequest) -> IntermediateScoreResult: ...


@dataclass(frozen=True)
class RouteDecision:
    """路由结果。"""

    mode: ScoringMode
    scorer_name: str
    reason: str
    signals: list[str] = field(default_factory=list)

    @property
    def track(self) -> str:
        return self.scorer_name


_MODE_TO_SCORER = {
    ScoringMode.TEXT: "TextRerankerScorer",
    ScoringMode.SQL: "SQLStructureScorer",
    ScoringMode.CODE: "CodeHybridScorer",
}


class QuestionTypeRouter:
    """元数据优先的题型路由器。

    优先级（设计文档 §6）：
    1. scoring_mode 显式字段
    2. question_type 题型字段
    3. code_language 代码语言字段
    4. course_type 课程字段
    5. reference_answer 特征
    6. student_answer 特征兜底
    默认 text。
    """

    def __init__(
        self,
        *,
        scorers: Mapping[ScoringMode, ScorerProtocol] | None = None,
        allow_content_fallback: bool = True,
    ) -> None:
        self._scorers = dict(scorers or {})
        self.allow_content_fallback = allow_content_fallback

    def resolve(self, request: ScoringRequest) -> RouteDecision:
        signals: list[str] = []

        if request.scoring_mode is not None:
            mode = request.scoring_mode
            signals.append(f"scoring_mode={mode.value}")
            return RouteDecision(
                mode=mode,
                scorer_name=_MODE_TO_SCORER[mode],
                reason="显式 scoring_mode",
                signals=signals,
            )

        qtype = (request.question_type or "").strip().lower()
        # 强题型（sql/code 专用）优先；泛化 text 类型（含默认 subjective）
        # 不压过 code_language / course_type，避免默认 question_type=subjective 短路路由。
        strong_qtype_mode: ScoringMode | None = None
        weak_text_qtype = False
        if qtype:
            signals.append(f"question_type={qtype}")
            if qtype in _SQL_QUESTION_TYPES:
                strong_qtype_mode = ScoringMode.SQL
            elif qtype in _CODE_QUESTION_TYPES:
                strong_qtype_mode = ScoringMode.CODE
            elif qtype in _TEXT_QUESTION_TYPES:
                weak_text_qtype = True

        if strong_qtype_mode is not None:
            return self._decision(strong_qtype_mode, "question_type", signals)

        lang = (request.code_language or "").strip().lower()
        if lang:
            signals.append(f"code_language={lang}")
            if lang in _SQL_LANGS:
                return self._decision(ScoringMode.SQL, "code_language", signals)
            if lang in _CODE_LANGS:
                return self._decision(ScoringMode.CODE, "code_language", signals)

        course = (request.course_type or "").strip().lower()
        if course:
            signals.append(f"course_type={course}")
            if course in _SQL_COURSE_HINTS or any(
                h in course for h in _SQL_COURSE_HINTS
            ):
                return self._decision(ScoringMode.SQL, "course_type", signals)
            if course in _CODE_COURSE_HINTS or any(
                h in course for h in ("program", "coding", "algorithm")
            ):
                return self._decision(ScoringMode.CODE, "course_type", signals)

        if weak_text_qtype:
            return self._decision(ScoringMode.TEXT, "question_type", signals)

        if self.allow_content_fallback:
            ref = request.reference_answer or ""
            stu = request.student_answer or ""
            if self._looks_like_sql(ref):
                signals.append("reference_answer~sql")
                return self._decision(
                    ScoringMode.SQL, "reference_answer 特征", signals
                )
            if self._looks_like_code(ref):
                signals.append("reference_answer~code")
                return self._decision(
                    ScoringMode.CODE, "reference_answer 特征", signals
                )
            if self._looks_like_sql(stu):
                signals.append("student_answer~sql")
                return self._decision(
                    ScoringMode.SQL, "student_answer 特征兜底", signals
                )
            if self._looks_like_code(stu):
                signals.append("student_answer~code")
                return self._decision(
                    ScoringMode.CODE, "student_answer 特征兜底", signals
                )
            signals.append("content_fallback=none")

        signals.append("default=text")
        return self._decision(ScoringMode.TEXT, "默认 text", signals)

    def select_scorer(self, request: ScoringRequest) -> tuple[ScorerProtocol, RouteDecision]:
        decision = self.resolve(request)
        scorer = self._scorers.get(decision.mode)
        if scorer is None:
            raise KeyError(
                f"未注册 {decision.mode.value} 评分引擎（需要 {decision.scorer_name}）；"
                f"请在 QuestionTypeRouter(scorers=...) 中注入"
            )
        return scorer, decision

    def route(self, request: ScoringRequest) -> IntermediateScoreResult:
        """解析路由并调用对应引擎 score。"""
        scorer, decision = self.select_scorer(request)
        result = scorer.score(request)
        # 附带路由元数据（不覆盖引擎 metadata）
        meta = dict(result.metadata)
        meta.setdefault("route_reason", decision.reason)
        meta.setdefault("route_signals", decision.signals)
        meta.setdefault("route_mode", decision.mode.value)
        return result.model_copy(update={"metadata": meta})

    def register(self, mode: ScoringMode, scorer: ScorerProtocol) -> None:
        self._scorers[mode] = scorer

    @staticmethod
    def _decision(
        mode: ScoringMode, reason: str, signals: list[str]
    ) -> RouteDecision:
        return RouteDecision(
            mode=mode,
            scorer_name=_MODE_TO_SCORER[mode],
            reason=reason,
            signals=list(signals),
        )

    @staticmethod
    def _looks_like_sql(text: str) -> bool:
        if not text or not text.strip():
            return False
        if not _SQL_LOOKS_RE.search(text):
            return False
        # 避免把自然语言中的 "select the best" 误判：需要 from 或 where 等
        lowered = text.lower()
        return " from " in f" {lowered} " or " where " in f" {lowered} " or lowered.strip().startswith(
            ("select", "with", "insert", "update", "delete")
        )

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        if not text or not text.strip():
            return False
        return bool(_CODE_LOOKS_RE.search(text))


__all__ = ["QuestionTypeRouter", "RouteDecision", "ScorerProtocol"]
