"""主观题评分顶层服务：组装归一化、路由、引擎与聚合。

设计文档：
docs/superpowers/specs/2026-07-11-subjective-scoring-design.md

第一版采用纯 Python 编排（Haystack 仅作为可选后续封装层），
判分逻辑全部在各引擎内完成，本服务不二次智能评分。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from subjective_scoring.components.aggregator import ScoreAggregatorComponent
from subjective_scoring.components.normalizer import (
    InputNormalizerComponent,
    NormalizationResult,
)
from subjective_scoring.components.router import (
    QuestionTypeRouter,
    RouteDecision,
    ScorerProtocol,
)
from subjective_scoring.engines.code_hybrid import CodeHybridScorer
from subjective_scoring.engines.sql_structure import SQLStructureScorer
from subjective_scoring.engines.text_reranker import TextRerankerScorer
from subjective_scoring.models import (
    IntermediateScoreResult,
    ScoringMode,
    ScoringRequest,
    ScoringResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineTrace:
    """单次评分的可观测轨迹（调试 / 复核后台用）。"""

    route: RouteDecision
    normalized_request: ScoringRequest
    intermediate: IntermediateScoreResult
    normalization_warnings: list[str] = field(default_factory=list)
    pipeline_warnings: list[str] = field(default_factory=list)


@dataclass
class ScoringServiceResult:
    """对外结果：标准 ScoringResult + 可选轨迹。"""

    result: ScoringResult
    trace: PipelineTrace | None = None

    def __getattr__(self, name: str) -> Any:
        # 允许 service.score(...).score / .confidence 直接访问
        return getattr(self.result, name)


class SubjectiveScoringService:
    """主观题自动评分统一入口。

    流程::

        ScoringRequest
              │
              ▼
        QuestionTypeRouter.resolve   # 元数据优先
              │
              ▼
        InputNormalizerComponent     # 按 mode 差异化清洗
              │
              ▼
        Text / SQL / Code Scorer
              │
              ▼
        ScoreAggregatorComponent
              │
              ▼
        ScoringResult

    Parameters
    ----------
    allow_model_load:
        是否允许文本/代码引擎懒加载 CrossEncoder（BGE 等）。
        测试与无 GPU 环境建议 False。
    text_model / code_model:
        CrossEncoder 的 HuggingFace 模型 ID；默认同为 BAAI/bge-reranker-base。
        可用环境变量 SUBJECTIVE_SCORING_TEXT_MODEL / CODE_MODEL 覆盖。
    strip_code_comments:
        代码归一化是否去注释。
    allow_content_fallback:
        路由在元数据不足时是否根据答案内容猜测题型。
    include_trace:
        score() 是否在 ScoringServiceResult.trace 中返回中间轨迹。
        也可用 score_with_trace() 强制返回轨迹。
    scorers / normalizer / router / aggregator:
        可注入自定义组件（测试或替换实现）。
    """

    DEFAULT_TEXT_MODEL = "BAAI/bge-reranker-base"
    DEFAULT_CODE_MODEL = "BAAI/bge-reranker-base"

    def __init__(
        self,
        *,
        allow_model_load: bool = True,
        strip_code_comments: bool = True,
        allow_content_fallback: bool = True,
        include_trace: bool = False,
        text_model: str | None = None,
        code_model: str | None = None,
        text_scorer: ScorerProtocol | None = None,
        sql_scorer: ScorerProtocol | None = None,
        code_scorer: ScorerProtocol | None = None,
        scorers: Mapping[ScoringMode, ScorerProtocol] | None = None,
        normalizer: InputNormalizerComponent | None = None,
        router: QuestionTypeRouter | None = None,
        aggregator: ScoreAggregatorComponent | None = None,
        text_pair_scorer=None,
        code_pair_scorer=None,
    ) -> None:
        """创建评分服务。

        text_model / code_model:
            HuggingFace 模型 ID，供 sentence-transformers CrossEncoder 加载。
            二者相同则进程内只缓存加载一份权重。
            也可通过环境变量覆盖：
              SUBJECTIVE_SCORING_TEXT_MODEL / SUBJECTIVE_SCORING_CODE_MODEL
        """
        import os

        self.include_trace = include_trace
        self.allow_model_load = allow_model_load
        self.text_model = (
            text_model
            or os.environ.get("SUBJECTIVE_SCORING_TEXT_MODEL")
            or self.DEFAULT_TEXT_MODEL
        )
        self.code_model = (
            code_model
            or os.environ.get("SUBJECTIVE_SCORING_CODE_MODEL")
            or text_model
            or os.environ.get("SUBJECTIVE_SCORING_TEXT_MODEL")
            or self.DEFAULT_CODE_MODEL
        )

        if scorers is not None:
            scorer_map = dict(scorers)
        else:
            scorer_map = {
                ScoringMode.TEXT: text_scorer
                or TextRerankerScorer(
                    pair_scorer=text_pair_scorer,
                    allow_model_load=allow_model_load,
                    model_name=self.text_model,
                ),
                ScoringMode.SQL: sql_scorer or SQLStructureScorer(),
                ScoringMode.CODE: code_scorer
                or CodeHybridScorer(
                    pair_scorer=code_pair_scorer,
                    allow_model_load=allow_model_load,
                    strip_comments=strip_code_comments,
                    model_name=self.code_model,
                ),
            }

        self.normalizer = normalizer or InputNormalizerComponent(
            strip_code_comments=strip_code_comments,
            write_resolved_mode=True,
        )
        self.router = router or QuestionTypeRouter(
            scorers=scorer_map,
            allow_content_fallback=allow_content_fallback,
        )
        # 若外部传入 router 但未带齐 scorers，补注册
        for mode, scorer in scorer_map.items():
            if mode not in getattr(self.router, "_scorers", {}):
                self.router.register(mode, scorer)
        self.aggregator = aggregator or ScoreAggregatorComponent()
        self._scorers = scorer_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        request: ScoringRequest | dict[str, Any],
        *,
        include_trace: bool | None = None,
    ) -> ScoringResult | ScoringServiceResult:
        """对单题评分。

        Returns
        -------
        ScoringResult
            默认返回标准结果。
        ScoringServiceResult
            当 include_trace=True（或构造时 include_trace=True）时，
            返回带 PipelineTrace 的包装对象；可通过 .result 取 ScoringResult。
        """
        req = self._coerce_request(request)
        result, trace = self._run_pipeline(req)
        want_trace = self.include_trace if include_trace is None else include_trace
        if want_trace:
            return ScoringServiceResult(result=result, trace=trace)
        return result

    def score_with_trace(
        self,
        request: ScoringRequest | dict[str, Any],
    ) -> ScoringServiceResult:
        """评分并始终返回轨迹。"""
        out = self.score(request, include_trace=True)
        assert isinstance(out, ScoringServiceResult)
        return out

    def score_many(
        self,
        requests: Sequence[ScoringRequest | dict[str, Any]],
        *,
        include_trace: bool | None = None,
    ) -> list[ScoringResult | ScoringServiceResult]:
        """批量评分（顺序执行；后续可改为线程池）。"""
        return [self.score(req, include_trace=include_trace) for req in requests]

    def __call__(
        self,
        request: ScoringRequest | dict[str, Any],
        *,
        include_trace: bool | None = None,
    ) -> ScoringResult | ScoringServiceResult:
        return self.score(request, include_trace=include_trace)

    def register_scorer(self, mode: ScoringMode, scorer: ScorerProtocol) -> None:
        """运行时替换 / 注册引擎。"""
        self._scorers[mode] = scorer
        self.router.register(mode, scorer)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        request: ScoringRequest,
    ) -> tuple[ScoringResult, PipelineTrace]:
        pipeline_warnings: list[str] = []

        # 1) 路由：使用原始元数据与原始答案做特征兜底
        decision = self.router.resolve(request)
        logger.debug(
            "route question_id=%s mode=%s reason=%s",
            request.question_id,
            decision.mode.value,
            decision.reason,
        )

        # 2) 按路由模式归一化
        norm: NormalizationResult = self.normalizer.normalize(
            request, mode=decision.mode
        )
        pipeline_warnings.extend(norm.warnings)

        normalized = norm.request
        # 保证后续聚合与结果 mode 一致
        if normalized.scoring_mode != decision.mode:
            normalized = normalized.model_copy(
                update={"scoring_mode": decision.mode}
            )

        # 3) 引擎评分
        scorer = self._scorers.get(decision.mode)
        if scorer is None:
            scorer, _ = self.router.select_scorer(normalized)

        try:
            intermediate = scorer.score(normalized)
        except Exception as exc:
            logger.exception(
                "评分引擎异常 question_id=%s mode=%s",
                request.question_id,
                decision.mode.value,
            )
            intermediate = IntermediateScoreResult(
                scorer=decision.scorer_name,
                scoring_mode=decision.mode,
                score=0.0,
                max_score=normalized.max_score,
                confidence=0.0,
                warnings=[f"评分引擎异常: {exc}"],
                force_manual_review=True,
                metadata={"error": type(exc).__name__, "message": str(exc)},
            )
            pipeline_warnings.append(f"评分引擎异常，已降级为 0 分: {exc}")

        # 附加路由信息
        meta = dict(intermediate.metadata)
        meta.setdefault("route_reason", decision.reason)
        meta.setdefault("route_signals", decision.signals)
        meta.setdefault("route_mode", decision.mode.value)
        intermediate = intermediate.model_copy(update={"metadata": meta})

        # 4) 聚合
        result = self.aggregator.aggregate(normalized, intermediate)

        # 合并流水线警告（保持引擎 / 聚合警告在前）
        if pipeline_warnings:
            result = result.model_copy(
                update={"warnings": list(result.warnings) + pipeline_warnings}
            )

        # track 以实际 scorer 为准；若与路由不一致附加说明
        if result.track != decision.scorer_name and decision.scorer_name not in result.track:
            result = result.model_copy(
                update={
                    "warnings": list(result.warnings)
                    + [
                        f"路由期望 {decision.scorer_name}，实际 track={result.track}"
                    ],
                }
            )

        trace = PipelineTrace(
            route=decision,
            normalized_request=normalized,
            intermediate=intermediate,
            normalization_warnings=list(norm.warnings),
            pipeline_warnings=pipeline_warnings,
        )
        return result, trace

    @staticmethod
    def _coerce_request(
        request: ScoringRequest | dict[str, Any],
    ) -> ScoringRequest:
        if isinstance(request, ScoringRequest):
            return request
        if isinstance(request, Mapping):
            return ScoringRequest.model_validate(request)
        raise TypeError(
            f"request 须为 ScoringRequest 或 dict，收到 {type(request).__name__}"
        )


def create_default_service(
    *,
    allow_model_load: bool = True,
    **kwargs: Any,
) -> SubjectiveScoringService:
    """工厂：创建默认配置的评分服务。"""
    return SubjectiveScoringService(allow_model_load=allow_model_load, **kwargs)


__all__ = [
    "PipelineTrace",
    "ScoringServiceResult",
    "SubjectiveScoringService",
    "create_default_service",
]
