"""LLM 判分分支：OpenAI 兼容 chat/completions 客户端 + 全题型 LLM 评分器。

设计文档：docs/superpowers/specs/2026-08-05-llm-judge-design.md

三个公开类型 + 一个异常族：
- ``LLMJudgeConfig``：判分网关配置（pydantic BaseModel）。
- ``LLMJudgeClient``：OpenAI 兼容 /chat/completions 客户端（httpx，可注入）。
- ``LLMJudgeScorer``：实现 ScorerProtocol 的整题 LLM 评分器。
- ``LLMJudgeError`` / ``LLMJudgeRequestError`` / ``LLMJudgeResponseError``。

安全约定（沿用 rerankers/cohere.py）：api_key 以 pydantic SecretStr 私有存储，
不进 repr / 异常文本；异常不整段回显响应体（网关可能回显考生作答内容）。
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from subjective_scoring.components.router import ScorerProtocol
from subjective_scoring.models import (
    EvidenceItem,
    IntermediateScoreResult,
    JudgeBackend,
    PointRelation,
    ScoringMode,
    ScoringPoint,
    ScoringRequest,
)

# ---------------------------------------------------------------------------
# 异常族
# ---------------------------------------------------------------------------


class LLMJudgeError(RuntimeError):
    """LLM 判分基础异常。"""


class LLMJudgeRequestError(LLMJudgeError):
    """请求在收到可用响应前失败（网络 / 超时 / 非 2xx）。"""


class LLMJudgeResponseError(LLMJudgeError):
    """远程服务返回了无法解析的响应（非 JSON / 结构损坏）。"""


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


class LLMJudgeConfig(BaseModel):
    """OpenAI 兼容判分网关配置。

    url 兼容传完整 /chat/completions 地址（如
    ``https://router.tumuer.me/v1/chat/completions``），也可只传前缀
    （``https://router.tumuer.me/v1``），客户端自动拼接。
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="OpenAI 兼容网关地址，可带 /v1 前缀")
    api_key: SecretStr = Field(..., description="网关 API Key，私有存储不进 repr/异常")
    model: str = Field(..., description="判分模型名称")
    timeout: float = Field(
        default=90.0,
        gt=0,
        description="请求超时（秒）；LLM 判分时延高于 reranker",
    )
    max_retries: int = Field(default=2, ge=0, description="最大重试次数")
    retry_backoff_seconds: float = Field(
        default=1.0,
        ge=0,
        description="重试退避基数（秒），按 2^n 指数退避",
    )
    temperature: float = Field(
        default=0.0,
        ge=0,
        le=2.0,
        description="采样温度；判分固定 0 保证确定性",
    )
    max_tokens: int = Field(default=2048, ge=1, description="生成最大 token 数")
    json_mode: bool = Field(
        default=True,
        description="发送 response_format={'type':'json_object'}",
    )


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


def _load_httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise ImportError(
            "LLM judge requires the 'remote' extra: "
            "pip install 'subjective-scoring[remote]'"
        ) from exc
    return httpx


class LLMJudgeClient:
    """OpenAI 兼容 ``POST {url}/chat/completions`` 客户端。

    构造器支持注入 ``client``（测试用 ``httpx.MockTransport``）；未注入时自持
    httpx.Client 并支持 ``close()`` / 上下文管理。httpx 归属现有 remote extra，
    不新增依赖。
    """

    def __init__(
        self,
        *,
        config: LLMJudgeConfig,
        client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._httpx = _load_httpx()
        self._api_key = config.api_key.get_secret_value()
        self._sleep_fn = sleep_fn
        self._owns_client = client is None
        self._client = (
            client if client is not None else self._httpx.Client(timeout=config.timeout)
        )
        self._endpoint = self._completions_endpoint()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(url={self._endpoint!r}, "
            f"model={self._config.model!r})"
        )

    def __enter__(self) -> LLMJudgeClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _completions_endpoint(self) -> str:
        url = self._config.url.strip().rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        return f"{url}/chat/completions"

    def chat_completions(self, messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        """发送一次 chat/completions 请求并返回解析后的响应体。"""
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [dict(message) for message in messages],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if self._config.json_mode:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._client.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except self._httpx.HTTPError:
                if attempt >= self._config.max_retries:
                    raise LLMJudgeRequestError(
                        "LLM judge request failed"
                    ) from None
                self._sleep_fn(self._config.retry_backoff_seconds * (2**attempt))
                continue

            if 200 <= response.status_code < 300:
                break

            retryable_status = response.status_code == 429 or (
                500 <= response.status_code < 600
            )
            if not retryable_status or attempt >= self._config.max_retries:
                raise LLMJudgeRequestError(
                    f"LLM judge returned HTTP {response.status_code}"
                )

            delay = self._config.retry_backoff_seconds * (2**attempt)
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    retry_after_seconds = float(retry_after)
                except ValueError:
                    retry_after_seconds = -1.0
                if math.isfinite(retry_after_seconds) and retry_after_seconds >= 0:
                    delay = max(delay, retry_after_seconds)
            self._sleep_fn(delay)

        try:
            body = response.json()
        except ValueError:
            raise LLMJudgeResponseError(
                "LLM judge returned invalid JSON"
            ) from None

        if not isinstance(body, Mapping):
            raise LLMJudgeResponseError(
                "LLM judge response must be an object"
            )
        return body


# ---------------------------------------------------------------------------
# LLM 评分器
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是一名严谨的主观题自动评分专家。请严格依据参考答案与评分点，对学生答案逐点评分。
评分要求：
1. 只输出一个 JSON 对象，禁止输出任何解释文字、markdown 代码块或前后缀。
2. 每个评分点独立评分，分数不得超过该评分点满分。
3. 所有评分点得分之和不得超过题目满分。
4. relation 只能取 "supported"（学生答案支持该评分点）、"contradicted"（与评分点结论矛盾）、"unknown"（无法判断）之一。
5. evidence 给出学生答案中支撑该判分的原文片段，reason 简要说明判分理由。
输出格式：
{"points": [{"point_id": "评分点ID", "score": 分数, "relation": "supported|contradicted|unknown", "confidence": 0~1, "evidence": "片段", "reason": "理由"}], "notes": "整题补充说明（可选）"}"""

_OPEN_ENDED_PROMPT_NOTE = (
    "本题未配置显式评分点，请对整题作答给出综合评分，"
    "将综合分填入 point_id 为 \"whole\" 的评分点（满分等于题目满分）。"
)

_EXACT_WHITESPACE_RE = re.compile(r"\s+")
_EXACT_PUNCTUATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "、": ",",
        "；": ";",
        "：": ":",
        "！": "!",
        "？": "?",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "…": "...",
        "—": "-",
        "–": "-",
    }
)


def _normalize_exact_match_text(text: str) -> str:
    """确定性短路使用的宽松归一化：NFKC + 大小写 + 标点 + 空白。"""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.casefold().translate(_EXACT_PUNCTUATION)
    return _EXACT_WHITESPACE_RE.sub(" ", normalized).strip()


def _parse_relation(value: Any) -> PointRelation:
    if isinstance(value, PointRelation):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "supported":
            return PointRelation.SUPPORTED
        if lowered == "contradicted":
            return PointRelation.CONTRADICTED
        if lowered == "unknown":
            return PointRelation.UNKNOWN
    return PointRelation.UNKNOWN


class LLMJudgeScorer:
    """全题型 LLM 评分器（text / code / sql / calculation）。

    Parameters
    ----------
    config:
        判分网关配置。
    client:
        可注入的 LLMJudgeClient（测试用 MockTransport）；未注入时自持。
    fallback:
        降级后端（经典引擎）；LLM 失败时回退。None 表示失败即强制人工复核。
    """

    name = "LLMJudgeScorer"

    def __init__(
        self,
        *,
        config: LLMJudgeConfig,
        client: LLMJudgeClient | Any | None = None,
        fallback: ScorerProtocol | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self.client = (
            client if client is not None else LLMJudgeClient(config=config)
        )
        self.fallback = fallback

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(url={self.client!r}, "
            f"model={self.config.model!r}, fallback={self.fallback is not None})"
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    # ------------------------------------------------------------------
    # ScorerProtocol
    # ------------------------------------------------------------------

    def score(self, request: ScoringRequest) -> IntermediateScoreResult:
        # D5：请求级显式回退经典引擎，不发起 HTTP。
        if request.scoring_config.judge_backend == JudgeBackend.RERANKER:
            if self.fallback is None:
                return self._manual_review_result(
                    request,
                    warnings=["judge_backend=reranker 但未配置降级后端"],
                    decision_reason="reranker_fallback_unavailable",
                )
            return self.fallback.score(request)

        # 4.1 确定性短路（与经典引擎一致，不消耗 API）
        student = (request.student_answer or "").strip()
        reference = (request.reference_answer or "").strip()
        if not student:
            return self._shortcut_result(
                request, score=0.0, decision_reason="blank_answer"
            )
        if reference and _normalize_exact_match_text(student) == _normalize_exact_match_text(
            reference
        ):
            return self._shortcut_result(
                request,
                score=request.max_score,
                decision_reason="exact_reference_match",
            )

        messages = self._build_messages(request)
        start = time.monotonic()
        try:
            body = self.client.chat_completions(messages)
            content = self._extract_content(body)
            parsed = self._parse_json_content(content)
            return self._map_result(request, parsed, latency_ms=(time.monotonic() - start) * 1000)
        except LLMJudgeError as exc:
            if self.fallback is not None:
                return self._fallback_result(request, exc)
            return self._manual_review_result(
                request,
                warnings=[f"LLM judge 判分失败：{type(exc).__name__}"],
                decision_reason="llm_judge_error",
                llm_error=type(exc).__name__,
            )

    # ------------------------------------------------------------------
    # Prompt 组装
    # ------------------------------------------------------------------

    def _build_messages(
        self, request: ScoringRequest
    ) -> list[dict[str, str]]:
        points = list(request.scoring_points)
        implicit_whole = not points
        if implicit_whole:
            points = [
                ScoringPoint(id="whole", text="整题综合评分", score=request.max_score)
            ]
        user_prompt = self._build_user_prompt(request, points, implicit_whole)
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _build_user_prompt(
        self,
        request: ScoringRequest,
        points: list[ScoringPoint],
        implicit_whole: bool,
    ) -> str:
        parts: list[str] = []
        mode = request.scoring_mode or ScoringMode.TEXT
        parts.append(f"题目：{request.question or '（无题干）'}")
        parts.append(f"题型：{mode.value}")
        if mode in (ScoringMode.CODE, ScoringMode.SQL):
            language = request.code_language or ("sql" if mode == ScoringMode.SQL else "")
            if language:
                parts.append(f"编程语言：{language}")
        if mode == ScoringMode.CALCULATION:
            calc = request.scoring_config.calculation
            calc_lines = [
                f"步骤 {item.id}（满分 {item.score}）：{item.description}，"
                f"期望值 {item.expected}{' ' + item.unit if item.unit else ''}"
                for item in calc.steps
            ]
            calc_lines += [
                f"最终答案 {item.id}（满分 {item.score}）：{item.description}，"
                f"期望值 {item.expected}{' ' + item.unit if item.unit else ''}"
                for item in calc.final_answers
            ]
            if calc_lines:
                parts.append("计算配置：\n" + "\n".join(calc_lines))
        parts.append(f"参考答案：{request.reference_answer or '（无）'}")
        if implicit_whole:
            parts.append(_OPEN_ENDED_PROMPT_NOTE)
        else:
            point_lines = "\n".join(
                f"- {point.id}（满分 {point.score}）：{point.text}"
                for point in points
            )
            parts.append(f"评分点：\n{point_lines}")
        parts.append(f"学生答案：{request.student_answer or '（空）'}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(body: Mapping[str, Any]) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMJudgeResponseError("LLM judge response missing choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise LLMJudgeResponseError("LLM judge choice must be an object")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise LLMJudgeResponseError("LLM judge choice missing message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMJudgeResponseError("LLM judge message content must be non-empty")
        return content

    @staticmethod
    def _parse_json_content(content: str) -> Mapping[str, Any]:
        text = content.strip()
        # 兼容 markdown 代码块包裹
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            # 网关未支持 json_object 时，从 content 提取首个 { 到末尾 }
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end <= start:
                raise LLMJudgeResponseError(
                    "LLM judge output is not valid JSON"
                ) from None
            try:
                parsed = json.loads(text[start : end + 1])
            except (ValueError, TypeError):
                raise LLMJudgeResponseError(
                    "LLM judge output is not valid JSON"
                ) from None
        if not isinstance(parsed, Mapping):
            raise LLMJudgeResponseError("LLM judge output must be a JSON object")
        return parsed

    # ------------------------------------------------------------------
    # 结果映射
    # ------------------------------------------------------------------

    def _map_result(
        self,
        request: ScoringRequest,
        parsed: Mapping[str, Any],
        *,
        latency_ms: float,
    ) -> IntermediateScoreResult:
        points = list(request.scoring_points)
        implicit_whole = not points
        if implicit_whole:
            points = [
                ScoringPoint(id="whole", text="整题综合评分", score=request.max_score)
            ]
        point_by_id = {point.id: point for point in points}
        precision = request.scoring_config.score_precision

        entries = parsed.get("points")
        if not isinstance(entries, list):
            raise LLMJudgeResponseError("LLM judge points must be a list")

        matched: list[EvidenceItem] = []
        missed: list[EvidenceItem] = []
        warnings: list[str] = []
        seen_ids: set[str] = set()
        total = 0.0
        weighted_conf = 0.0
        weight_sum = 0.0
        force_review = False
        relation_counts = {
            "supported": 0,
            "contradicted": 0,
            "unknown": 0,
        }

        for entry in entries:
            if not isinstance(entry, Mapping):
                warnings.append("LLM judge 返回的评分点条目不是对象，已丢弃")
                continue
            point_id = entry.get("point_id")
            if point_id not in point_by_id:
                warnings.append(f"LLM judge 返回未知评分点 {point_id}，已丢弃")
                continue
            seen_ids.add(point_id)
            point = point_by_id[point_id]

            score_value = float(max(0.0, min(point.score, _as_float(entry.get("score")))))
            confidence = float(max(0.0, min(1.0, _as_float(entry.get("confidence")))))
            relation = _parse_relation(entry.get("relation"))
            # 隐式整题点：relation 由得分推断（score>0 -> supported，score=0 -> unknown）
            if implicit_whole:
                relation = (
                    PointRelation.SUPPORTED
                    if score_value > 0
                    else PointRelation.UNKNOWN
                )

            evidence_text = entry.get("evidence")
            reason = entry.get("reason")
            item = EvidenceItem(
                point_id=point_id,
                score=round(score_value, precision),
                max_score=point.score,
                evidence=str(evidence_text) if evidence_text else None,
                reason=str(reason) if reason else None,
                relation=relation,
                relation_confidence=round(confidence, 4),
            )
            total += score_value
            weighted_conf += confidence * point.score
            weight_sum += point.score
            relation_counts[relation.value] += 1
            if relation == PointRelation.SUPPORTED:
                matched.append(item)
            else:
                missed.append(item)
            if (
                relation in (PointRelation.UNKNOWN, PointRelation.CONTRADICTED)
                and (point.required or point.critical)
            ):
                force_review = True
                warnings.append(
                    f"必答/关键评分点 {point_id} 关系为 {relation.value}，要求人工复核"
                )

        # 缺失评分点按 0 分进入 missed_evidence
        for point in points:
            if point.id not in seen_ids:
                missed.append(
                    EvidenceItem(
                        point_id=point.id,
                        score=0.0,
                        max_score=point.score,
                        reason="LLM judge 未返回该评分点，按 0 分计",
                        relation=PointRelation.UNKNOWN,
                        relation_confidence=0.0,
                    )
                )
                warnings.append(f"LLM judge 缺失评分点 {point.id}，按 0 分计")
                relation_counts["unknown"] += 1
                if point.required or point.critical:
                    force_review = True

        # 总分封顶 max_score；confidence 按评分点分值加权平均
        final_score = round(min(max(total, 0.0), request.max_score), precision)
        confidence = (
            round(float(max(0.0, min(1.0, weighted_conf / weight_sum))), 4)
            if weight_sum
            else 0.0
        )

        if force_review:
            decision_reason = "llm_point_unknown_or_conflict"
            decision = "manual_review"
        elif final_score <= 0.0:
            decision_reason = "no_supported_points"
            decision = "auto_zero"
        else:
            decision_reason = "supported_points"
            decision = "auto_score"

        notes = parsed.get("notes")
        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=request.scoring_mode or ScoringMode.TEXT,
            score=final_score,
            max_score=request.max_score,
            confidence=confidence,
            matched_evidence=matched,
            missed_evidence=missed,
            warnings=warnings,
            force_manual_review=force_review,
            metadata={
                "model": self.config.model,
                "judge_backend": "llm",
                "decision_reason": decision_reason,
                "decision": decision,
                "latency_ms": round(latency_ms, 1),
                "implicit_whole": implicit_whole,
                "point_count": len(points),
                "relation_counts": relation_counts,
                "llm_notes": str(notes) if notes else None,
            },
        )

    # ------------------------------------------------------------------
    # 短路 / 降级 / 强制人工
    # ------------------------------------------------------------------

    def _shortcut_result(
        self,
        request: ScoringRequest,
        *,
        score: float,
        decision_reason: str,
    ) -> IntermediateScoreResult:
        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=request.scoring_mode or ScoringMode.TEXT,
            score=score,
            max_score=request.max_score,
            confidence=1.0,
            metadata={
                "model": self.config.model,
                "judge_backend": "llm",
                "decision_reason": decision_reason,
                "decision": "auto_zero" if score <= 0 else "auto_score",
                "deterministic": True,
            },
        )

    def _fallback_result(
        self,
        request: ScoringRequest,
        exc: LLMJudgeError,
    ) -> IntermediateScoreResult:
        result = self.fallback.score(request)
        return result.model_copy(
            update={
                "warnings": list(result.warnings)
                + [f"LLM judge 失败已回退经典引擎：{type(exc).__name__}"],
                "metadata": {
                    **result.metadata,
                    "judge_backend": "llm_fallback",
                    "llm_error": type(exc).__name__,
                },
            }
        )

    def _manual_review_result(
        self,
        request: ScoringRequest,
        *,
        warnings: list[str],
        decision_reason: str,
        llm_error: str | None = None,
    ) -> IntermediateScoreResult:
        metadata: dict[str, Any] = {
            "model": self.config.model,
            "judge_backend": "llm",
            "decision_reason": decision_reason,
            "decision": "manual_review",
        }
        if llm_error is not None:
            metadata["llm_error"] = llm_error
        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=request.scoring_mode or ScoringMode.TEXT,
            score=0.0,
            max_score=request.max_score,
            confidence=0.0,
            warnings=list(warnings),
            force_manual_review=True,
            metadata=metadata,
        )

    def __call__(self, request: ScoringRequest) -> IntermediateScoreResult:
        return self.score(request)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "LLMJudgeConfig",
    "LLMJudgeClient",
    "LLMJudgeScorer",
    "LLMJudgeError",
    "LLMJudgeRequestError",
    "LLMJudgeResponseError",
]
