"""轻量相似度后端：支持注入、BGE/CrossEncoder 懒加载、关键词回退。"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)

SimilarityFn = Callable[[str, str], float]


class PairScorer(Protocol):
    """批量 (query, document) 打分接口。"""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        ...


_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_CHAR_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """中英混合分词：拉丁按词；CJK 按单字 + 双字，便于短句重叠检测。"""
    text = text or ""
    tokens: list[str] = []
    tokens.extend(t.lower() for t in _LATIN_TOKEN_RE.findall(text))
    cjk = _CJK_CHAR_RE.findall(text)
    tokens.extend(cjk)
    if len(cjk) >= 2:
        tokens.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return tokens


def lexical_similarity(a: str, b: str) -> float:
    """字符 n-gram + token Jaccard 的混合，无模型依赖，测试与回退可用。"""
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    ta, tb = set(tokenize(a)), set(tokenize(b))
    jacc = len(ta & tb) / len(ta | tb) if ta and tb else 0.0

    def grams(s: str, n: int = 2) -> set[str]:
        if len(s) < n:
            return {s}
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ga, gb = grams(a), grams(b)
    dice = (2 * len(ga & gb) / (len(ga) + len(gb))) if ga and gb else 0.0
    # 包含关系加分，但按短文本占长文本的比例衰减，避免“通常幂等”之类
    # 很短的公共片段压过包含完整主语、条件和结论的局部证据。
    contain = 0.0
    if a in b or b in a:
        shorter, longer = sorted((len(a), len(b)))
        contain = 0.85 * (shorter / longer) ** 0.5
    score = max(0.55 * jacc + 0.45 * dice, contain * 0.9)
    return float(max(0.0, min(1.0, score)))


def sigmoid(x: float) -> float:
    if x >= 20:
        return 1.0
    if x <= -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class FunctionalPairScorer:
    """将单对相似度函数包装为批量 scorer。"""

    def __init__(self, fn: SimilarityFn, name: str = "functional") -> None:
        self._fn = fn
        self.name = name

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [float(max(0.0, min(1.0, self._fn(q, d)))) for q, d in pairs]


class CrossEncoderPairScorer:
    """sentence-transformers CrossEncoder 包装。"""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        raw = self._model.predict(list(pairs), convert_to_numpy=True)
        scores: list[float] = []
        for value in raw:
            v = float(value)
            # CrossEncoder 可能输出 logits 或已归一化分数
            if v < 0.0 or v > 1.0:
                v = sigmoid(v)
            scores.append(float(max(0.0, min(1.0, v))))
        return scores


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str) -> CrossEncoderPairScorer | None:
    try:
        return CrossEncoderPairScorer(model_name)
    except Exception:
        logger.exception("加载 CrossEncoder 失败: %s", model_name)
        return None


def resolve_pair_scorer(
    *,
    prefer_model: str | None = "BAAI/bge-reranker-base",
    injected: PairScorer | SimilarityFn | None = None,
    allow_model_load: bool = True,
) -> tuple[PairScorer, str]:
    """解析相似度后端：注入 > 模型 > 词法回退。"""
    if injected is not None:
        if callable(injected) and not hasattr(injected, "score_pairs"):
            return FunctionalPairScorer(injected, name="injected"), "injected"
        return injected, getattr(injected, "name", injected.__class__.__name__)

    if allow_model_load and prefer_model:
        model = _load_cross_encoder(prefer_model)
        if model is not None:
            return model, prefer_model

    return (
        FunctionalPairScorer(lexical_similarity, name="lexical_fallback"),
        "lexical_fallback",
    )


__all__ = [
    "CrossEncoderPairScorer",
    "FunctionalPairScorer",
    "PairScorer",
    "SimilarityFn",
    "lexical_similarity",
    "resolve_pair_scorer",
    "sigmoid",
    "tokenize",
]
