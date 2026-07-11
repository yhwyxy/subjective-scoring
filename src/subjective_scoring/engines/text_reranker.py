"""文本主观题：结构化评分点 + 语义匹配 + 规则拦截。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from subjective_scoring.engines._similarity import (
    PairScorer,
    SimilarityFn,
    lexical_similarity,
    resolve_pair_scorer,
    tokenize,
)
from subjective_scoring.models import (
    EvidenceItem,
    IntermediateScoreResult,
    ScoringMode,
    ScoringPoint,
    ScoringRequest,
)

_DEFAULT_MATCH_THRESHOLD = 0.55
_NEGATION_RE = re.compile(
    r"(不|没|无|非|未|并非|无法|不能|不会|不可|没有|不是|并非是|never|not|no|without|cannot|can't|won't)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?%?|[零〇一二三四五六七八九十百千万亿两]+)",
)
_UNIT_RE = re.compile(
    r"(ms|s|秒|分钟|小时|天|%|％|倍|次|个|条|行|列|MB|GB|KB|mb|gb|kb)",
    re.IGNORECASE,
)
_DIRECTION_PAIRS = (
    ("提高", "降低"),
    ("增加", "减少"),
    ("上升", "下降"),
    ("大于", "小于"),
    (">", "<"),
    (">=", "<="),
    ("left", "right"),
    ("升序", "降序"),
    ("正向", "反向"),
)
_ANTONYM_PAIRS = (
    ("正确", "错误"),
    ("成功", "失败"),
    ("同步", "异步"),
    ("静态", "动态"),
    ("开启", "关闭"),
    ("允许", "禁止"),
)


@dataclass
class ResolvedScoringPoint:
    id: str
    text: str
    score: float
    required: bool = False
    synthetic: bool = False


@dataclass
class RuleHit:
    point_id: str
    kind: str
    message: str
    severity: str = "hard"  # hard -> zero score + force review; soft -> penalize


@dataclass
class RuleInterceptResult:
    hits: list[RuleHit] = field(default_factory=list)

    @property
    def hard_hits(self) -> list[RuleHit]:
        return [h for h in self.hits if h.severity == "hard"]


class ScoringPointResolver:
    """人工评分点优先；否则标准答案全文兜底（可标记强制复核）。"""

    def resolve(self, request: ScoringRequest) -> tuple[list[ResolvedScoringPoint], list[str], bool]:
        warnings: list[str] = []
        force_review = False

        if request.scoring_points:
            points = [
                ResolvedScoringPoint(
                    id=p.id,
                    text=p.text,
                    score=float(p.score),
                    required=p.required,
                )
                for p in request.scoring_points
            ]
            return points, warnings, force_review

        ref = (request.reference_answer or "").strip()
        if ref:
            warnings.append("未配置 scoring_points，使用 reference_answer 全文兜底")
            force_review = True
            return (
                [
                    ResolvedScoringPoint(
                        id="__full_reference__",
                        text=ref,
                        score=float(request.max_score),
                        required=True,
                        synthetic=True,
                    )
                ],
                warnings,
                force_review,
            )

        if request.scoring_config.allow_auto_scoring_point_generation:
            warnings.append(
                "allow_auto_scoring_point_generation=true 但第一版未实现自动生成评分点"
            )
        warnings.append("无评分点且无标准答案，文本评分无法进行")
        force_review = True
        return [], warnings, force_review


class RuleInterceptor:
    """轻量规则拦截：否定、数字、单位、方向、反义。"""

    def check(self, point_text: str, student_answer: str, point_id: str) -> RuleInterceptResult:
        hits: list[RuleHit] = []
        pt = point_text or ""
        sa = student_answer or ""

        if self._negation_conflict(pt, sa):
            hits.append(
                RuleHit(
                    point_id=point_id,
                    kind="negation",
                    message=f"否定词冲突：评分点与学生答案极性相反（{point_id}）",
                    severity="hard",
                )
            )

        if self._number_mismatch(pt, sa):
            hits.append(
                RuleHit(
                    point_id=point_id,
                    kind="number",
                    message=f"数字不一致：评分点中的关键数字未在学生答案中出现（{point_id}）",
                    severity="hard",
                )
            )

        if self._unit_mismatch(pt, sa):
            hits.append(
                RuleHit(
                    point_id=point_id,
                    kind="unit",
                    message=f"单位不一致：评分点单位与学生答案不匹配（{point_id}）",
                    severity="soft",
                )
            )

        if self._direction_conflict(pt, sa):
            hits.append(
                RuleHit(
                    point_id=point_id,
                    kind="direction",
                    message=f"方向词冲突：提高/降低等方向相反（{point_id}）",
                    severity="hard",
                )
            )

        if self._antonym_conflict(pt, sa):
            hits.append(
                RuleHit(
                    point_id=point_id,
                    kind="antonym",
                    message=f"反义词冲突：关键对立概念不一致（{point_id}）",
                    severity="soft",
                )
            )

        return RuleInterceptResult(hits=hits)

    @staticmethod
    def _has_negation(text: str) -> bool:
        return bool(_NEGATION_RE.search(text or ""))

    def _negation_conflict(self, point: str, student: str) -> bool:
        # 仅当评分点与学生答案在共享内容词上极性相反时触发
        pt_tokens = set(tokenize(point))
        sa_tokens = set(tokenize(student))
        content = (pt_tokens & sa_tokens) - set(tokenize("的 了 和 与 或 是 在 为 等 及"))
        if len(content) < 1:
            return False
        return self._has_negation(point) != self._has_negation(student) and (
            self._has_negation(point) or self._has_negation(student)
        )

    @staticmethod
    def _extract_numbers(text: str) -> set[str]:
        return {m.group(0).lower() for m in _NUMBER_RE.finditer(text or "")}

    def _number_mismatch(self, point: str, student: str) -> bool:
        pn = self._extract_numbers(point)
        if not pn:
            return False
        sn = self._extract_numbers(student)
        # 评分点中出现的数字，学生答案应覆盖；缺失则冲突
        return not pn.issubset(sn)

    def _unit_mismatch(self, point: str, student: str) -> bool:
        pu = {m.group(0).lower() for m in _UNIT_RE.finditer(point or "")}
        if not pu:
            return False
        su = {m.group(0).lower() for m in _UNIT_RE.finditer(student or "")}
        return not pu.intersection(su)

    def _direction_conflict(self, point: str, student: str) -> bool:
        pl, sl = (point or "").lower(), (student or "").lower()
        for a, b in _DIRECTION_PAIRS:
            if (a in pl and b in sl) or (b in pl and a in sl):
                # 两边各自只含对立一侧
                if not (a in pl and b in pl) and not (a in sl and b in sl):
                    return True
        return False

    def _antonym_conflict(self, point: str, student: str) -> bool:
        pl, sl = point or "", student or ""
        for a, b in _ANTONYM_PAIRS:
            if (a in pl and b in sl) or (b in pl and a in sl):
                if not (a in pl and b in pl) and not (a in sl and b in sl):
                    return True
        return False


class TextRerankerScorer:
    """文本题评分引擎。

    Parameters
    ----------
    pair_scorer:
        可注入的成对打分器或 (query, doc) -> float，便于测试。
    match_threshold:
        similarity >= threshold 记为 matched_evidence。
    allow_model_load:
        False 时不尝试加载 BGE/CrossEncoder（单元测试默认）。
    model_name:
        默认 BAAI/bge-reranker-base。
    """

    name = "TextRerankerScorer"

    def __init__(
        self,
        *,
        pair_scorer: PairScorer | SimilarityFn | None = None,
        match_threshold: float = _DEFAULT_MATCH_THRESHOLD,
        allow_model_load: bool = True,
        model_name: str = "BAAI/bge-reranker-base",
        point_resolver: ScoringPointResolver | None = None,
        rule_interceptor: RuleInterceptor | None = None,
    ) -> None:
        self._injected = pair_scorer
        self.match_threshold = match_threshold
        self.allow_model_load = allow_model_load
        self.model_name = model_name
        self.point_resolver = point_resolver or ScoringPointResolver()
        self.rule_interceptor = rule_interceptor or RuleInterceptor()

    def score(self, request: ScoringRequest) -> IntermediateScoreResult:
        points, warnings, force_review = self.point_resolver.resolve(request)
        student = (request.student_answer or "").strip()
        precision = request.scoring_config.score_precision

        if not points:
            return IntermediateScoreResult(
                scorer=self.name,
                scoring_mode=ScoringMode.TEXT,
                score=0.0,
                max_score=request.max_score,
                confidence=0.0,
                warnings=warnings,
                force_manual_review=True,
                metadata={"model": None, "parser": None},
            )

        scorer, backend_name = resolve_pair_scorer(
            prefer_model=self.model_name,
            injected=self._injected,
            allow_model_load=self.allow_model_load,
        )
        if backend_name == "lexical_fallback":
            warnings.append("语义模型不可用，已回退到词法相似度")

        pairs = [(student, p.text) for p in points]
        similarities = scorer.score_pairs(pairs) if student else [0.0] * len(points)

        matched: list[EvidenceItem] = []
        missed: list[EvidenceItem] = []
        total = 0.0
        weighted_conf = 0.0
        weight_sum = 0.0
        hard_conflict = False

        for point, sim in zip(points, similarities):
            sim = float(max(0.0, min(1.0, sim)))
            rule = self.rule_interceptor.check(point.text, student, point.id)
            point_score = sim * point.score
            reason_parts = [f"语义相似度 {sim:.2f}"]

            if rule.hits:
                for hit in rule.hits:
                    warnings.append(hit.message)
                    reason_parts.append(hit.message)
                if rule.hard_hits:
                    hard_conflict = True
                    point_score = 0.0
                    sim = min(sim, 0.2)
                else:
                    point_score *= 0.5
                    sim *= 0.7

            point_score = round(min(point_score, point.score), precision)
            total += point_score
            weighted_conf += sim * point.score
            weight_sum += point.score

            evidence = EvidenceItem(
                point_id=point.id,
                score=point_score,
                max_score=point.score,
                evidence=self._snippet(student, point.text) if point_score > 0 else None,
                reason="；".join(reason_parts),
                similarity=round(sim, 4),
            )
            if point_score > 0 and sim >= self.match_threshold and not rule.hard_hits:
                matched.append(evidence)
            else:
                missed.append(
                    EvidenceItem(
                        point_id=point.id,
                        score=point_score,
                        max_score=point.score,
                        reason=evidence.reason
                        if rule.hits
                        else f"未充分覆盖评分点：{point.text}",
                        similarity=round(sim, 4),
                    )
                )

        # 若评分点合计小于题目满分，按比例映射到 max_score
        points_total = sum(p.score for p in points) or request.max_score
        if abs(points_total - request.max_score) > 1e-6 and points_total > 0:
            mapped = total / points_total * request.max_score
        else:
            mapped = total
        final_score = round(min(max(mapped, 0.0), request.max_score), precision)

        confidence = weighted_conf / weight_sum if weight_sum else 0.0
        if hard_conflict:
            confidence = min(confidence, 0.4)
            force_review = True
        if not student:
            confidence = 0.0
            force_review = True
            warnings.append("学生答案为空")

        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=ScoringMode.TEXT,
            score=final_score,
            max_score=request.max_score,
            confidence=round(float(max(0.0, min(1.0, confidence))), 4),
            matched_evidence=matched,
            missed_evidence=missed,
            warnings=warnings,
            force_manual_review=force_review or hard_conflict,
            metadata={
                "model": backend_name,
                "parser": None,
                "match_threshold": self.match_threshold,
                "point_count": len(points),
            },
        )

    def __call__(self, request: ScoringRequest) -> IntermediateScoreResult:
        return self.score(request)

    @staticmethod
    def _snippet(student: str, point_text: str, max_len: int = 80) -> str | None:
        if not student:
            return None
        # 取与评分点 token 重叠最多的短句
        sentences = re.split(r"[。！？!?\n；;]+", student)
        best = student
        best_score = -1.0
        pt = set(tokenize(point_text))
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            st = set(tokenize(s))
            overlap = len(pt & st)
            if overlap > best_score:
                best_score = overlap
                best = s
        if len(best) > max_len:
            return best[: max_len - 1] + "…"
        return best


__all__ = [
    "RuleInterceptor",
    "ScoringPointResolver",
    "TextRerankerScorer",
]
