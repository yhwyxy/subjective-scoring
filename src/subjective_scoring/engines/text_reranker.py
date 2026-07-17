"""文本主观题：结构化评分点 + 语义匹配 + 规则拦截。"""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import RLock

from subjective_scoring.engines.calibration import (
    ScoreCalibrator,
    default_calibrator_for_backend,
)
from subjective_scoring.engines._similarity import (
    DocumentBatchScorer,
    PairScorer,
    SimilarityFn,
    lexical_similarity,
    resolve_pair_scorer,
    tokenize,
)
from subjective_scoring.models import (
    EvidenceItem,
    IntermediateScoreResult,
    PointConflictPolicy,
    PointRelation,
    ScoringMode,
    ScoringPoint,
    ScoringRequest,
)

_DEFAULT_MATCH_THRESHOLD = 0.55
_EXACT_MATCH_WHITESPACE_RE = re.compile(r"\s+")
_EXACT_MATCH_PUNCTUATION = str.maketrans(
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


def _normalize_exact_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.casefold().translate(_EXACT_MATCH_PUNCTUATION)
    return _EXACT_MATCH_WHITESPACE_RE.sub(" ", normalized).strip()
_ANTONYM_PAIRS = (
    ("正确", "错误"),
    ("成功", "失败"),
    ("同步", "异步"),
    ("静态", "动态"),
    ("开启", "关闭"),
    ("允许", "禁止"),
)
_CLAUSE_RE = re.compile(r"[。！？!?；;，,\n]+")
_STOPWORDS = set(tokenize("的 了 和 与 或 是 在 为 等 及 使用 通过 可以 应该 保持"))
_DEFAULT_POLARITY_RULES = (
    (
        "stateless",
        r"(无状态|不保存.{0,10}(会话|客户端状态)|不依赖.{0,10}(历史|上下文|会话))",
        r"(有状态|保存.{0,8}(客户端)?会话状态|依赖.{0,8}(历史|上下文|会话))",
    ),
)


@dataclass
class ResolvedScoringPoint:
    id: str
    text: str
    score: float
    required: bool = False
    critical: bool = False
    conflict_policy: PointConflictPolicy = PointConflictPolicy.POINT_ZERO
    conflict_score_cap_ratio: float = 0.4
    synthetic: bool = False


@dataclass(frozen=True)
class PointEvidenceMatch:
    evidence: str
    raw_similarity: float
    whole_answer_similarity: float
    candidate_count: int


@dataclass
class RuleHit:
    point_id: str
    kind: str
    message: str
    severity: str = "hard"
    evidence: str | None = None
    confidence: float = 1.0


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
                    critical=p.critical,
                    conflict_policy=p.conflict_policy,
                    conflict_score_cap_ratio=p.conflict_score_cap_ratio,
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

    def __init__(
        self,
        polarity_rules: Sequence[tuple[str, str, str]] | None = None,
    ) -> None:
        rules = [*_DEFAULT_POLARITY_RULES, *(polarity_rules or ())]
        self.polarity_rules = [
            (name, re.compile(positive, re.IGNORECASE), re.compile(negative, re.IGNORECASE))
            for name, positive, negative in rules
        ]

    def check(self, point_text: str, student_answer: str, point_id: str) -> RuleInterceptResult:
        hits: list[RuleHit] = []
        pt = point_text or ""
        sa = student_answer or ""

        negation_conflict, negation_evidence, negation_confidence = (
            self._negation_conflict(pt, sa)
        )
        if negation_conflict:
            hits.append(
                RuleHit(
                    point_id=point_id,
                    kind="negation",
                    message=f"否定词冲突：评分点与学生答案极性相反（{point_id}）",
                    severity="hard",
                    evidence=negation_evidence,
                    confidence=negation_confidence,
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

    def _negation_conflict(self, point: str, student: str) -> tuple[bool, str | None, float]:
        """只在与评分点局部相关的语句中判断极性。"""

        point_polarity = self._concept_polarity(point)
        point_tokens = self._content_tokens(point)
        candidates: list[tuple[int, str]] = []
        for clause in _CLAUSE_RE.split(student or ""):
            clause = clause.strip()
            if not clause:
                continue
            overlap = len(point_tokens & self._content_tokens(clause))
            if overlap:
                candidates.append((overlap, clause))

        if not candidates:
            return False, None, 0.0

        for overlap, clause in sorted(candidates, reverse=True):
            clause_polarity = self._concept_polarity(clause)
            if point_polarity and clause_polarity and point_polarity[0] == clause_polarity[0]:
                if point_polarity[1] != clause_polarity[1]:
                    return True, clause, 1.0
                continue

            point_negated = self._has_negation(point)
            clause_negated = self._has_negation(clause)
            if point_negated == clause_negated:
                continue
            # 评分点本身为否定命题时，证据片段没有重复否定词只代表信息不完整，
            # 不能据此推断学生明确给出了相反结论。领域极性规则已在上方处理
            # “无状态/有状态”等显式对立。
            if point_negated and not clause_negated:
                continue
            # 单个宽泛 token 容易把别处的否定词错误关联到评分点。
            if overlap >= 2 or any(
                len(token) >= 4 and token in clause for token in point_tokens
            ):
                confidence = min(1.0, 0.55 + overlap * 0.15)
                return True, clause, confidence
        return False, None, 0.0

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {token for token in tokenize(text or "") if token not in _STOPWORDS}

    def _concept_polarity(self, text: str) -> tuple[str, int] | None:
        value = text or ""
        for name, positive, negative in self.polarity_rules:
            if positive.search(value):
                return name, 1
            if negative.search(value):
                return name, -1
        return None

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
        # “行/列/次”等汉字在普通词语中很常见；只有评分点明确包含数字时，
        # 才将其解释为需要严格核对的计量单位。
        if not self._extract_numbers(point):
            return False
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
        可选的支持阈值覆盖；未传入时使用请求中的 text_relation_thresholds.support。
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
        match_threshold: float | None = None,
        allow_model_load: bool = True,
        model_name: str = "BAAI/bge-reranker-base",
        point_resolver: ScoringPointResolver | None = None,
        rule_interceptor: RuleInterceptor | None = None,
        calibrator: ScoreCalibrator | None = None,
        reference_cache_size: int = 256,
    ) -> None:
        if reference_cache_size < 0:
            raise ValueError("reference_cache_size must not be negative")
        self._injected = pair_scorer
        self.match_threshold = match_threshold
        self.allow_model_load = allow_model_load
        self.model_name = model_name
        self.point_resolver = point_resolver or ScoringPointResolver()
        self.rule_interceptor = rule_interceptor or RuleInterceptor()
        self.calibrator = calibrator
        self.reference_cache_size = reference_cache_size
        self._reference_match_cache: OrderedDict[
            tuple[object, ...], tuple[PointEvidenceMatch, ...]
        ] = OrderedDict()
        self._reference_cache_lock = RLock()

    def _support_threshold_for_backend(
        self,
        request: ScoringRequest,
        backend_name: str,
    ) -> float:
        options = request.scoring_config.text_relation_thresholds
        if self.match_threshold is not None:
            return self.match_threshold
        # 显式配置 support 时维持 v0.1.3 的全后端覆盖语义。
        if "support" in options.model_fields_set:
            return options.support
        if backend_name == "lexical_fallback":
            return options.lexical_fallback_support
        if backend_name.startswith("cohere:"):
            return options.remote_reranker_support
        return options.local_cross_encoder_support

    @staticmethod
    def _evidence_candidates(answer: str) -> list[str]:
        value = (answer or "").strip()
        if not value:
            return []
        clauses = [
            clause.strip()
            for clause in _CLAUSE_RE.split(value)
            if clause.strip()
        ]
        windows = [
            "，".join(clauses[index : index + size])
            for size in (2, 3)
            for index in range(max(0, len(clauses) - size + 1))
        ]
        candidates: list[str] = []
        for candidate in [*clauses, *windows, value]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _best_evidence_match(
        answer: str,
        candidates: list[str],
        scores: Sequence[float],
    ) -> PointEvidenceMatch:
        if not candidates:
            return PointEvidenceMatch("", 0.0, 0.0, 0)
        values = [float(score) for score in scores[: len(candidates)]]
        if len(values) < len(candidates):
            values.extend([0.0] * (len(candidates) - len(values)))
        best_index = max(range(len(candidates)), key=values.__getitem__)
        whole_index = candidates.index((answer or "").strip())
        return PointEvidenceMatch(
            evidence=candidates[best_index],
            raw_similarity=values[best_index],
            whole_answer_similarity=values[whole_index],
            candidate_count=len(candidates),
        )

    def _score_evidence_sets(
        self,
        scorer: PairScorer,
        answers: Sequence[str],
        points: list[ResolvedScoringPoint],
    ) -> list[list[PointEvidenceMatch]]:
        candidate_sets = [self._evidence_candidates(answer) for answer in answers]
        matches = [
            [PointEvidenceMatch("", 0.0, 0.0, 0) for _point in points]
            for _answer in answers
        ]

        if isinstance(scorer, DocumentBatchScorer):
            for point_index, point in enumerate(points):
                documents = [
                    candidate
                    for candidates in candidate_sets
                    for candidate in candidates
                ]
                if not documents:
                    continue
                document_scores = scorer.score_documents(point.text, documents)
                offset = 0
                for answer_index, candidates in enumerate(candidate_sets):
                    end = offset + len(candidates)
                    matches[answer_index][point_index] = self._best_evidence_match(
                        answers[answer_index],
                        candidates,
                        document_scores[offset:end],
                    )
                    offset = end
            return matches

        for answer_index, (answer, candidates) in enumerate(
            zip(answers, candidate_sets)
        ):
            if not candidates:
                continue
            pairs = [
                (candidate, point.text)
                for point in points
                for candidate in candidates
            ]
            scores = [float(score) for score in scorer.score_pairs(pairs)]
            candidate_count = len(candidates)
            for point_index, _point in enumerate(points):
                start = point_index * candidate_count
                end = start + candidate_count
                matches[answer_index][point_index] = self._best_evidence_match(
                    answer,
                    candidates,
                    scores[start:end],
                )
        return matches

    def _score_evidence_matches(
        self,
        scorer: PairScorer,
        answer: str,
        points: list[ResolvedScoringPoint],
    ) -> list[PointEvidenceMatch]:
        return self._score_evidence_sets(scorer, [answer], points)[0]

    @staticmethod
    def _reference_cache_key(
        backend_name: str,
        reference: str,
        points: list[ResolvedScoringPoint],
    ) -> tuple[object, ...]:
        return (
            backend_name,
            reference,
            tuple(
                (point.id, point.text, point.required, point.critical)
                for point in points
            ),
        )

    def _get_cached_reference_matches(
        self,
        key: tuple[object, ...],
    ) -> list[PointEvidenceMatch] | None:
        if self.reference_cache_size == 0:
            return None
        with self._reference_cache_lock:
            cached = self._reference_match_cache.get(key)
            if cached is None:
                return None
            self._reference_match_cache.move_to_end(key)
            return list(cached)

    def _cache_reference_matches(
        self,
        key: tuple[object, ...],
        matches: list[PointEvidenceMatch],
    ) -> None:
        if self.reference_cache_size == 0:
            return
        with self._reference_cache_lock:
            self._reference_match_cache[key] = tuple(matches)
            self._reference_match_cache.move_to_end(key)
            while len(self._reference_match_cache) > self.reference_cache_size:
                self._reference_match_cache.popitem(last=False)

    def score(self, request: ScoringRequest) -> IntermediateScoreResult:
        raw_student = request.student_answer or ""
        reference = request.reference_answer or ""
        if not raw_student.strip():
            return IntermediateScoreResult(
                scorer=self.name,
                scoring_mode=ScoringMode.TEXT,
                score=0.0,
                max_score=request.max_score,
                confidence=1.0,
                metadata={
                    "model": None,
                    "parser": None,
                    "decision": "auto_zero",
                    "decision_reason": "blank_answer",
                    "deterministic": True,
                    "point_count": len(request.scoring_points),
                    "relation_counts": {
                        "supported": 0,
                        "contradicted": 0,
                        "unknown": 0,
                    },
                },
            )

        normalized_reference = _normalize_exact_match_text(reference)
        if (
            normalized_reference
            and _normalize_exact_match_text(raw_student) == normalized_reference
        ):
            return IntermediateScoreResult(
                scorer=self.name,
                scoring_mode=ScoringMode.TEXT,
                score=request.max_score,
                max_score=request.max_score,
                confidence=1.0,
                metadata={
                    "model": None,
                    "parser": None,
                    "decision": "auto_score",
                    "decision_reason": "exact_reference_match",
                    "deterministic": True,
                    "point_count": len(request.scoring_points),
                    "relation_counts": {
                        "supported": len(request.scoring_points),
                        "contradicted": 0,
                        "unknown": 0,
                    },
                },
            )

        points, warnings, force_review = self.point_resolver.resolve(request)
        student = raw_student.strip()
        precision = request.scoring_config.score_precision
        relation_options = request.scoring_config.text_relation_thresholds
        conflict_threshold = relation_options.conflict

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

        support_threshold = self._support_threshold_for_backend(request, backend_name)
        calibrator = self.calibrator or default_calibrator_for_backend(backend_name)
        reference = reference.strip()
        reference_matches: list[PointEvidenceMatch] | None = None
        reference_cache_hit = False
        if relation_options.validate_reference_points and reference:
            reference_cache_key = self._reference_cache_key(
                backend_name,
                reference,
                points,
            )
            reference_matches = self._get_cached_reference_matches(
                reference_cache_key
            )
            if reference_matches is not None:
                reference_cache_hit = True
                student_matches = self._score_evidence_matches(
                    scorer,
                    student,
                    points,
                )
            else:
                student_matches, reference_matches = self._score_evidence_sets(
                    scorer,
                    [student, reference],
                    points,
                )
                self._cache_reference_matches(
                    reference_cache_key,
                    reference_matches,
                )
        else:
            student_matches = self._score_evidence_matches(
                scorer,
                student,
                points,
            )

        matched: list[EvidenceItem] = []
        missed: list[EvidenceItem] = []
        total = 0.0
        weighted_conf = 0.0
        weight_sum = 0.0
        hard_conflict = False
        supported_count = 0
        contradicted_count = 0
        unknown_count = 0
        required_unknown = False
        total_conflict_zero = False
        total_conflict_cap_ratio = 1.0
        applied_caps: list[str] = []
        decision_reason: str | None = None
        provisional_total = 0.0
        rubric_diagnostics: list[dict[str, object]] = []
        point_diagnostics: list[dict[str, object]] = []

        if reference_matches is not None:
            for point, reference_match in zip(points, reference_matches):
                if not (point.required or point.critical):
                    continue
                reference_raw = float(
                    max(0.0, min(1.0, reference_match.raw_similarity))
                )
                reference_sim = float(
                    max(0.0, min(1.0, calibrator.calibrate(reference_raw)))
                )
                reference_rule = self.rule_interceptor.check(
                    point.text,
                    reference_match.evidence,
                    point.id,
                )
                reference_conflict = any(
                    hit.confidence >= conflict_threshold
                    for hit in reference_rule.hard_hits
                )
                reference_supported = (
                    reference_sim >= support_threshold and not reference_conflict
                )
                rubric_diagnostics.append(
                    {
                        "point_id": point.id,
                        "supported": reference_supported,
                        "raw_similarity": round(reference_raw, 4),
                        "calibrated_similarity": round(reference_sim, 4),
                        "evidence": reference_match.evidence,
                        "hard_conflict": reference_conflict,
                    }
                )
                if not reference_supported:
                    force_review = True
                    decision_reason = "rubric_self_check_failed"
                    warnings.append(
                        f"评分表自检失败：标准答案未可靠支持必答或关键评分点 {point.id}"
                    )

        for point, evidence_match in zip(points, student_matches):
            raw_sim = float(
                max(0.0, min(1.0, evidence_match.raw_similarity))
            )
            calibrated_sim = float(
                max(0.0, min(1.0, calibrator.calibrate(raw_sim)))
            )
            rule = self.rule_interceptor.check(
                point.text,
                evidence_match.evidence,
                point.id,
            )
            confident_hard_hits = [
                hit for hit in rule.hard_hits if hit.confidence >= conflict_threshold
            ]
            uncertain_hard_hits = [
                hit for hit in rule.hard_hits if hit.confidence < conflict_threshold
            ]

            if confident_hard_hits:
                relation = PointRelation.CONTRADICTED
                relation_confidence = max(hit.confidence for hit in confident_hard_hits)
            elif uncertain_hard_hits:
                relation = PointRelation.UNKNOWN
                relation_confidence = max(hit.confidence for hit in uncertain_hard_hits)
            elif calibrated_sim >= support_threshold:
                relation = PointRelation.SUPPORTED
                relation_confidence = calibrated_sim
            else:
                relation = PointRelation.UNKNOWN
                relation_confidence = max(0.0, 1.0 - calibrated_sim)

            point_score = 0.0
            point_provisional_score = 0.0
            adjusted_confidence = relation_confidence
            reason_parts = [
                f"原始相关度 {raw_sim:.2f}",
                f"校准覆盖度 {calibrated_sim:.2f}",
                f"关系 {relation.value}",
            ]

            if rule.hits:
                for hit in rule.hits:
                    warnings.append(hit.message)
                    reason_parts.append(hit.message)

            if relation == PointRelation.SUPPORTED:
                supported_count += 1
                point_score = calibrated_sim * point.score
                soft_hits = [hit for hit in rule.hits if hit.severity != "hard"]
                if soft_hits:
                    point_score *= 0.65
                    adjusted_confidence = calibrated_sim * 0.8
                point_provisional_score = point_score
            elif relation == PointRelation.CONTRADICTED:
                contradicted_count += 1
                hard_conflict = True
                force_review = True
                adjusted_confidence = min(relation_confidence, 0.4)
                if point.conflict_policy == PointConflictPolicy.ZERO_TOTAL:
                    total_conflict_zero = True
                elif point.conflict_policy == PointConflictPolicy.CAP_TOTAL:
                    total_conflict_cap_ratio = min(
                        total_conflict_cap_ratio,
                        point.conflict_score_cap_ratio,
                    )
            else:
                unknown_count += 1
                point_provisional_score = calibrated_sim * point.score
                if uncertain_hard_hits:
                    point_provisional_score *= 0.35
                elif any(hit.severity != "hard" for hit in rule.hits):
                    point_provisional_score *= 0.65
                if point.required or point.critical:
                    required_unknown = True
                # 未拆分全文仅用于兼容兜底估分，始终要求人工复核。
                if point.synthetic and not relation_options.apply_gate_to_synthetic_reference:
                    force_review = True
                    reason_parts.append("全文兜底点未应用原子评分门槛，仅作待复核估分")

            point_diagnostics.append(
                {
                    "point_id": point.id,
                    "raw_similarity": round(raw_sim, 4),
                    "calibrated_similarity": round(calibrated_sim, 4),
                    "whole_answer_similarity": round(
                        max(0.0, min(1.0, evidence_match.whole_answer_similarity)),
                        4,
                    ),
                    "evidence": evidence_match.evidence,
                    "candidate_count": evidence_match.candidate_count,
                    "adjusted_confidence": round(adjusted_confidence, 4),
                    "relation": relation.value,
                    "relation_confidence": round(relation_confidence, 4),
                    "synthetic": point.synthetic,
                    "rule_hits": [
                        {
                            "kind": hit.kind,
                            "severity": hit.severity,
                            "confidence": round(hit.confidence, 4),
                            "evidence": hit.evidence,
                        }
                        for hit in rule.hits
                    ],
                }
            )

            point_score = round(min(point_score, point.score), precision)
            point_provisional_score = round(
                min(point_provisional_score, point.score),
                precision,
            )
            total += point_score
            provisional_total += point_provisional_score
            weighted_conf += relation_confidence * point.score
            weight_sum += point.score

            evidence = EvidenceItem(
                point_id=point.id,
                score=point_score,
                provisional_score=point_provisional_score,
                max_score=point.score,
                evidence=evidence_match.evidence or None,
                reason="；".join(reason_parts),
                similarity=round(calibrated_sim, 4),
                relation=relation,
                relation_confidence=round(relation_confidence, 4),
            )
            if relation == PointRelation.SUPPORTED:
                matched.append(evidence)
            else:
                missed.append(
                    EvidenceItem(
                        point_id=point.id,
                        score=point_score,
                        provisional_score=point_provisional_score,
                        max_score=point.score,
                        reason=evidence.reason
                        if rule.hits
                        else f"未充分覆盖评分点：{point.text}",
                        similarity=round(calibrated_sim, 4),
                        evidence=evidence_match.evidence or None,
                        relation=relation,
                        relation_confidence=round(relation_confidence, 4),
                    )
                )

        # 若评分点合计小于题目满分，按比例映射到 max_score
        points_total = sum(p.score for p in points) or request.max_score
        if abs(points_total - request.max_score) > 1e-6 and points_total > 0:
            mapped = total / points_total * request.max_score
            mapped_provisional = provisional_total / points_total * request.max_score
        else:
            mapped = total
            mapped_provisional = provisional_total
        final_score = round(min(max(mapped, 0.0), request.max_score), precision)
        provisional_score = round(
            min(max(mapped_provisional, 0.0), request.max_score),
            precision,
        )

        if total_conflict_zero:
            final_score = 0.0
            provisional_score = 0.0
            applied_caps.append("critical_conflict:0.0")
            decision_reason = "critical_point_conflict_zero_total"
        elif total_conflict_cap_ratio < 1.0:
            cap = round(request.max_score * total_conflict_cap_ratio, precision)
            final_score = min(final_score, cap)
            provisional_score = min(provisional_score, cap)
            applied_caps.append(f"critical_conflict:{total_conflict_cap_ratio}")
            decision_reason = "critical_point_conflict_cap"

        confidence = weighted_conf / weight_sum if weight_sum else 0.0
        if hard_conflict:
            confidence = min(confidence, 0.4)
            force_review = True
            decision_reason = decision_reason or "hard_conflict"
        if required_unknown and relation_options.required_unknown_requires_review:
            force_review = True
            confidence = min(confidence, 0.55)
            warnings.append("必答或关键评分点关系不确定，拒绝自动定分")
            decision_reason = decision_reason or "required_point_unknown"
        if (
            supported_count == 0
            and unknown_count > 0
            and relation_options.reject_when_no_supported
            and any(not point.synthetic for point in points)
        ):
            final_score = 0.0
            force_review = True
            confidence = min(confidence, 0.55)
            warnings.append("没有评分点被可靠支持，存在不确定关系，拒绝自动定分")
            decision_reason = decision_reason or "no_supported_points_uncertain"
        elif supported_count == 0 and contradicted_count == len(points):
            final_score = 0.0
            force_review = True
            decision_reason = decision_reason or "all_points_contradicted"
        if decision_reason is None:
            if final_score <= 0.0:
                decision_reason = "no_supported_points"
            else:
                decision_reason = "supported_points"

        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=ScoringMode.TEXT,
            score=final_score,
            provisional_score=provisional_score,
            max_score=request.max_score,
            confidence=round(float(max(0.0, min(1.0, confidence))), 4),
            matched_evidence=matched,
            missed_evidence=missed,
            warnings=warnings,
            force_manual_review=force_review,
            metadata={
                "model": backend_name,
                "parser": None,
                "match_threshold": support_threshold,
                "support_threshold": support_threshold,
                "conflict_threshold": conflict_threshold,
                "point_count": len(points),
                "relation_counts": {
                    "supported": supported_count,
                    "contradicted": contradicted_count,
                    "unknown": unknown_count,
                },
                "decision": (
                    "manual_review"
                    if force_review
                    else "auto_zero"
                    if final_score <= 0.0
                    else "auto_score"
                ),
                "decision_reason": decision_reason,
                "applied_caps": applied_caps,
                "rubric_validation": rubric_diagnostics,
                "reference_cache_hit": reference_cache_hit,
                "evidence_batch_mode": (
                    "query_documents"
                    if isinstance(scorer, DocumentBatchScorer)
                    else "pair_matrix"
                ),
                "calibrator": getattr(calibrator, "name", type(calibrator).__name__),
                "point_diagnostics": point_diagnostics,
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
    "ScoreCalibrator",
    "ScoringPointResolver",
    "TextRerankerScorer",
]
