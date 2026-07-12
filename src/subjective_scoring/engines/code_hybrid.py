"""通用代码题：tree-sitter 结构分 + CrossEncoder/词法语义分 融合。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from subjective_scoring.components.normalizer import CodeNormalizer
from subjective_scoring.engines._similarity import (
    PairScorer,
    SimilarityFn,
    resolve_pair_scorer,
)
from subjective_scoring.models import (
    EvidenceItem,
    IntermediateScoreResult,
    ScoringMode,
    ScoringRequest,
)

logger = logging.getLogger(__name__)

try:
    from tree_sitter import Language, Parser
except ImportError:  # pragma: no cover
    Language = None  # type: ignore[misc, assignment]
    Parser = None  # type: ignore[misc, assignment]

# language key -> importable module name providing language()
_LANG_MODULES = {
    "python": "tree_sitter_python",
    "py": "tree_sitter_python",
    "java": "tree_sitter_java",
    "javascript": "tree_sitter_javascript",
    "js": "tree_sitter_javascript",
    "typescript": "tree_sitter_javascript",
    "ts": "tree_sitter_javascript",
    "cpp": "tree_sitter_cpp",
    "c++": "tree_sitter_cpp",
    "cc": "tree_sitter_cpp",
    "cxx": "tree_sitter_cpp",
    "c": "tree_sitter_cpp",
}

_FEATURE_NODE_TYPES: dict[str, set[str]] = {
    "loop": {
        "for_statement",
        "for_in_statement",
        "for_range_loop",
        "while_statement",
        "do_statement",
        "enhanced_for_statement",
        "for_each_statement",
    },
    "conditional": {
        "if_statement",
        "switch_statement",
        "case_statement",
        "conditional_expression",
        "ternary_expression",
    },
    "function": {
        "function_definition",
        "function_declaration",
        "method_declaration",
        "method_definition",
        "lambda",
        "lambda_expression",
        "arrow_function",
        "function",
    },
    "return": {
        "return_statement",
        "return",
    },
    "exception": {
        "try_statement",
        "catch_clause",
        "except_clause",
        "finally_clause",
        "throw_statement",
        "raise_statement",
    },
    "io": {
        "print_statement",
        "call_expression",
        "method_invocation",
    },
}

_IO_IDENTIFIERS = {
    "print",
    "println",
    "printf",
    "input",
    "scanf",
    "cout",
    "cin",
    "console",
    "log",
    "write",
    "readline",
    "readlines",
}


@dataclass
class StructureFeatures:
    flags: dict[str, bool] = field(default_factory=dict)
    call_names: set[str] = field(default_factory=set)
    identifiers: set[str] = field(default_factory=set)
    function_names: set[str] = field(default_factory=set)
    recursive_functions: set[str] = field(default_factory=set)
    node_types: set[str] = field(default_factory=set)
    parse_ok: bool = True
    error: str | None = None


class TreeSitterAstExtractor:
    """提取结构特征；解析失败时返回 parse_ok=False。"""

    def extract(self, code: str, language: str | None) -> StructureFeatures:
        if Parser is None or Language is None:
            return StructureFeatures(parse_ok=False, error="tree_sitter 未安装")
        lang_key = (language or "python").lower()
        mod_name = _LANG_MODULES.get(lang_key)
        if mod_name is None:
            return StructureFeatures(
                parse_ok=False,
                error=f"不支持的代码语言: {language}",
            )
        try:
            mod = __import__(mod_name)
            language_obj = Language(mod.language())
            parser = Parser(language_obj)
            tree = parser.parse(code.encode("utf-8"))
        except Exception as e:
            logger.exception("tree-sitter 解析失败")
            return StructureFeatures(parse_ok=False, error=str(e))

        root = tree.root_node
        if getattr(root, "has_error", False):
            return StructureFeatures(parse_ok=False, error="代码包含语法错误")
        node_types: set[str] = set()
        identifiers: set[str] = set()
        call_names: set[str] = set()
        function_defs: list[tuple[str, int, int]] = []
        call_sites: list[tuple[str, int]] = []
        source = code.encode("utf-8")

        def node_text(node: Any) -> str:
            return source[node.start_byte : node.end_byte].decode(
                "utf-8", errors="ignore"
            )

        stack = [root]
        while stack:
            node = stack.pop()
            node_types.add(node.type)
            if node.type in {"identifier", "type_identifier", "property_identifier"}:
                try:
                    identifiers.add(node_text(node))
                except Exception:
                    pass
            if node.type in _FEATURE_NODE_TYPES["function"]:
                name_node = None
                try:
                    name_node = node.child_by_field_name("name")
                except Exception:
                    pass
                if name_node is None:
                    name_node = next(
                        (child for child in node.children if child.type == "identifier"),
                        None,
                    )
                if name_node is not None:
                    name = node_text(name_node).split(".")[-1].lower()
                    if name:
                        function_defs.append((name, node.start_byte, node.end_byte))
            if node.type in {"call", "call_expression", "method_invocation"}:
                for child in node.children:
                    if child.type in {
                        "identifier",
                        "attribute",
                        "member_expression",
                        "field_access",
                    }:
                        try:
                            name = node_text(child).split(".")[-1].lower()
                            call_names.add(name)
                            call_sites.append((name, node.start_byte))
                        except Exception:
                            pass
                        break
            stack.extend(reversed(node.children))

        flags = {
            name: bool(node_types & types)
            for name, types in _FEATURE_NODE_TYPES.items()
        }
        if call_names & _IO_IDENTIFIERS or identifiers & _IO_IDENTIFIERS:
            flags["io"] = True

        function_names = {name for name, _, _ in function_defs}
        recursive_functions = {
            name
            for name, start, end in function_defs
            if any(call_name == name and start < position < end for call_name, position in call_sites)
        }
        flags["recursion"] = bool(recursive_functions)

        return StructureFeatures(
            flags=flags,
            call_names={c.lower() for c in call_names},
            identifiers={i.lower() for i in identifiers},
            function_names=function_names,
            recursive_functions=recursive_functions,
            node_types=node_types,
            parse_ok=True,
        )


class StructureScoreCalculator:
    """比较学生与参考代码的结构特征。"""

    FEATURE_KEYS = (
        "loop",
        "conditional",
        "function",
        "return",
        "exception",
        "io",
        "recursion",
    )

    def score(
        self,
        reference: StructureFeatures,
        student: StructureFeatures,
    ) -> tuple[float, list[EvidenceItem], list[EvidenceItem], list[str]]:
        warnings: list[str] = []
        matched: list[EvidenceItem] = []
        missed: list[EvidenceItem] = []

        if not reference.parse_ok or not student.parse_ok:
            if not reference.parse_ok:
                warnings.append(f"参考代码 AST 提取失败: {reference.error}")
            if not student.parse_ok:
                warnings.append(f"学生代码 AST 提取失败: {student.error}")
            return 0.0, matched, missed, warnings

        active = [k for k in self.FEATURE_KEYS if reference.flags.get(k)]
        if not active:
            sim = self._jaccard(reference.node_types, student.node_types)
            item = EvidenceItem(
                point_id="code.structure.nodes",
                score=sim,
                max_score=1.0,
                reason=f"节点类型 Jaccard={sim:.2f}",
                similarity=round(sim, 4),
            )
            if sim >= 0.5:
                matched.append(item)
            else:
                missed.append(item)
            return sim, matched, missed, warnings

        hits = 0
        per = 1.0 / len(active)
        for key in active:
            ok = bool(student.flags.get(key))
            item = EvidenceItem(
                point_id=f"code.structure.{key}",
                score=per if ok else 0.0,
                max_score=per,
                reason=f"结构特征 {key}: 参考需要，学生{'具备' if ok else '缺失'}",
                similarity=1.0 if ok else 0.0,
            )
            if ok:
                hits += 1
                matched.append(item)
            else:
                missed.append(item)

        ref_calls = reference.call_names - {"", "print"}
        if ref_calls:
            cover = len(ref_calls & student.call_names) / len(ref_calls)
            api_item = EvidenceItem(
                point_id="code.structure.api",
                score=cover * 0.15,
                max_score=0.15,
                reason=f"API/调用覆盖 {cover:.0%}: ref={sorted(ref_calls)[:8]}",
                similarity=round(cover, 4),
            )
            base = hits / len(active)
            overall = 0.85 * base + 0.15 * cover
            if cover >= 0.5:
                matched.append(api_item)
            else:
                missed.append(api_item)
        else:
            overall = hits / len(active)

        return float(max(0.0, min(1.0, overall))), matched, missed, warnings

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


class CodeHybridScorer:
    """代码混合评分：semantic * w_s + structure * w_t。"""

    name = "CodeHybridScorer"
    DEFAULT_MODEL = "BAAI/bge-reranker-base"
    _POINT_FEATURE_PATTERNS = {
        "recursion": re.compile(r"递归|自身调用|self.?call", re.IGNORECASE),
        "loop": re.compile(r"循环|遍历|for|while", re.IGNORECASE),
        "conditional": re.compile(r"条件|判断|分支|if|switch", re.IGNORECASE),
        "function": re.compile(r"函数|方法|function|method", re.IGNORECASE),
        "return": re.compile(r"返回|return", re.IGNORECASE),
        "exception": re.compile(r"异常|错误处理|try|catch|except", re.IGNORECASE),
        "io": re.compile(r"输入|输出|读写|print|input|I/O", re.IGNORECASE),
    }

    def __init__(
        self,
        *,
        pair_scorer: PairScorer | SimilarityFn | None = None,
        allow_model_load: bool = True,
        model_name: str = DEFAULT_MODEL,
        strip_comments: bool = True,
        conflict_gap: float = 0.45,
    ) -> None:
        self._injected = pair_scorer
        self.allow_model_load = allow_model_load
        self.model_name = model_name
        self.normalizer = CodeNormalizer(strip_comments=strip_comments)
        self.extractor = TreeSitterAstExtractor()
        self.structure_calculator = StructureScoreCalculator()
        self.conflict_gap = conflict_gap

    def score(self, request: ScoringRequest) -> IntermediateScoreResult:
        precision = request.scoring_config.score_precision
        weights = request.scoring_config.code_score_weights
        lang = request.code_language or "python"

        ref = self.normalizer.normalize(request.reference_answer, lang)
        stu = self.normalizer.normalize(request.student_answer, lang)
        warnings: list[str] = []
        force_review = False
        applied_caps: list[str] = []
        warnings.append("代码题结果为静态估分，未执行学生代码")

        if not ref:
            warnings.append("参考代码为空")
            force_review = True
        if not stu:
            warnings.append("学生代码为空")
            force_review = True

        if ref:
            ref_feat = self.extractor.extract(ref, lang)
        else:
            ref_feat = StructureFeatures(parse_ok=False, error="empty")
        if stu:
            stu_feat = self.extractor.extract(stu, lang)
        else:
            stu_feat = StructureFeatures(parse_ok=False, error="empty")

        structure_sim, s_matched, s_missed, s_warnings = self.structure_calculator.score(
            ref_feat, stu_feat
        )
        warnings.extend(s_warnings)
        if not ref_feat.parse_ok or not stu_feat.parse_ok:
            force_review = True

        scorer, backend = resolve_pair_scorer(
            prefer_model=self.model_name,
            injected=self._injected,
            allow_model_load=self.allow_model_load,
        )
        if backend == "lexical_fallback":
            warnings.append("代码语义模型不可用，已回退到词法相似度")

        if ref and stu:
            semantic_sim = float(scorer.score_pairs([(stu, ref)])[0])
        else:
            semantic_sim = 0.0

        final_sim = weights.semantic * semantic_sim + weights.structure * structure_sim
        final_score = round(
            min(max(final_sim * request.max_score, 0.0), request.max_score),
            precision,
        )

        if abs(semantic_sim - structure_sim) >= self.conflict_gap and ref and stu:
            warnings.append(
                f"语义分 ({semantic_sim:.2f}) 与结构分 ({structure_sim:.2f}) 差异较大"
            )
            force_review = True

        structure_conflict = semantic_sim >= 0.75 and structure_sim < 0.30
        if structure_conflict:
            warnings.append("语义相似度高但结构明显冲突，得分上限为满分的 50%")
            force_review = True
            applied_caps.append("semantic_structure_conflict:0.5")

        confidence = final_sim
        if force_review:
            confidence = min(confidence, 0.55)
        confidence = float(max(0.0, min(1.0, confidence)))

        sem_score = round(semantic_sim * request.max_score * weights.semantic, precision)
        sem_max = round(request.max_score * weights.semantic, precision)
        semantic_item = EvidenceItem(
            point_id="code.semantic",
            score=sem_score,
            max_score=sem_max,
            reason=f"语义相似度 {semantic_sim:.2f}",
            similarity=round(semantic_sim, 4),
        )

        matched: list[EvidenceItem] = []
        missed: list[EvidenceItem] = []
        if semantic_sim >= 0.5:
            matched.append(semantic_item)
        else:
            missed.append(
                EvidenceItem(
                    point_id="code.semantic",
                    score=sem_score,
                    max_score=sem_max,
                    reason=f"语义相似度偏低 {semantic_sim:.2f}",
                    similarity=round(semantic_sim, 4),
                )
            )

        def _scale(items: list[EvidenceItem]) -> list[EvidenceItem]:
            out = []
            for it in items:
                out.append(
                    EvidenceItem(
                        point_id=it.point_id,
                        score=round(
                            it.score * request.max_score * weights.structure, precision
                        ),
                        max_score=round(
                            it.max_score * request.max_score * weights.structure,
                            precision,
                        ),
                        evidence=it.evidence,
                        reason=it.reason,
                        similarity=it.similarity,
                    )
                )
            return out

        matched.extend(_scale(s_matched))
        missed.extend(_scale(s_missed))

        point_diagnostics: list[dict[str, object]] = []
        if request.scoring_points:
            matched = []
            missed = []
            point_total = sum(point.score for point in request.scoring_points) or request.max_score
            awarded = 0.0
            for point in request.scoring_points:
                feature = self._feature_for_point(point.text)
                if feature is None:
                    point_similarity = final_sim
                    reason = "该行为评分点无法由静态分析确认，按混合相似度估分，待执行测试验证"
                else:
                    point_similarity = (
                        1.0 if stu_feat.parse_ok and stu_feat.flags.get(feature) else 0.0
                    )
                    reason = (
                        f"静态结构特征 {feature}: "
                        f"学生代码{'具备' if point_similarity else '缺失'}"
                    )
                point_score = round(point.score * point_similarity, precision)
                awarded += point_score
                item = EvidenceItem(
                    point_id=point.id,
                    score=point_score,
                    max_score=point.score,
                    reason=reason,
                    similarity=round(float(point_similarity), 4),
                )
                if point_similarity >= 0.5:
                    matched.append(item)
                else:
                    missed.append(item)
                    if point.required and feature is not None:
                        force_review = True
                        warnings.append(f"必答代码评分点缺失: {point.id}")
                point_diagnostics.append(
                    {
                        "point_id": point.id,
                        "feature": feature,
                        "statically_verifiable": feature is not None,
                        "similarity": round(float(point_similarity), 4),
                    }
                )
            final_score = round(awarded / point_total * request.max_score, precision)

        if not stu_feat.parse_ok:
            cap = round(request.max_score * 0.20, precision)
            final_score = min(final_score, cap)
            applied_caps.append("student_parse_failure:0.2")
        if structure_conflict:
            cap = round(request.max_score * 0.50, precision)
            final_score = min(final_score, cap)
        if force_review:
            confidence = min(confidence, 0.55)

        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=ScoringMode.CODE,
            score=final_score,
            max_score=request.max_score,
            confidence=round(confidence, 4),
            matched_evidence=matched,
            missed_evidence=missed,
            warnings=warnings,
            force_manual_review=force_review,
            metadata={
                "model": backend,
                "parser": "tree-sitter",
                "language": lang,
                "semantic_similarity": round(semantic_sim, 4),
                "structure_similarity": round(structure_sim, 4),
                "weights": {
                    "semantic": weights.semantic,
                    "structure": weights.structure,
                },
                "assessment_type": "static_estimate",
                "applied_caps": applied_caps,
                "point_diagnostics": point_diagnostics,
            },
        )

    @classmethod
    def _feature_for_point(cls, text: str) -> str | None:
        for feature, pattern in cls._POINT_FEATURE_PATTERNS.items():
            if pattern.search(text or ""):
                return feature
        return None

    def __call__(self, request: ScoringRequest) -> IntermediateScoreResult:
        return self.score(request)


__all__ = [
    "CodeHybridScorer",
    "CodeNormalizer",
    "StructureScoreCalculator",
    "TreeSitterAstExtractor",
]
