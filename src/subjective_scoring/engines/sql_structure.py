"""SQL 题：sqlglot AST 结构比较评分（不走模型）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from subjective_scoring.components.normalizer import SQLNormalizer
from subjective_scoring.models import (
    EvidenceItem,
    IntermediateScoreResult,
    ScoringMode,
    ScoringRequest,
)

try:
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import ParseError
except ImportError:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]
    ParseError = Exception  # type: ignore[misc, assignment]


# 结构维度及默认权重（合计 1.0）
_DEFAULT_WEIGHTS: dict[str, float] = {
    "select": 0.18,
    "from": 0.14,
    "join": 0.12,
    "where": 0.18,
    "group_by": 0.08,
    "having": 0.06,
    "order_by": 0.08,
    "limit": 0.04,
    "aggregates": 0.06,
    "subquery": 0.04,
    "operators": 0.02,
}


@dataclass
class DimensionScore:
    name: str
    weight: float
    similarity: float
    detail: str
    matched: bool

    @property
    def weighted(self) -> float:
        return self.weight * self.similarity


@dataclass
class SQLComparisonResult:
    dimensions: list[DimensionScore]
    warnings: list[str]
    force_review: bool
    reference_type: str | None = None
    student_type: str | None = None
    active_dimensions: list[str] = field(default_factory=list)
    rejection_reason: str | None = None


class SQLAstComparator:
    """基于 sqlglot AST 的结构维度比较。"""

    def compare(self, ref_sql: str, stu_sql: str) -> tuple[list[DimensionScore], list[str], bool]:
        result = self.compare_detailed(ref_sql, stu_sql)
        return result.dimensions, result.warnings, result.force_review

    def compare_detailed(self, ref_sql: str, stu_sql: str) -> SQLComparisonResult:
        warnings: list[str] = []

        if sqlglot is None:
            warnings.append("sqlglot 未安装，SQL 结构评分不可用")
            return SQLComparisonResult([], warnings, True)

        ref_ast, ref_err = self._parse(ref_sql)
        stu_ast, stu_err = self._parse(stu_sql)

        if ref_err:
            warnings.append(f"标准答案 SQL 解析失败: {ref_err}")
        if stu_err:
            warnings.append(f"学生答案 SQL 解析失败: {stu_err}")

        if ref_ast is None or stu_ast is None:
            return SQLComparisonResult(
                dimensions=[DimensionScore("statement", 1.0, 0.0, "AST 不可用", False)],
                warnings=warnings,
                force_review=True,
                rejection_reason="parse_error",
            )

        ref_type = self._statement_type(ref_ast)
        stu_type = self._statement_type(stu_ast)
        if ref_type != stu_type:
            message = f"顶层语句类型不一致: ref={ref_type} stu={stu_type}"
            warnings.append(message)
            return SQLComparisonResult(
                dimensions=[DimensionScore("statement", 1.0, 0.0, message, False)],
                warnings=warnings,
                force_review=True,
                reference_type=ref_type,
                student_type=stu_type,
                active_dimensions=["statement"],
                rejection_reason="statement_type_mismatch",
            )

        if ref_type != "SELECT":
            warnings.append(f"第一阶段仅支持 SELECT 结构评分，当前为 {ref_type}")
            return SQLComparisonResult(
                dimensions=[DimensionScore("statement", 1.0, 0.0, "非 SELECT", False)],
                warnings=warnings,
                force_review=True,
                reference_type=ref_type,
                student_type=stu_type,
                active_dimensions=["statement"],
                rejection_reason="unsupported_statement_type",
            )

        all_dims = [
            self._score_select(ref_ast, stu_ast),
            self._score_from(ref_ast, stu_ast),
            self._score_join(ref_ast, stu_ast),
            self._score_where(ref_ast, stu_ast),
            self._score_group_by(ref_ast, stu_ast),
            self._score_having(ref_ast, stu_ast),
            self._score_order_by(ref_ast, stu_ast),
            self._score_limit(ref_ast, stu_ast),
            self._score_aggregates(ref_ast, stu_ast),
            self._score_subquery(ref_ast, stu_ast),
            self._score_operators(ref_ast, stu_ast),
        ]
        optional_active = {
            "join": bool(self._join_signature(ref_ast)),
            "where": bool(self._where_signature(ref_ast)),
            "group_by": bool(self._group_by_exprs(ref_ast)),
            "having": bool(self._clause_sql_set(ref_ast, exp.Having)),
            "order_by": bool(self._order_by_exprs(ref_ast)),
            "limit": self._limit_value(ref_ast) is not None,
            "aggregates": bool(self._aggregate_signature(ref_ast)),
            "subquery": self._subquery_count(ref_ast) > 0,
            "operators": bool(self._operator_signature(ref_ast)),
        }
        dims = [
            dimension
            for dimension in all_dims
            if dimension.name in {"select", "from"}
            or optional_active.get(dimension.name, False)
        ]
        return SQLComparisonResult(
            dimensions=dims,
            warnings=warnings,
            force_review=False,
            reference_type=ref_type,
            student_type=stu_type,
            active_dimensions=[dimension.name for dimension in dims],
        )

    def _parse(self, sql: str) -> tuple[Any | None, str | None]:
        if not sql:
            return None, "空 SQL"
        try:
            statements = [statement for statement in sqlglot.parse(sql) if statement is not None]
            if len(statements) != 1:
                return None, f"只允许单条 SQL，检测到 {len(statements)} 条语句"
            return statements[0], None
        except ParseError as e:
            return None, str(e)
        except Exception as e:  # pragma: no cover
            return None, str(e)

    @staticmethod
    def _statement_type(ast: Any) -> str:
        if exp is not None and isinstance(ast, exp.Select):
            return "SELECT"
        return type(ast).__name__.upper()

    def _score_select(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["select"]
        r = self._select_exprs(ref)
        s = self._select_exprs(stu)
        sim = self._set_similarity(r, s)
        return DimensionScore("select", w, sim, f"SELECT 字段 ref={sorted(r)} stu={sorted(s)}", sim >= 0.99)

    def _score_from(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["from"]
        r = self._table_names(ref, include_joins=False)
        s = self._table_names(stu, include_joins=False)
        sim = self._set_similarity(r, s)
        return DimensionScore("from", w, sim, f"FROM 表 ref={sorted(r)} stu={sorted(s)}", sim >= 0.99)

    def _score_join(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["join"]
        r_joins = self._join_signature(ref)
        s_joins = self._join_signature(stu)
        if not r_joins and not s_joins:
            return DimensionScore("join", w, 1.0, "双方均无 JOIN", True)
        if not r_joins or not s_joins:
            return DimensionScore("join", w, 0.0, f"JOIN 存在性不一致 ref={r_joins} stu={s_joins}", False)
        sim = self._set_similarity(r_joins, s_joins)
        return DimensionScore("join", w, sim, f"JOIN ref={sorted(r_joins)} stu={sorted(s_joins)}", sim >= 0.99)

    def _score_where(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["where"]
        r = self._where_signature(ref)
        s = self._where_signature(stu)
        if not r and not s:
            return DimensionScore("where", w, 1.0, "双方均无 WHERE", True)
        if not r or not s:
            return DimensionScore("where", w, 0.0, f"WHERE 存在性不一致", False)
        sim = self._set_similarity(r, s)
        # 运算符方向错误会体现在 signature 不同 → 低分
        return DimensionScore("where", w, sim, f"WHERE 条件 ref={sorted(r)} stu={sorted(s)}", sim >= 0.99)

    def _score_group_by(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["group_by"]
        r = self._group_by_exprs(ref)
        s = self._group_by_exprs(stu)
        if not r and not s:
            return DimensionScore("group_by", w, 1.0, "双方均无 GROUP BY", True)
        sim = self._set_similarity(r, s)
        return DimensionScore("group_by", w, sim, f"GROUP BY ref={sorted(r)} stu={sorted(s)}", sim >= 0.99)

    def _score_having(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["having"]
        r = self._clause_sql_set(ref, exp.Having)
        s = self._clause_sql_set(stu, exp.Having)
        if not r and not s:
            return DimensionScore("having", w, 1.0, "双方均无 HAVING", True)
        sim = self._set_similarity(r, s)
        return DimensionScore("having", w, sim, f"HAVING ref={sorted(r)} stu={sorted(s)}", sim >= 0.99)

    def _score_order_by(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["order_by"]
        r = self._order_by_exprs(ref)
        s = self._order_by_exprs(stu)
        if not r and not s:
            return DimensionScore("order_by", w, 1.0, "双方均无 ORDER BY", True)
        sim = self._set_similarity(r, s)
        return DimensionScore("order_by", w, sim, f"ORDER BY ref={sorted(r)} stu={sorted(s)}", sim >= 0.99)

    def _score_limit(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["limit"]
        r = self._limit_value(ref)
        s = self._limit_value(stu)
        if r is None and s is None:
            return DimensionScore("limit", w, 1.0, "双方均无 LIMIT", True)
        if r == s:
            return DimensionScore("limit", w, 1.0, f"LIMIT={r}", True)
        return DimensionScore("limit", w, 0.0, f"LIMIT 不一致 ref={r} stu={s}", False)

    def _score_aggregates(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["aggregates"]
        r = self._aggregate_signature(ref)
        s = self._aggregate_signature(stu)
        if not r and not s:
            return DimensionScore("aggregates", w, 1.0, "双方均无聚合函数", True)
        sim = self._set_similarity(r, s)
        return DimensionScore("aggregates", w, sim, f"聚合 ref={sorted(r)} stu={sorted(s)}", sim >= 0.99)

    def _score_subquery(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["subquery"]
        r = self._subquery_count(ref)
        s = self._subquery_count(stu)
        if r == 0 and s == 0:
            return DimensionScore("subquery", w, 1.0, "双方均无子查询", True)
        if r == s:
            return DimensionScore("subquery", w, 1.0, f"子查询数量={r}", True)
        # 数量接近给部分分
        sim = 1.0 - min(1.0, abs(r - s) / max(r, s, 1))
        return DimensionScore("subquery", w, sim, f"子查询数量 ref={r} stu={s}", sim >= 0.99)

    def _score_operators(self, ref, stu) -> DimensionScore:
        w = _DEFAULT_WEIGHTS["operators"]
        r = self._operator_signature(ref)
        s = self._operator_signature(stu)
        if not r and not s:
            return DimensionScore("operators", w, 1.0, "双方均无比较运算符", True)
        sim = self._set_similarity(r, s)
        return DimensionScore("operators", w, sim, f"运算符 ref={sorted(r)} stu={sorted(s)}", sim >= 0.99)

    # ----- AST helpers -----

    @staticmethod
    def _norm_sql(node: Any) -> str:
        try:
            return node.sql(normalize=True).lower()
        except Exception:
            return str(node).lower()

    def _select_exprs(self, ast) -> set[str]:
        out: set[str] = set()
        for sel in ast.find_all(exp.Select):
            # 仅顶层 select 的 expressions；嵌套在子查询的另外计
            for e in sel.expressions:
                out.add(self._norm_sql(e))
            break
        return out

    def _table_names(self, ast, *, include_joins: bool) -> set[str]:
        names: set[str] = set()
        for table in ast.find_all(exp.Table):
            parent_join = table.find_ancestor(exp.Join) if hasattr(table, "find_ancestor") else None
            # sqlglot Expression 用 parent 链
            is_in_join = any(isinstance(p, exp.Join) for p in self._ancestors(table))
            if include_joins or not is_in_join:
                name = (table.name or self._norm_sql(table)).lower()
                names.add(name)
        # FROM 主表：没有 join 祖先的 table
        if not include_joins:
            names = {
                (t.name or self._norm_sql(t)).lower()
                for t in ast.find_all(exp.Table)
                if not any(isinstance(p, exp.Join) for p in self._ancestors(t))
            }
        return names

    @staticmethod
    def _ancestors(node) -> list[Any]:
        out = []
        p = getattr(node, "parent", None)
        while p is not None:
            out.append(p)
            p = getattr(p, "parent", None)
        return out

    def _join_signature(self, ast) -> set[str]:
        sigs: set[str] = set()
        for join in ast.find_all(exp.Join):
            kind = (join.args.get("side") or join.args.get("kind") or "join")
            if isinstance(kind, str):
                kind = kind.lower()
            else:
                kind = str(kind).lower()
            tables = [
                (t.name or "").lower()
                for t in join.find_all(exp.Table)
            ]
            on = join.args.get("on")
            on_sql = self._norm_sql(on) if on is not None else ""
            sigs.add(f"{kind}|{','.join(sorted(tables))}|{on_sql}")
        return sigs

    def _where_signature(self, ast) -> set[str]:
        where = ast.find(exp.Where)
        if where is None:
            return set()
        return self._predicate_atoms(where.this)

    def _predicate_atoms(self, node) -> set[str]:
        if node is None:
            return set()
        atoms: set[str] = set()
        if isinstance(node, exp.And):
            atoms |= self._predicate_atoms(node.left)
            atoms |= self._predicate_atoms(node.right)
            return atoms
        if isinstance(node, exp.Or):
            # 保留 OR 整体，避免拆散改变语义
            atoms.add("or:" + self._norm_sql(node))
            return atoms
        atoms.add(self._norm_sql(node))
        return atoms

    def _group_by_exprs(self, ast) -> set[str]:
        g = ast.find(exp.Group)
        if g is None:
            return set()
        return {self._norm_sql(e) for e in g.expressions}

    def _clause_sql_set(self, ast, clause_type) -> set[str]:
        node = ast.find(clause_type)
        if node is None:
            return set()
        return {self._norm_sql(node)}

    def _order_by_exprs(self, ast) -> set[str]:
        o = ast.find(exp.Order)
        if o is None:
            return set()
        return {self._norm_sql(e) for e in o.expressions}

    def _limit_value(self, ast) -> str | None:
        lim = ast.find(exp.Limit)
        if lim is None:
            return None
        expr = lim.expression
        return self._norm_sql(expr) if expr is not None else ""

    def _aggregate_signature(self, ast) -> set[str]:
        sigs: set[str] = set()
        for fn in ast.find_all(exp.AggFunc):
            sigs.add(self._norm_sql(fn))
        return sigs

    def _subquery_count(self, ast) -> int:
        return sum(1 for _ in ast.find_all(exp.Subquery))

    def _operator_signature(self, ast) -> set[str]:
        ops: set[str] = set()
        for cls_name in ("EQ", "NEQ", "GT", "GTE", "LT", "LTE", "Like", "In", "Between"):
            cls = getattr(exp, cls_name, None)
            if cls is None:
                continue
            for node in ast.find_all(cls):
                ops.add(self._norm_sql(node))
        return ops

    @staticmethod
    def _set_similarity(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0


class SQLScoreMapper:
    """将维度相似度映射为题目分数。"""

    def map(
        self,
        dimensions: list[DimensionScore],
        max_score: float,
        precision: int,
    ) -> tuple[float, float, list[EvidenceItem], list[EvidenceItem]]:
        if not dimensions:
            return 0.0, 0.0, [], []

        weight_sum = sum(d.weight for d in dimensions) or 1.0
        # 归一化权重
        overall = sum(d.weighted for d in dimensions) / weight_sum
        score = round(overall * max_score, precision)

        matched: list[EvidenceItem] = []
        missed: list[EvidenceItem] = []
        for d in dimensions:
            dim_max = round((d.weight / weight_sum) * max_score, precision)
            dim_score = round(d.similarity * dim_max, precision)
            item = EvidenceItem(
                point_id=f"sql.{d.name}",
                score=dim_score,
                max_score=dim_max,
                evidence=d.detail if d.matched else None,
                reason=d.detail,
                similarity=round(d.similarity, 4),
            )
            if d.similarity >= 0.99:
                matched.append(item)
            else:
                missed.append(item)

        confidence = float(max(0.0, min(1.0, overall)))
        return score, confidence, matched, missed


class SQLStructureScorer:
    """SQL 结构评分引擎。"""

    name = "SQLStructureScorer"

    def __init__(
        self,
        *,
        normalizer: SQLNormalizer | None = None,
        comparator: SQLAstComparator | None = None,
        mapper: SQLScoreMapper | None = None,
    ) -> None:
        self.normalizer = normalizer or SQLNormalizer()
        self.comparator = comparator or SQLAstComparator()
        self.mapper = mapper or SQLScoreMapper()

    def score(self, request: ScoringRequest) -> IntermediateScoreResult:
        precision = request.scoring_config.score_precision
        ref = self.normalizer.normalize(request.reference_answer)
        stu = self.normalizer.normalize(request.student_answer)

        warnings: list[str] = []
        if not ref:
            warnings.append("标准答案 SQL 为空")
        if not stu:
            warnings.append("学生答案 SQL 为空")

        comparison = self.comparator.compare_detailed(ref, stu)
        warnings.extend(comparison.warnings)

        score, confidence, matched, missed = self.mapper.map(
            comparison.dimensions, request.max_score, precision
        )

        force_review = comparison.force_review
        if force_review:
            confidence = min(confidence, 0.4)
        if not stu or not ref:
            force_review = True
            confidence = min(confidence, 0.2)

        return IntermediateScoreResult(
            scorer=self.name,
            scoring_mode=ScoringMode.SQL,
            score=score,
            max_score=request.max_score,
            confidence=round(confidence, 4),
            matched_evidence=matched,
            missed_evidence=missed,
            warnings=warnings,
            force_manual_review=force_review,
            metadata={
                "model": None,
                "parser": "sqlglot" if sqlglot is not None else None,
                "normalized_reference": ref,
                "normalized_student": stu,
                "reference_statement_type": comparison.reference_type,
                "student_statement_type": comparison.student_type,
                "active_dimensions": comparison.active_dimensions,
                "rejection_reason": comparison.rejection_reason,
            },
        )

    def __call__(self, request: ScoringRequest) -> IntermediateScoreResult:
        return self.score(request)


__all__ = [
    "SQLAstComparator",
    "SQLComparisonResult",
    "SQLNormalizer",
    "SQLScoreMapper",
    "SQLStructureScorer",
]
