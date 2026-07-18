"""主观题自动评分模块的统一请求 / 中间结果 / 最终结果数据模型。

字段语义对齐设计文档：
docs/superpowers/specs/2026-07-11-subjective-scoring-design.md

约定：
- Python 侧统一 snake_case（与现有 FastAPI 接口一致）。
- 枚举使用 str 继承，便于 JSON 序列化与日志输出。
- 分数 / 置信度做基础边界校验，业务阈值由 ScoringOptions 配置。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class ScoringMode(str, Enum):
    """评分模式：Router 优先依赖显式 scoring_mode。"""

    TEXT = "text"
    SQL = "sql"
    CODE = "code"
    CALCULATION = "calculation"


class ReviewLevel(str, Enum):
    """人工复核等级。"""

    AUTO_PASS = "auto_pass"
    SUGGESTED_REVIEW = "suggested_review"
    MANUAL_REQUIRED = "manual_required"


class PointRelation(str, Enum):
    """学生答案证据与原子评分点之间的语义关系。"""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class PointConflictPolicy(str, Enum):
    """评分点发生高置信硬冲突时的整题处理策略。"""

    POINT_ZERO = "point_zero"
    CAP_TOTAL = "cap_total"
    ZERO_TOTAL = "zero_total"


class ScoringDecision(str, Enum):
    """自动评分器对当前答案作出的最终决策。"""

    AUTO_SCORE = "auto_score"
    AUTO_ZERO = "auto_zero"
    MANUAL_REVIEW = "manual_review"


# 复核等级严重程度：数值越大越严格，供 Aggregator 取最严结果。
REVIEW_LEVEL_RANK: dict[ReviewLevel, int] = {
    ReviewLevel.AUTO_PASS: 0,
    ReviewLevel.SUGGESTED_REVIEW: 1,
    ReviewLevel.MANUAL_REQUIRED: 2,
}


# ---------------------------------------------------------------------------
# 请求侧嵌套结构
# ---------------------------------------------------------------------------


class ScoringPoint(BaseModel):
    """人工配置的原子评分点；每项应描述一个可独立验证的结论。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="评分点唯一标识")
    text: str = Field(..., min_length=1, description="评分点描述文本")
    score: float = Field(..., ge=0, description="该评分点满分")
    required: bool = Field(default=False, description="是否为必答知识点")
    critical: bool = Field(
        default=False,
        description="是否为关键结论；不确定或冲突时必须人工复核",
    )
    conflict_policy: PointConflictPolicy = Field(
        default=PointConflictPolicy.POINT_ZERO,
        description="高置信硬冲突时仅该点归零、整题封顶或整题归零",
    )
    conflict_score_cap_ratio: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="conflict_policy=cap_total 时的整题最高得分比例",
    )


class ManualReviewThresholds(BaseModel):
    """置信度阈值：auto_pass / review 分界。"""

    model_config = ConfigDict(extra="forbid")

    auto_pass: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="confidence >= auto_pass 时自动通过",
    )
    review: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="confidence < review 时必须人工处理；介于 review 与 auto_pass 之间建议复核",
    )

    @model_validator(mode="after")
    def _check_order(self) -> ManualReviewThresholds:
        if self.review > self.auto_pass:
            raise ValueError("manual_review_thresholds.review 不得大于 auto_pass")
        return self


class TextRelationThresholds(BaseModel):
    """文本原子评分点的支持、冲突与拒判阈值。"""

    model_config = ConfigDict(extra="forbid")

    support: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="校准相似度达到该值时，评分点可判定为 supported",
    )
    local_cross_encoder_support: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="本地 CrossEncoder 的默认支持阈值",
    )
    remote_reranker_support: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="云端 Reranker 的默认支持阈值",
    )
    lexical_fallback_support: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="词法回退后端的默认支持阈值",
    )
    conflict: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="硬冲突规则达到该置信度时，评分点判定为 contradicted",
    )
    reject_when_no_supported: bool = Field(
        default=True,
        description="无 supported 点但存在 unknown 点时拒绝自动定分并要求人工复核",
    )
    required_unknown_requires_review: bool = Field(
        default=True,
        description="required 或 critical 评分点为 unknown 时要求人工复核",
    )
    apply_gate_to_synthetic_reference: bool = Field(
        default=False,
        description="是否对未拆分的全文标准答案兜底点应用原子评分门槛",
    )
    validate_reference_points: bool = Field(
        default=True,
        description="评分前验证标准答案能否支持 required / critical 原子评分点",
    )


class CodeScoreWeights(BaseModel):
    """代码题语义分 / 结构分融合权重，默认 0.7 / 0.3。"""

    model_config = ConfigDict(extra="forbid")

    semantic: float = Field(default=0.7, ge=0.0, le=1.0)
    structure: float = Field(default=0.3, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum(self) -> CodeScoreWeights:
        total = self.semantic + self.structure
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"code_score_weights.semantic + structure 必须等于 1.0，当前为 {total}"
            )
        return self


class CalculationItem(BaseModel):
    """一个可确定性核验的计算步骤或最终答案。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    expected: float = Field(...)
    score: float = Field(..., ge=0)
    tolerance: float = Field(default=0.0, ge=0)
    unit: str | None = Field(default=None, description="可选单位，如 kg、%")
    keywords: tuple[str, ...] = Field(
        default_factory=tuple,
        description="可选步骤标签；配置后只在包含标签的行中匹配",
    )


class CalculationScoringConfig(BaseModel):
    """固定题目的计算步骤评分配置，不执行任意学生表达式。"""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["static_values"] = "static_values"
    steps: tuple[CalculationItem, ...] = Field(default_factory=tuple)
    final_answers: tuple[CalculationItem, ...] = Field(default_factory=tuple)
    require_working: bool = Field(
        default=False,
        description="要求过程时，只有检测到公式/等式结构才给步骤分",
    )
    final_only_score_cap: float | None = Field(
        default=None,
        ge=0,
        description="只提交最终答案时的最高得分；为空表示不封顶",
    )

    @model_validator(mode="after")
    def _check_ids_and_total(self) -> CalculationScoringConfig:
        items = [*self.steps, *self.final_answers]
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("calculation steps/final_answers 的 id 必须唯一")
        total = sum(item.score for item in items)
        if self.final_only_score_cap is not None and self.final_only_score_cap > total + 1e-6:
            raise ValueError("final_only_score_cap 不得超过计算配置总分")
        return self


class ScoringOptions(BaseModel):
    """单次评分的可调参数（对应设计文档 scoringConfig）。

    与 backend.config.ScoringConfig（客观题给分规则）无关。
    """

    model_config = ConfigDict(extra="forbid")

    manual_review_thresholds: ManualReviewThresholds = Field(
        default_factory=ManualReviewThresholds,
    )
    text_relation_thresholds: TextRelationThresholds = Field(
        default_factory=TextRelationThresholds,
    )
    code_score_weights: CodeScoreWeights = Field(
        default_factory=CodeScoreWeights,
    )
    calculation: CalculationScoringConfig = Field(
        default_factory=CalculationScoringConfig,
    )
    allow_auto_scoring_point_generation: bool = Field(
        default=False,
        description="是否允许自动生成评分点；第一版默认关闭",
    )
    score_precision: int = Field(
        default=1,
        ge=0,
        le=6,
        description="最终得分小数位数",
    )
    calibration_points: tuple[tuple[float, float], ...] | None = Field(
        default=None,
        description="本次文本评分使用的单调校准曲线控制点",
    )

    @field_validator("calibration_points")
    @classmethod
    def _validate_calibration_points(
        cls,
        value: tuple[tuple[float, float], ...] | None,
    ) -> tuple[tuple[float, float], ...] | None:
        if value is None:
            return None
        if len(value) < 2:
            raise ValueError("calibration_points 至少需要两个控制点")
        previous_x = previous_y = -1.0
        for x, y in value:
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError("calibration_points 必须位于 0..1")
            if x <= previous_x or y < previous_y:
                raise ValueError(
                    "calibration_points 必须按 x 严格递增且 y 单调不减"
                )
            previous_x, previous_y = x, y
        return value


# ---------------------------------------------------------------------------
# ScoringRequest
# ---------------------------------------------------------------------------


class ScoringRequest(BaseModel):
    """主观题评分统一请求。"""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., min_length=1)
    paper_id: str | None = Field(default=None, description="试卷 ID，可选")
    question_type: str = Field(
        default="subjective",
        description="题型元数据；Router 次优先依赖此字段",
    )
    scoring_mode: ScoringMode | None = Field(
        default=None,
        description="显式评分模式 text/sql/code/calculation；优先于其他路由信号",
    )
    code_scoring_profile: str | None = Field(
        default=None,
        description="代码静态评分模板，如 nested_loop_static / find_index_static",
    )
    code_language: str | None = Field(
        default=None,
        description="代码语言，如 sql/python/java/javascript/cpp/go",
    )
    course_type: str | None = Field(
        default=None,
        description="课程 / 学科类型，作为弱路由信号",
    )
    max_score: float = Field(..., ge=0, description="题目满分")
    question: str = Field(default="", description="题干")
    reference_answer: str = Field(default="", description="标准答案")
    scoring_points: list[ScoringPoint] = Field(
        default_factory=list,
        description="人工评分点；文本题第一版推荐配置",
    )
    student_answer: str = Field(default="", description="学生作答")
    scoring_config: ScoringOptions = Field(
        default_factory=ScoringOptions,
        description="本次评分的阈值与权重配置",
    )

    @field_validator("code_language")
    @classmethod
    def _normalize_code_language(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalized = v.strip().lower()
        return normalized or None

    @model_validator(mode="after")
    def _check_scoring_points_total(self) -> ScoringRequest:
        if not self.scoring_points:
            return self
        point_ids = [point.id for point in self.scoring_points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("scoring_points.id 必须唯一，才能独立追踪原子评分点")
        total = sum(p.score for p in self.scoring_points)
        # 允许评分点合计略小于满分（部分选答点），但禁止明显超分
        if total > self.max_score + 1e-6:
            raise ValueError(
                f"scoring_points 分值合计 ({total}) 超过 max_score ({self.max_score})"
            )
        return self


# ---------------------------------------------------------------------------
# 中间结果（各 Scorer 统一输出）
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """命中或未命中证据条目。"""

    model_config = ConfigDict(extra="forbid")

    point_id: str | None = Field(
        default=None,
        description="关联评分点 ID；SQL/Code 结构证据可为空，由 Aggregator 合成",
    )
    score: float = Field(default=0.0, ge=0)
    provisional_score: float | None = Field(
        default=None,
        ge=0,
        description="关系不确定时供人工复核参考的估分，不可直接自动入账",
    )
    max_score: float = Field(default=0.0, ge=0)
    evidence: str | None = Field(default=None, description="学生答案中的对应片段")
    reason: str | None = Field(default=None, description="判分说明")
    similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="语义 / 结构相似度，可选",
    )
    relation: PointRelation | None = Field(
        default=None,
        description="文本评分点关系：supported / contradicted / unknown",
    )
    relation_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="关系判定置信度",
    )


class IntermediateScoreResult(BaseModel):
    """各评分引擎统一中间结果，供 ScoreAggregator 合并。"""

    model_config = ConfigDict(extra="forbid")

    scorer: str = Field(..., min_length=1, description="评分器名称")
    scoring_mode: ScoringMode
    score: float = Field(..., ge=0)
    provisional_score: float | None = Field(
        default=None,
        ge=0,
        description="拒判时的待复核估分；正式自动分仍使用 score",
    )
    max_score: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    matched_evidence: list[EvidenceItem] = Field(default_factory=list)
    missed_evidence: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    force_manual_review: bool = Field(
        default=False,
        description=(
            "引擎强制人工复核（解析失败、否定冲突、AST 失败、全文兜底等）；"
            "Aggregator 取最严等级，不会因 confidence 高而被自动通过掩盖"
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="引擎侧元数据，如 model/parser 名称",
    )

    @model_validator(mode="after")
    def _check_score_cap(self) -> IntermediateScoreResult:
        if self.score > self.max_score + 1e-6:
            raise ValueError(
                f"中间分 score ({self.score}) 超过 max_score ({self.max_score})"
            )
        if (
            self.provisional_score is not None
            and self.provisional_score > self.max_score + 1e-6
        ):
            raise ValueError(
                "中间待复核估分 provisional_score "
                f"({self.provisional_score}) 超过 max_score ({self.max_score})"
            )
        return self


# ---------------------------------------------------------------------------
# ScoringResult
# ---------------------------------------------------------------------------


class MatchedPoint(BaseModel):
    """最终结果中的命中评分点。"""

    model_config = ConfigDict(extra="forbid")

    point_id: str
    score: float = Field(..., ge=0)
    provisional_score: float | None = Field(default=None, ge=0)
    max_score: float = Field(..., ge=0)
    similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = None
    reason: str | None = None
    relation: PointRelation | None = None
    relation_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class MissedPoint(BaseModel):
    """最终结果中的未命中评分点。"""

    model_config = ConfigDict(extra="forbid")

    point_id: str
    score: float = Field(default=0.0, ge=0)
    provisional_score: float | None = Field(default=None, ge=0)
    max_score: float = Field(..., ge=0)
    reason: str | None = None
    similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = None
    relation: PointRelation | None = None
    relation_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ScoringResult(BaseModel):
    """主观题评分统一输出。"""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., min_length=1)
    score: float = Field(..., ge=0)
    provisional_score: float | None = Field(
        default=None,
        ge=0,
        description="manual_review 时供人工参考的估分，不代表最终成绩",
    )
    max_score: float = Field(..., ge=0)
    scoring_mode: ScoringMode
    track: str = Field(
        ...,
        min_length=1,
        description="实际走的评分轨道，如 TextRerankerScorer",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    need_manual_review: bool = Field(
        ...,
        description="是否需要人工介入（suggested_review 或 manual_required）",
    )
    review_level: ReviewLevel
    matched_points: list[MatchedPoint] = Field(default_factory=list)
    missed_points: list[MissedPoint] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    decision: ScoringDecision | None = Field(
        default=None,
        description="自动给分、自动归零或拒绝自动定分并转人工复核",
    )
    decision_reason: str | None = Field(
        default=None,
        description="触发当前决策的稳定原因码",
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> ScoringResult:
        if self.score > self.max_score + 1e-6:
            raise ValueError(
                f"最终分 score ({self.score}) 超过 max_score ({self.max_score})"
            )
        if (
            self.provisional_score is not None
            and self.provisional_score > self.max_score + 1e-6
        ):
            raise ValueError(
                "待复核估分 provisional_score "
                f"({self.provisional_score}) 超过 max_score ({self.max_score})"
            )
        # review_level 与 need_manual_review 保持一致
        expects_review = self.review_level != ReviewLevel.AUTO_PASS
        if self.need_manual_review != expects_review:
            raise ValueError(
                "need_manual_review 必须与 review_level 一致："
                "auto_pass -> False，其余 -> True"
            )
        return self
