"""主观题自动评分模块。

第一版架构见 docs/superpowers/specs/2026-07-11-subjective-scoring-design.md。

快速使用::

    from subjective_scoring import SubjectiveScoringService, ScoringRequest

    service = SubjectiveScoringService(allow_model_load=False)
    result = service.score({
        "question_id": "q1",
        "max_score": 10,
        "scoring_mode": "text",
        "student_answer": "...",
        "scoring_points": [{"id": "p1", "text": "...", "score": 10}],
    })
"""

from .components import (
    CodeNormalizer,
    InputNormalizerComponent,
    NormalizationResult,
    QuestionTypeRouter,
    RouteDecision,
    SQLNormalizer,
    ScoreAggregatorComponent,
    TextNormalizer,
)
from .engines import (
    CodeHybridScorer,
    RuleInterceptor,
    SQLStructureScorer,
    ScoringPointResolver,
    TextRerankerScorer,
)
from .models import (
    REVIEW_LEVEL_RANK,
    CodeScoreWeights,
    EvidenceItem,
    IntermediateScoreResult,
    ManualReviewThresholds,
    MatchedPoint,
    MissedPoint,
    ReviewLevel,
    ScoringMode,
    ScoringOptions,
    ScoringPoint,
    ScoringRequest,
    ScoringResult,
)
from .service import (
    PipelineTrace,
    ScoringServiceResult,
    SubjectiveScoringService,
    create_default_service,
)
from .rerankers import (
    CohereRerankerPairScorer,
    RemoteRerankerError,
    RemoteRerankerRequestError,
    RemoteRerankerResponseError,
)

__all__ = [
    "REVIEW_LEVEL_RANK",
    "CodeHybridScorer",
    "CodeNormalizer",
    "CodeScoreWeights",
    "EvidenceItem",
    "InputNormalizerComponent",
    "IntermediateScoreResult",
    "ManualReviewThresholds",
    "MatchedPoint",
    "MissedPoint",
    "NormalizationResult",
    "PipelineTrace",
    "QuestionTypeRouter",
    "CohereRerankerPairScorer",
    "RemoteRerankerError",
    "RemoteRerankerRequestError",
    "RemoteRerankerResponseError",
    "ReviewLevel",
    "RouteDecision",
    "RuleInterceptor",
    "SQLNormalizer",
    "SQLStructureScorer",
    "ScoreAggregatorComponent",
    "ScoringMode",
    "ScoringOptions",
    "ScoringPoint",
    "ScoringPointResolver",
    "ScoringRequest",
    "ScoringResult",
    "ScoringServiceResult",
    "SubjectiveScoringService",
    "TextNormalizer",
    "TextRerankerScorer",
    "create_default_service",
]
