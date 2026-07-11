"""CodeHybridScorer 单元测试（注入语义分，真实 tree-sitter 结构分）。"""

from __future__ import annotations

from subjective_scoring import CodeScoreWeights, ScoringMode, ScoringOptions, ScoringRequest
from subjective_scoring.engines import CodeHybridScorer
from subjective_scoring.engines.code_hybrid import CodeNormalizer, TreeSitterAstExtractor


REF_PY = (
    "def sum_n(n):\n"
    "    total = 0\n"
    "    for i in range(n):\n"
    "        total += i\n"
    "    return total\n"
)

STU_PY_OK = (
    "def sum_n(n):\n"
    "    s = 0\n"
    "    for x in range(n):\n"
    "        s = s + x\n"
    "    return s\n"
)

STU_PY_NO_LOOP = (
    "def sum_n(n):\n"
    "    return n * (n - 1) // 2\n"
)


def _req(stu: str, ref: str = REF_PY, **kwargs) -> ScoringRequest:
    data = dict(
        question_id="c1",
        max_score=10,
        scoring_mode=ScoringMode.CODE,
        code_language="python",
        reference_answer=ref,
        student_answer=stu,
        scoring_config=ScoringOptions(
            code_score_weights=CodeScoreWeights(semantic=0.7, structure=0.3),
            score_precision=1,
        ),
    )
    data.update(kwargs)
    return ScoringRequest.model_validate(data)


def test_hybrid_fusion_with_injected_semantic():
    scorer = CodeHybridScorer(
        pair_scorer=lambda a, b: 0.8,
        allow_model_load=False,
    )
    result = scorer.score(_req(STU_PY_OK))

    assert result.scorer == "CodeHybridScorer"
    assert result.scoring_mode is ScoringMode.CODE
    assert result.metadata["semantic_similarity"] == 0.8
    assert result.metadata["structure_similarity"] >= 0.5
    assert result.score >= 5.5
    assert any(e.point_id == "code.semantic" for e in result.matched_evidence)
    assert any(
        e.point_id and e.point_id.startswith("code.structure")
        for e in (result.matched_evidence + result.missed_evidence)
    )


def test_missing_loop_lowers_structure_score():
    scorer = CodeHybridScorer(
        pair_scorer=lambda a, b: 0.9,
        allow_model_load=False,
    )
    with_loop = scorer.score(_req(STU_PY_OK))
    no_loop = scorer.score(_req(STU_PY_NO_LOOP))
    assert with_loop.metadata["structure_similarity"] > no_loop.metadata["structure_similarity"]


def test_semantic_structure_conflict_forces_review():
    scorer = CodeHybridScorer(
        pair_scorer=lambda a, b: 0.95,
        allow_model_load=False,
        conflict_gap=0.3,
    )
    result = scorer.score(_req(STU_PY_NO_LOOP))
    if abs(0.95 - result.metadata["structure_similarity"]) >= 0.3:
        assert result.force_manual_review is True
        assert any("差异较大" in w for w in result.warnings)


def test_normalizer_strips_comments_keeps_indent():
    n = CodeNormalizer(strip_comments=True)
    code = "def f():\n    # comment\n    return 1\n"
    out = n.normalize(code, "python")
    assert "# comment" not in out
    assert "    return 1" in out


def test_tree_sitter_detects_loop():
    feat = TreeSitterAstExtractor().extract(STU_PY_OK, "python")
    assert feat.parse_ok is True
    assert feat.flags.get("loop") is True
    assert feat.flags.get("function") is True
    assert feat.flags.get("return") is True
