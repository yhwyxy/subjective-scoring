"""代码语言自动探测回归测试（真实 54 份重评中发现的失分场景）。"""

from __future__ import annotations

import pytest

from subjective_scoring import ScoringMode, ScoringRequest
from subjective_scoring.engines.code_hybrid import CodeHybridScorer, TreeSitterAstExtractor

_JS_FIND_INDEX = "function findIndex(array, item) {\n    return array.indexOf(item);\n}"
_PY_NESTED_LOOP = "for i in range(5):\n    for j in range(5):\n        s = i + j"


def _has_tree_sitter() -> bool:
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_javascript  # noqa: F401
        import tree_sitter_python  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_tree_sitter(), reason="tree-sitter 语言包未安装"
)


def test_extract_auto_detects_javascript_when_declared_python():
    """真实失分回归（software-development q21）：JS 代码被按 python 解析失败
    -> 结构分 0。extract_auto 应探测出 javascript。"""
    features, lang = TreeSitterAstExtractor().extract_auto(_JS_FIND_INDEX, "python")
    assert features.parse_ok is True
    assert lang == "javascript"


def test_extract_auto_prefers_declared_language():
    features, lang = TreeSitterAstExtractor().extract_auto(_PY_NESTED_LOOP, "python")
    assert features.parse_ok is True
    assert lang == "python"


def test_hybrid_scorer_recovers_structure_for_js_answer_without_language():
    """code_language 缺失（默认 python）+ 参考/学生都是 JS：
    结构分不得再因解析失败归零。"""
    scorer = CodeHybridScorer(
        pair_scorer=lambda a, b: 0.9,
        allow_model_load=False,
    )
    result = scorer.score(
        ScoringRequest(
            question_id="q21",
            max_score=15,
            scoring_mode=ScoringMode.CODE,
            reference_answer=_JS_FIND_INDEX,
            student_answer="function find(arr, x) {\n    for (var i = 0; i < arr.length; i++) {\n        if (arr[i] === x) { return i; }\n    }\n    return -1;\n}",
        )
    )

    assert result.score > 7.0  # 旧行为: 双侧 AST 失败 -> 0 分
    assert not any("AST 提取失败" in w for w in result.warnings)


def test_static_profile_find_index_accepts_js_answer():
    """静态结构轨道：find_index_static 对 JS 作答按结构覆盖给分。"""
    from subjective_scoring.engines.code_static import CodeStaticScorer

    result = CodeStaticScorer().score(
        ScoringRequest(
            question_id="q21",
            max_score=15,
            scoring_mode=ScoringMode.CODE,
            code_scoring_profile="find_index_static",
            code_language="javascript",
            student_answer=_JS_FIND_INDEX,
        )
    )

    assert result.score == 15.0
    assert result.metadata["checks"]["parse"] is True


def test_static_profile_nested_loop_full_marks_for_python():
    from subjective_scoring.engines.code_static import CodeStaticScorer

    result = CodeStaticScorer().score(
        ScoringRequest(
            question_id="q16",
            max_score=10,
            scoring_mode=ScoringMode.CODE,
            code_scoring_profile="nested_loop_static",
            code_language="python",
            student_answer=_PY_NESTED_LOOP,
        )
    )

    # 裸嵌套循环（无函数包装）complete_structure 检查扣 1 分：9/10
    assert result.score >= 9.0
    assert result.metadata["checks"]["nested_loop"] is True
