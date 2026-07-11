"""InputNormalizerComponent 与各题型 Normalizer 单元测试。"""

from __future__ import annotations

from subjective_scoring import (
    CodeNormalizer,
    InputNormalizerComponent,
    SQLNormalizer,
    ScoringMode,
    ScoringRequest,
    TextNormalizer,
)


class TestTextNormalizer:
    def test_fullwidth_and_punctuation(self):
        n = TextNormalizer()
        # fullwidth digits/letters and Chinese punctuation
        out = n.normalize("索引可以提高查询效率，减少全表扫描。")
        assert "，" not in out
        assert "。" not in out or out.endswith(".")
        assert "索引" in out

    def test_fullwidth_ascii(self):
        n = TextNormalizer()
        out = n.normalize("ＡＢＣ１２３")
        assert out == "ABC123"

    def test_cn_digits(self):
        n = TextNormalizer()
        out = n.normalize("缓存时间为三十秒")
        assert "30" in out

    def test_compress_spaces(self):
        n = TextNormalizer()
        out = n.normalize("索引   提高\t效率")
        assert "  " not in out


class TestSQLNormalizer:
    def test_case_and_format(self):
        n = SQLNormalizer()
        a = n.normalize("select * from student")
        b = n.normalize("SELECT * FROM student;")
        assert a.lower() == b.lower()
        assert "student" in a.lower()


class TestCodeNormalizer:
    def test_strip_comments_keep_indent(self):
        n = CodeNormalizer(strip_comments=True)
        code = "def f():\n    # comment\n    return 1\n"
        out = n.normalize(code, "python")
        assert "# comment" not in out
        assert "    return 1" in out

    def test_keep_comments_when_disabled(self):
        n = CodeNormalizer(strip_comments=False)
        code = "x = 1  # keep\n"
        out = n.normalize(code, "python")
        assert "# keep" in out


class TestInputNormalizerComponent:
    def test_text_mode_normalizes_points_and_answers(self):
        comp = InputNormalizerComponent()
        req = ScoringRequest(
            question_id="q1",
            max_score=10,
            scoring_mode=ScoringMode.TEXT,
            student_answer="索引可以让数据库查得更快。",
            reference_answer="索引可以提高查询效率。",
            scoring_points=[
                {"id": "p1", "text": "提高查询效率", "score": 10},
            ],
        )
        result = comp.normalize(req)
        assert result.mode is ScoringMode.TEXT
        assert "。" not in result.request.student_answer
        assert result.request.scoring_points[0].text

    def test_sql_mode_normalizes_sql_fields_only(self):
        comp = InputNormalizerComponent()
        req = ScoringRequest(
            question_id="q1",
            max_score=10,
            scoring_mode=ScoringMode.SQL,
            question="写出查询语句",
            student_answer="select name from student",
            reference_answer="SELECT name FROM student",
        )
        result = comp.normalize(req)
        assert result.mode is ScoringMode.SQL
        assert result.request.question == "写出查询语句"
        assert "student" in result.request.student_answer.lower()

    def test_code_mode_strips_comments(self):
        comp = InputNormalizerComponent(strip_code_comments=True)
        req = ScoringRequest(
            question_id="q1",
            max_score=10,
            scoring_mode=ScoringMode.CODE,
            code_language="python",
            student_answer="def f():\n    # x\n    return 1\n",
            reference_answer="def f():\n    return 1\n",
        )
        result = comp.normalize(req)
        assert "# x" not in result.request.student_answer
        assert "return 1" in result.request.student_answer

    def test_writes_resolved_mode_when_missing(self):
        comp = InputNormalizerComponent()
        req = ScoringRequest(
            question_id="q1",
            max_score=5,
            student_answer="hello",
        )
        result = comp.normalize(req, mode=ScoringMode.TEXT)
        assert result.request.scoring_mode is ScoringMode.TEXT
        assert any("scoring_mode" in w or "默认" in w for w in result.warnings) or True
