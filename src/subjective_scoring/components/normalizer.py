"""输入归一化：按题型差异化清洗，避免破坏评分语义。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from subjective_scoring.models import ScoringMode, ScoringPoint, ScoringRequest

try:
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None  # type: ignore[assignment]

try:
    import sqlglot
except ImportError:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_FULLWIDTH_ASCII_START = 0xFF01
_FULLWIDTH_ASCII_END = 0xFF5E
_FULLWIDTH_OFFSET = 0xFEE0  # fullwidth '!' (0xFF01) - '!' (0x21)

_PUNCT_MAP = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "、": ",",
        "《": "<",
        "》": ">",
        "—": "-",
        "–": "-",
        "…": "...",
        "～": "~",
        "　": " ",  # ideographic space
    }
)

_CN_DIGITS = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}

# 简单「十X / X十 / X十Y」模式，避免破坏否定词与长句
_CN_NUM_RE = re.compile(
    r"(?<![零〇一二三四五六七八九十两])"
    r"(?:[一二三四五六七八九两]?十[一二三四五六七八九]?|[零〇一二三四五六七八九两])"
    r"(?![零〇一二三四五六七八九十两])"
)

_MULTI_SPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def fullwidth_to_halfwidth(text: str) -> str:
    """全角 ASCII / 空格转半角。"""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:  # ideographic space
            out.append(" ")
        elif _FULLWIDTH_ASCII_START <= code <= _FULLWIDTH_ASCII_END:
            out.append(chr(code - _FULLWIDTH_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


def _cn_num_to_arabic(token: str) -> str:
    if token in _CN_DIGITS and token != "十":
        return _CN_DIGITS[token]
    if token == "十":
        return "10"
    if token.startswith("十"):
        # 十一 -> 11
        rest = token[1:]
        return "1" + _CN_DIGITS.get(rest, rest)
    if token.endswith("十"):
        # 二十 -> 20
        head = token[:-1]
        return _CN_DIGITS.get(head, head) + "0"
    if "十" in token:
        head, _, tail = token.partition("十")
        return _CN_DIGITS.get(head, head) + _CN_DIGITS.get(tail, tail)
    return _CN_DIGITS.get(token, token)


class TextNormalizer:
    """文本题归一化：Unicode 修复、全半角、标点、空白、可选中文数字。"""

    def __init__(
        self,
        *,
        fix_unicode: bool = True,
        normalize_cn_digits: bool = True,
        unify_punctuation: bool = True,
    ) -> None:
        self.fix_unicode = fix_unicode
        self.normalize_cn_digits = normalize_cn_digits
        self.unify_punctuation = unify_punctuation

    def normalize(self, text: str) -> str:
        if text is None:
            return ""
        s = str(text)
        if self.fix_unicode and ftfy is not None:
            s = ftfy.fix_text(s)
        # NFC 统一组合字符
        s = unicodedata.normalize("NFC", s)
        s = fullwidth_to_halfwidth(s)
        if self.unify_punctuation:
            s = s.translate(_PUNCT_MAP)
        if self.normalize_cn_digits:
            s = _CN_NUM_RE.sub(lambda m: _cn_num_to_arabic(m.group(0)), s)
        # 空白：保留换行语义，压缩横向空白
        s = _MULTI_SPACE_RE.sub(" ", s)
        s = _MULTI_NL_RE.sub("\n\n", s)
        return s.strip()


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


class SQLNormalizer:
    """SQL 归一化：去多余空白、关键字/标识符规范化（sqlglot）。"""

    def normalize(self, sql: str) -> str:
        text = (sql or "").strip()
        if not text:
            return ""
        text = text.rstrip(";").strip()
        text = re.sub(r"\s+", " ", text)
        if sqlglot is None:
            return text
        try:
            parsed = sqlglot.parse_one(text, read=None)
            return parsed.sql(normalize=True)
        except Exception:
            return text


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------


def _comment_patterns() -> dict[str, list[re.Pattern[str]]]:
    dq = chr(34) * 3
    sq = chr(39) * 3
    return {
        "python": [
            re.compile(r"#.*?$", re.M),
            re.compile(dq + r"[\s\S]*?" + dq),
            re.compile(sq + r"[\s\S]*?" + sq),
        ],
        "java": [
            re.compile(r"//.*?$", re.M),
            re.compile(r"/\*[\s\S]*?\*/"),
        ],
        "javascript": [
            re.compile(r"//.*?$", re.M),
            re.compile(r"/\*[\s\S]*?\*/"),
        ],
        "cpp": [
            re.compile(r"//.*?$", re.M),
            re.compile(r"/\*[\s\S]*?\*/"),
        ],
    }


_COMMENT_PATTERNS = _comment_patterns()


def _canonical_lang(language: str | None) -> str:
    lang = (language or "python").lower().strip()
    if lang in {"py"}:
        return "python"
    if lang in {"js", "typescript", "ts"}:
        return "javascript"
    if lang in {"c++", "cc", "cxx", "c"}:
        return "cpp"
    return lang


class CodeNormalizer:
    """代码归一化：编码/换行统一、可选去注释；禁止改写逻辑与变量名。"""

    def __init__(
        self,
        *,
        strip_comments: bool = True,
        collapse_extra_blank_lines: bool = True,
    ) -> None:
        self.strip_comments = strip_comments
        self.collapse_extra_blank_lines = collapse_extra_blank_lines

    def normalize(self, code: str, language: str | None = None) -> str:
        text = (code or "").replace("\r\n", "\n").replace("\r", "\n")
        # 去除 BOM
        text = text.lstrip("\ufeff")
        text = text.strip("\n")
        if self.strip_comments:
            lang = _canonical_lang(language)
            patterns = _COMMENT_PATTERNS.get(lang, _COMMENT_PATTERNS["python"])
            for pat in patterns:
                text = pat.sub("", text)
        lines = [ln.rstrip() for ln in text.split("\n")]
        if not self.collapse_extra_blank_lines:
            return "\n".join(lines).strip()
        cleaned: list[str] = []
        blank_run = 0
        for ln in lines:
            if not ln.strip():
                blank_run += 1
                if blank_run <= 1:
                    cleaned.append("")
            else:
                blank_run = 0
                cleaned.append(ln)
        return "\n".join(cleaned).strip()


# ---------------------------------------------------------------------------
# InputNormalizerComponent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationResult:
    """归一化后的请求及过程信息。"""

    request: ScoringRequest
    mode: ScoringMode
    warnings: list[str]


class InputNormalizerComponent:
    """按评分模式差异化归一化 ScoringRequest。

    仅清洗文本字段（题干 / 参考答案 / 学生答案 / 评分点文本），
    不改写 max_score、配置与路由元数据（除可选写回 scoring_mode）。
    """

    def __init__(
        self,
        *,
        text_normalizer: TextNormalizer | None = None,
        sql_normalizer: SQLNormalizer | None = None,
        code_normalizer: CodeNormalizer | None = None,
        strip_code_comments: bool = True,
        write_resolved_mode: bool = True,
    ) -> None:
        self.text_normalizer = text_normalizer or TextNormalizer()
        self.sql_normalizer = sql_normalizer or SQLNormalizer()
        self.code_normalizer = code_normalizer or CodeNormalizer(
            strip_comments=strip_code_comments
        )
        self.write_resolved_mode = write_resolved_mode

    def normalize(
        self,
        request: ScoringRequest,
        *,
        mode: ScoringMode | None = None,
    ) -> NormalizationResult:
        """归一化请求。

        Parameters
        ----------
        mode:
            显式模式；为 None 时使用 request.scoring_mode，再缺省则按 text。
            完整路由优先级请先走 QuestionTypeRouter.resolve。
        """
        warnings: list[str] = []
        resolved = mode or request.scoring_mode or ScoringMode.TEXT
        if mode is None and request.scoring_mode is None:
            warnings.append(
                "normalize 时未指定 scoring_mode，默认按 text 清洗；"
                "建议先经 QuestionTypeRouter.resolve"
            )

        if resolved is ScoringMode.TEXT:
            student = self.text_normalizer.normalize(request.student_answer)
            reference = self.text_normalizer.normalize(request.reference_answer)
            question = self.text_normalizer.normalize(request.question)
            points = [
                ScoringPoint(
                    id=p.id,
                    text=self.text_normalizer.normalize(p.text),
                    score=p.score,
                    required=p.required,
                )
                for p in request.scoring_points
            ]
        elif resolved is ScoringMode.SQL:
            student = self.sql_normalizer.normalize(request.student_answer)
            reference = self.sql_normalizer.normalize(request.reference_answer)
            question = request.question  # 题干保持原样
            points = list(request.scoring_points)
        elif resolved is ScoringMode.CALCULATION:
            # 计算题保留等式、单位和换行，只做文本层 Unicode/空白归一化。
            student = self.text_normalizer.normalize(request.student_answer)
            reference = self.text_normalizer.normalize(request.reference_answer)
            question = self.text_normalizer.normalize(request.question)
            points = list(request.scoring_points)
        else:  # CODE
            lang = request.code_language
            student = self.code_normalizer.normalize(
                request.student_answer, language=lang
            )
            reference = self.code_normalizer.normalize(
                request.reference_answer, language=lang
            )
            question = request.question
            points = list(request.scoring_points)

        updates: dict = {
            "student_answer": student,
            "reference_answer": reference,
            "question": question,
            "scoring_points": points,
        }
        if self.write_resolved_mode and request.scoring_mode is None:
            updates["scoring_mode"] = resolved

        normalized = request.model_copy(update=updates)
        return NormalizationResult(
            request=normalized,
            mode=resolved,
            warnings=warnings,
        )

    def __call__(
        self,
        request: ScoringRequest,
        *,
        mode: ScoringMode | None = None,
    ) -> NormalizationResult:
        return self.normalize(request, mode=mode)


__all__ = [
    "CodeNormalizer",
    "InputNormalizerComponent",
    "NormalizationResult",
    "SQLNormalizer",
    "TextNormalizer",
    "fullwidth_to_halfwidth",
]
