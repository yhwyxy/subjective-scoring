"""关键实体抽取与命中判定：文本题有界修正（实体门槛 / 数值与短答案保底）的共享基础。

三条修正共用同一套实体命中判定，保证互斥性：
- 实体门槛只在"零命中"时压分；
- 数值 / 短答案保底只在"命中"时抬分；
两者不可能同时触发，因此每条规则只在自己的触发区起作用且单调。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# 负号仅在前面不是数字时生效："0-25" 是范围写法，应解析为 0 和 25 而非 -25
_NUMBER_RE = re.compile(r"(?<!\d)-?\d+(?:\.\d+)?%?")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")
_CJK_RUN_RE = re.compile(r"[一-鿿]+")
# 分句开头的枚举序号（"1. " / "2、" / "3)"）不是内容数值
_ENUMERATOR_RE = re.compile(r"(?:^|(?<=[\n。；;！？!?]))\s*\d{1,2}\s*[.、)）]")

# 拉丁 token 中不作为术语实体的：计量单位与英文虚词
_LATIN_UNITS = frozenset(
    "ml l mol kg mg g ms s mb gb kb mm cm km m ma kv mv v a hz khz mpa kpa pa "
    "min h ppm ppb rpm mah kw w".split()
)
_LATIN_STOPWORDS = frozenset(
    "the and or of to in for with is are was were not no a an on at by it this "
    "that be as if then else do does can could should would".split()
)

# 通用套话词：任何答案里都常见，命中不代表答对了内容，不作为关键实体
_GENERIC_TERMS = frozenset(
    (
        "处理 规范 操作 正确 及时 要求 注意 严格 认真 按照 标准 相关 环保 安全 "
        "管理 制度 流程 规定 仔细 做好 保证 确保 符合 合格 合理 有效 正常 问题 "
        "情况 工作 内容 方法 方式 措施 原则 首先 然后 最后 其次 需要 必须 应该 "
        "可以 使用 进行 完成 检查 培训 学习 提高 加强 重视 落实 执行 遵守 遵循 "
        "责任 意识 质量 效率 效果 目的 目标 作用 影响 重要 主要 基本 一般 通常 "
        "保持 良好 到位 严禁 禁止 避免 不得 不能 不要 定期 经常 随时 立即 马上 "
        "现场 岗位 人员 部门 单位 公司 领导 汇报 上报 报告 发现 出现 发生 存在 "
        "如果 由于 因为 所以 因此 同时 以及 或者 并且 而且 通过 采用 根据 依据"
    ).split()
)

# 切分术语块的单字虚词：刻意保守，凡可能出现在专业术语内部的字（应/和/中/能/为/按/对/当…）都不列入
_STOP_CHARS = "的了是在将把被给让向从至并且而则若如之或及与就还但才都很较更最呢吗啊也均即等来去要请"
# 多字虚词优先切分（长词在前）
_STOP_WORDS_RE = re.compile(
    "|".join(
        (
            "应当", "应该", "必须", "需要", "可以", "能够", "以及", "或者",
            "并且", "而且", "然后", "首先", "其次", "最后", "通过", "使用",
            "进行", "采用", "保持", "确保", "保证", "做到", "要求", "注意",
            "避免", "禁止", "不得", "不能", "不要", "严禁", "时候", "根据",
            "依据", "对于", "关于", "以便", "以免", "同时", "此外", "另外",
            "[" + _STOP_CHARS + "]",
        )
    )
)

_CN_DIGITS = {
    "零": "0", "一": "1", "两": "2", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
}
_MEASURE_WORDS = "次遍个种步条项点层级滴份台部位名组道"
_CN_MEASURE_RE = re.compile("([零一两二三四五六七八九十])([" + _MEASURE_WORDS + "])")
# 程度填充字（"过冷度越大" vs "过冷度增大"）会打断 bigram 匹配；只在后接
# 常见单字形容词时删除，保留"越权/越级/更换"等实义词
_DEGREE_FILLER_RE = re.compile(
    "[越更愈](?=[大小快慢高低多少长短粗细好差深浅厚薄强弱早晚密疏])"
)

# 常见同义 / 等价表述：命中任意变体即视为命中该实体
DEFAULT_EQUIVALENCES: tuple[tuple[str, ...], ...] = (
    ("校准", "标定", "校正"),
    ("蒸馏水", "纯水", "去离子水", "纯化水"),
    ("归零", "调零", "回零", "清零", "置零"),
    ("清洗", "冲洗", "洗涤", "润洗", "清洁"),
    ("干燥", "烘干", "晾干", "吹干"),
    ("记录", "登记", "记载", "填写"),
    ("佩戴", "穿戴", "戴好", "戴上"),
    ("通风", "排风", "换气"),
    ("断电", "切断电源", "关闭电源", "停电"),
    ("标签", "标识", "标示", "贴标"),
    ("复核", "复查", "核对", "校核", "复检"),
    ("维护", "保养", "检修"),
    ("泄漏", "泄露", "渗漏", "漏液", "漏气"),
    ("废液桶", "废液缸", "废液瓶", "收集桶"),
    ("护目镜", "防护镜", "防护眼镜"),
    ("手套", "防护手套"),
    ("灭火器", "消防器材"),
    ("酸碱中和", "中和"),
    ("样品", "试样", "样本"),
    ("试剂", "药品", "药剂"),
    ("真值", "真实值", "实际值"),
    ("示值", "测量值", "指示值", "读数"),
    ("量程", "满量程", "测量范围"),
)


def has_cjk(text: str) -> bool:
    """实体启发式面向中文语料；无 CJK 字符的评分点不参与有界修正。"""
    return bool(_CJK_RUN_RE.search(str(text or "")))


def normalize_entity_text(text: str) -> str:
    """NFKC + casefold + 汉字量词数字化（三次→3次），供实体抽取与命中共用。"""
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    value = _CN_MEASURE_RE.sub(
        lambda m: _CN_DIGITS[m.group(1)] + m.group(2), value
    )
    value = _DEGREE_FILLER_RE.sub("", value)
    return _ENUMERATOR_RE.sub(" ", value)


def _number_variants(token: str) -> frozenset[str]:
    """一个数字 token 的等价变体集：40% ≡ 40 ≡ 0.4。"""
    variants = {token}
    if token.endswith("%"):
        bare = token[:-1]
        variants.add(bare)
        try:
            variants.add(_format_number(float(bare) / 100.0))
        except ValueError:
            pass
    else:
        try:
            variants.add(_format_number(float(token)))
        except ValueError:
            pass
    return frozenset(variants)


def _format_number(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def extract_number_variant_sets(normalized_text: str) -> tuple[frozenset[str], ...]:
    """抽取文本中的数字实体，去重后每个实体带等价变体集。"""
    seen: dict[frozenset[str], None] = {}
    for match in _NUMBER_RE.finditer(normalized_text):
        seen.setdefault(_number_variants(match.group(0)), None)
    return tuple(seen)


@dataclass(frozen=True)
class TermEntity:
    """一个术语实体：surface 为评分点原文块，variants 含等价表述。"""

    surface: str
    variants: tuple[str, ...]


@dataclass(frozen=True)
class EntityHits:
    """学生答案对某评分点实体集的命中结果。"""

    term_hits: int
    term_total: int
    number_hits: int
    number_total: int
    informative_number_hits: int
    informative_number_total: int

    @property
    def total(self) -> int:
        return self.term_hits + self.number_hits

    @property
    def entity_total(self) -> int:
        return self.term_total + self.number_total

    @property
    def coverage(self) -> float:
        if self.entity_total == 0:
            return 0.0
        return self.total / self.entity_total

    @property
    def numbers_complete(self) -> bool:
        """点内数字全部命中（无数字时视为满足）。"""
        return self.number_hits == self.number_total

    @property
    def informative_numbers_complete(self) -> bool:
        """存在题干之外的数字且全部命中——数值保底的触发条件。"""
        return (
            self.informative_number_total > 0
            and self.informative_number_hits == self.informative_number_total
        )


def _bigram_coverage(term: str, student: str) -> float:
    grams = [term[i : i + 2] for i in range(len(term) - 1)]
    if not grams:
        return 1.0 if term and term in student else 0.0
    hit = sum(1 for gram in grams if gram in student)
    return hit / len(grams)


def _term_hit(entity: TermEntity, student_normalized: str) -> bool:
    for variant in entity.variants:
        if len(variant) <= 3:
            if variant in student_normalized:
                return True
        elif variant in student_normalized or _bigram_coverage(
            variant, student_normalized
        ) >= 0.6:
            return True
    return False


@dataclass(frozen=True)
class PointEntityProfile:
    """评分点的关键实体档案：术语 + 数字（区分题干已出现的数字）。"""

    terms: tuple[TermEntity, ...]
    numbers: tuple[frozenset[str], ...]
    informative_numbers: tuple[frozenset[str], ...]
    primarily_numeric: bool

    @property
    def entity_count(self) -> int:
        return len(self.terms) + len(self.numbers)

    def match(
        self,
        student_normalized: str,
        student_number_variants: frozenset[str],
    ) -> EntityHits:
        term_hits = sum(
            1 for term in self.terms if _term_hit(term, student_normalized)
        )
        number_hits = sum(
            1
            for variants in self.numbers
            if variants & student_number_variants
        )
        informative_hits = sum(
            1
            for variants in self.informative_numbers
            if variants & student_number_variants
        )
        return EntityHits(
            term_hits=term_hits,
            term_total=len(self.terms),
            number_hits=number_hits,
            number_total=len(self.numbers),
            informative_number_hits=informative_hits,
            informative_number_total=len(self.informative_numbers),
        )


def _extract_term_entities(
    normalized_point: str,
    equivalences: tuple[tuple[str, ...], ...],
) -> tuple[TermEntity, ...]:
    entities: dict[str, TermEntity] = {}

    for token in _LATIN_TOKEN_RE.findall(normalized_point):
        if len(token) < 2:
            continue
        if token in _LATIN_UNITS or token in _LATIN_STOPWORDS:
            continue
        entities.setdefault(token, TermEntity(surface=token, variants=(token,)))

    for run in _CJK_RUN_RE.findall(normalized_point):
        for chunk in _STOP_WORDS_RE.split(run):
            if len(chunk) < 2 or chunk in _GENERIC_TERMS:
                continue
            entities.setdefault(chunk, TermEntity(surface=chunk, variants=(chunk,)))

    for group in equivalences:
        matched = [v for v in group if v in normalized_point]
        if not matched:
            continue
        variants = tuple(dict.fromkeys(group))
        for surface, entity in list(entities.items()):
            if any(v in entity.surface for v in matched):
                merged = tuple(dict.fromkeys((*entity.variants, *variants)))
                entities[surface] = TermEntity(surface=surface, variants=merged)

    return tuple(entities.values())


def _content_char_count(normalized_point: str) -> int:
    """去掉数字 / 单位 / 虚词后剩余的内容字符数，用于判断"以数值为主"。"""
    value = _NUMBER_RE.sub(" ", normalized_point)
    kept_latin = [
        token
        for token in _LATIN_TOKEN_RE.findall(value)
        if len(token) >= 2
        and token not in _LATIN_UNITS
        and token not in _LATIN_STOPWORDS
    ]
    count = sum(len(token) for token in kept_latin)
    for run in _CJK_RUN_RE.findall(value):
        for chunk in _STOP_WORDS_RE.split(run):
            count += len(chunk)
    return count


def build_point_entity_profile(
    point_text: str,
    question_text: str = "",
    extra_equivalences: tuple[tuple[str, ...], ...] = (),
) -> PointEntityProfile:
    normalized_point = normalize_entity_text(point_text)
    equivalences = (*DEFAULT_EQUIVALENCES, *extra_equivalences)
    terms = _extract_term_entities(normalized_point, equivalences)
    numbers = extract_number_variant_sets(normalized_point)

    stem_variants: frozenset[str] = frozenset()
    if question_text:
        stem_numbers = extract_number_variant_sets(
            normalize_entity_text(question_text)
        )
        stem_variants = frozenset().union(*stem_numbers) if stem_numbers else frozenset()
    informative = tuple(
        variants for variants in numbers if not (variants & stem_variants)
    )

    primarily_numeric = bool(numbers) and _content_char_count(normalized_point) <= 12
    return PointEntityProfile(
        terms=terms,
        numbers=numbers,
        informative_numbers=informative,
        primarily_numeric=primarily_numeric,
    )


def student_entity_view(student_answer: str) -> tuple[str, frozenset[str]]:
    """学生答案的实体视角：归一化全文 + 展开后的数字变体全集。"""
    normalized = normalize_entity_text(student_answer)
    number_sets = extract_number_variant_sets(normalized)
    variants = frozenset().union(*number_sets) if number_sets else frozenset()
    return normalized, variants


__all__ = [
    "DEFAULT_EQUIVALENCES",
    "EntityHits",
    "PointEntityProfile",
    "TermEntity",
    "build_point_entity_profile",
    "extract_number_variant_sets",
    "has_cjk",
    "normalize_entity_text",
    "student_entity_view",
]
