# subjective-scoring

多引擎主观题自动评分库：文本评分点 + SQL AST + 代码结构/语义混合，统一 `ScoringRequest` → `ScoringResult` 接口。

适合考试系统、作业批改、本地轻量部署。编排为纯 Python（无强制 Haystack），语义通道可选 [sentence-transformers](https://www.sbert.net/) CrossEncoder。

## 特性

- **Text**：结构化评分点 × CrossEncoder/词法相似度 + 否定/数字等规则拦截
- **SQL**：`sqlglot` AST 结构比较（不走模型）
- **Code**：`tree-sitter` 结构分 + 语义分加权融合（默认 0.7 / 0.3）
- **统一契约**：置信度阈值、人工复核等级、证据点、可注入 scorer
- **可降级**：无 GPU / 无模型时词法回退，单测不下载权重

## 安装

```bash
# 核心（仅 pydantic）
pip install subjective-scoring

# 推荐：文本 + SQL + 代码
pip install "subjective-scoring[text,sql,code]"

# 启用语义 CrossEncoder（需 torch）
pip install "subjective-scoring[text,sql,code,semantic]"

# 全部
pip install "subjective-scoring[all]"
```

从 Git 安装：

```bash
pip install "subjective-scoring[text,sql,code,semantic] @ git+https://github.com/yhwyxy/subjective-scoring.git"
```

本地 editable：

```bash
uv pip install -e ".[text,sql,code,dev]"
```

## 快速开始

```python
from subjective_scoring import SubjectiveScoringService

service = SubjectiveScoringService(
    allow_model_load=False,  # CI / 离线；生产语义改为 True 并安装 [semantic]
)

result = service.score({
    "question_id": "q1",
    "max_score": 10,
    "scoring_mode": "text",
    "student_answer": "索引可以让数据库查得更快。",
    "scoring_points": [
        {"id": "p1", "text": "提高查询效率", "score": 5},
        {"id": "p2", "text": "减少全表扫描", "score": 5},
    ],
})

print(result.score, result.confidence, result.review_level, result.track)
print(result.matched_points, result.missed_points)
```

SQL / Code：

```python
service.score({
    "question_id": "s1",
    "max_score": 10,
    "scoring_mode": "sql",
    "reference_answer": "SELECT name FROM student WHERE age > 18",
    "student_answer": "select name from student where age > 18",
})

service.score({
    "question_id": "c1",
    "max_score": 10,
    "scoring_mode": "code",
    "code_language": "python",
    "reference_answer": "def f(n):\n    return sum(range(n))\n",
    "student_answer": "def f(n):\n    s=0\n    for i in range(n):\n        s+=i\n    return s\n",
})
```

## 更换 CrossEncoder 模型

**不要改库源码。** 在构造时传入模型 ID（HuggingFace）：

```python
service = SubjectiveScoringService(
    allow_model_load=True,
    text_model="BAAI/bge-reranker-base",       # 默认
    code_model="BAAI/bge-reranker-v2-m3",      # 可与 text 不同
)
```

环境变量（适合部署）：

```bash
export SUBJECTIVE_SCORING_TEXT_MODEL=BAAI/bge-reranker-v2-m3
export SUBJECTIVE_SCORING_CODE_MODEL=BAAI/bge-reranker-v2-m3
```

自定义打分函数：

```python
from subjective_scoring import TextRerankerScorer, SubjectiveScoringService, ScoringMode

def my_pairs(q: str, d: str) -> float:
    return 0.8

service = SubjectiveScoringService(
    text_scorer=TextRerankerScorer(pair_scorer=my_pairs, allow_model_load=False),
)
```

同名 `model_name` 在进程内只加载一份权重（`lru_cache`）。

## 流水线

```text
ScoringRequest
    → QuestionTypeRouter（元数据优先）
    → InputNormalizer（按 text/sql/code 差异化清洗）
    → TextRerankerScorer | SQLStructureScorer | CodeHybridScorer
    → ScoreAggregator
    → ScoringResult
```

路由优先级：`scoring_mode` → 强 `question_type` → `code_language` → `course_type` → 弱 text 类型 → 答案内容兜底 → 默认 text。

## 公共 API

```python
from subjective_scoring import (
    SubjectiveScoringService,
    create_default_service,
    ScoringRequest,
    ScoringResult,
    ScoringMode,
    ReviewLevel,
    ScoringPoint,
    ScoringOptions,
)
```

高级组件：`TextRerankerScorer`、`SQLStructureScorer`、`CodeHybridScorer`、`InputNormalizerComponent`、`QuestionTypeRouter`、`ScoreAggregatorComponent`。

## 测试

```bash
uv sync --extra text --extra sql --extra code --extra dev
uv run pytest -q
```

默认测试 **不下载** CrossEncoder 权重（`allow_model_load=False` 或注入 scorer）。

## 与考试系统集成

```python
# 业务侧只依赖本库
from subjective_scoring import SubjectiveScoringService, ScoringRequest

service = SubjectiveScoringService(allow_model_load=True, text_model="BAAI/bge-reranker-base")
result = service.score(req)
# 将 result 映射到你们的 grading_detail / 数据库字段
```

## 可选依赖说明

| extra | 用途 |
|-------|------|
| `text` | ftfy 文本归一化 |
| `sql` | sqlglot |
| `code` | tree-sitter 语言包 |
| `semantic` | sentence-transformers + torch |
| `all` | 以上全部 |
| `dev` | pytest + text/sql/code |

## License

MIT
