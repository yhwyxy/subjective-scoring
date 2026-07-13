# subjective-scoring

多引擎主观题自动评分库：文本评分点 + SQL AST + 代码结构/语义混合，统一 `ScoringRequest` → `ScoringResult` 接口。

适合考试系统、作业批改、本地轻量部署。编排为纯 Python（无强制 Haystack），语义通道可选 [sentence-transformers](https://www.sbert.net/) CrossEncoder。

## 特性

- **Text**：原子评分点 × 三态关系（支持/冲突/不确定）+ 局部否定/数字规则 + 拒判复核
- **SQL**：`sqlglot` AST 结构比较，单语句与顶层类型硬门槛（不走模型）
- **Code**：`tree-sitter` 结构分 + 语义分加权融合；明确标记为静态估分
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

# 云端 Cohere-compatible Reranker（无需本地模型权重）
pip install "subjective-scoring[text,sql,code,remote]"

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
        {
            "id": "p2",
            "text": "减少全表扫描",
            "score": 5,
            "critical": True,
            "conflict_policy": "cap_total",
            "conflict_score_cap_ratio": 0.4,
        },
    ],
    "scoring_config": {
        "text_relation_thresholds": {
            "support": 0.55,
            "conflict": 0.8,
            "reject_when_no_supported": True,
        }
    },
})

print(
    result.score,
    result.provisional_score,
    result.decision,
    result.review_level,
    result.track,
)
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

## 原子评分点、三态关系与拒判

文本题会独立判定每个评分点与学生答案的关系：

- `supported`：达到支持阈值且没有硬冲突，按校准覆盖度获得该点分数；
- `contradicted`：高置信否定、方向或关键数字冲突，该点为 0 分并要求复核；
- `unknown`：证据不足或冲突置信度不够，该点不计分。

评分前会先把答案拆成局部语句及相邻窗口，为每个原子评分点选择最相关证据；否定、数字和单位规则只检查该局部证据，避免答案其他段落中的“不”“没有”等词误伤当前评分点。

`required` 表示必答知识点，`critical` 表示关键结论。关键点可通过 `conflict_policy` 配置为 `point_zero`、`cap_total` 或 `zero_total`。如果没有任何评分点被可靠支持但仍存在 `unknown`，系统返回 `decision=manual_review`；正式 `score` 不包含未知点，但 `provisional_score` 会保留待复核估分，调用方不得把它直接写入最终成绩。

默认还会使用标准答案验证 required / critical 评分点。如果标准答案本身无法可靠支持评分点，结果会标记 `rubric_self_check_failed` 并进入人工复核。支持阈值按本地 CrossEncoder、云端 Reranker 和词法回退分别配置；显式设置 `support` 时仍可统一覆盖所有后端。

评分点应保持原子化：一项只描述一个可独立验证的结论。例如 REST 的资源导向、HTTP 方法和无状态应拆成三个评分点，而不是合成一个复合评分点。

## 分数校准与诊断

Reranker 返回的是排序相关度，并不直接等同于得分比例。文本评分会先按后端应用单调分段校准，再执行局部规则调整。内置配置区分本地 CrossEncoder、`cohere:*` 云端 Reranker 与 lexical/injected 后端。

需要覆盖默认曲线时可注入校准器：

```python
from subjective_scoring import PiecewiseLinearCalibrator, SubjectiveScoringService

service = SubjectiveScoringService(
    text_calibrator=PiecewiseLinearCalibrator(
        [(0.0, 0.0), (0.2, 0.6), (0.5, 0.9), (1.0, 1.0)],
        name="custom-v1",
    ),
)
```

启用 trace 后，文本中间结果的 `metadata.point_diagnostics` 会记录每个评分点的原始相关度、校准覆盖度、三态关系、关系置信度、规则命中证据和调整后置信度。

## 云端 Reranker

无法下载或运行本地 CrossEncoder 时，可以注入兼容 Cohere `/rerank` 协议的云端服务。URL、API Key 和模型 ID 应由应用环境提供，不要写入库源码或提交到 Git。

```python
import os

from subjective_scoring import (
    CohereRerankerPairScorer,
    SubjectiveScoringService,
)

reranker = CohereRerankerPairScorer(
    url=os.environ["RERANK_API_URL"],
    api_key=os.environ["RERANK_API_KEY"],
    model=os.environ["RERANK_MODEL"],
    timeout=30.0,
)

service = SubjectiveScoringService(
    allow_model_load=False,
    text_pair_scorer=reranker,
    code_pair_scorer=reranker,
)
```

适配器会把同一 query 的 documents 合并为一次请求，自动设置 `top_n`，并根据响应中的 `index` 恢复输入顺序。远端请求或响应失败时会抛出明确异常，由评分管线转为 0 分并要求人工复核，不会静默切换到词法相似度。

请求结构：

```json
{
  "model": "Pro/BAAI/bge-reranker-v2-m3",
  "query": "student answer",
  "documents": ["point one", "point two"],
  "top_n": 2,
  "return_documents": false
}
```

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

限制：第一阶段不会执行学生代码或 SQL。代码结果的中间元数据包含 `assessment_type=static_estimate`；无法静态验证的行为评分点会要求后续执行测试或人工复核。SQL 第一阶段只自动比较单条 `SELECT`，DML、DDL、多语句和解析失败答案为 0 分并要求复核。

## 测试

```bash
UV_CACHE_DIR=/private/tmp/uv-cache-subjective uv sync --extra dev
UV_CACHE_DIR=/private/tmp/uv-cache-subjective uv run --extra dev pytest -q
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
| `remote` | httpx + Cohere-compatible 云端 Reranker |
| `all` | 以上全部 |
| `dev` | pytest + text/sql/code/remote |

## License

MIT
