# LLM 评分分支设计落地方案

> 状态：定稿（核心设计；部署前置项与成本评估移至落地阶段处理）
> 范围：`subjective-scoring`（评分引擎库） + `examSystem`（评分 worker 接入）
> 版本：库 bump `v0.1.11` → `v0.1.12`

## 1. 背景与目标

当前主观题评分由 `subjective-scoring` 库的相似度引擎完成：文本题用 CrossEncoder/Reranker
逐评分点算相似度，代码题用静态分析 + 语义混合，SQL/计算题走确定性 AST/公式引擎。相似度
方案在「措辞差异大但逻辑等价」「代码写法不同但逻辑正确」等场景存在系统性压分（详见
`主观题评分误差分析.md`）。

目标：为评分系统增加 **LLM 评分分支**，在保留全部现有评分逻辑的前提下，允许部署方选择
「全部主观题走大模型 API 判分」，并将大部分改动收敛在 `subjective-scoring` 库内。

### 1.1 核心决策（已确认）

| # | 决策 | 说明 |
|---|---|---|
| D1 | **选 LLM 即全量** | 一旦启用 LLM 分支，四种主观题型（text / code / sql / calculation）**全部**走大模型评分，不做按题型分流 |
| D2 | **保留经典引擎** | 未启用 LLM 时行为与现状完全一致；经典引擎同时作为 LLM 失败时的降级后端 |
| D3 | **输出契约不变** | LLM 引擎输出与现有引擎一致的 `IntermediateScoreResult`，聚合器、`MatchedPoint/MissedPoint`、`review_status` 链路复用，前端与 Go 零改动 |
| D4 | **三层降级** | LLM 失败 → 经典引擎（可选）→ 0 分 + 强制人工复核 |
| D5 | **显式单题例外** | 请求级 `judge_backend=reranker` 可作为单题/单卷的显式回退，默认永远是 LLM |
| D6 | **开放题也走 LLM** | LLM 分支下 `scoring_points=[]` 的开放题不再强制人工，由 LLM 整题判分（低置信仍上抛人工） |
| D7 | **降级默认开** | `judge_fallback` 默认 `True`：LLM 失败静默回退经典引擎，避免整卷宕机 |

## 2. 总体架构

```text
examSystem scoring_worker
  grading.py -> grader_bridge.py -> SubjectiveScoringService
                                        |
                         router 按题型路由（TEXT/CODE/SQL/CALCULATION）
                                        |
              +---------------------------+---------------------------+
              | LLM 分支（启用时）          | 经典分支（未启用 / 降级）   |
              | LLMJudgeScorer            | TextRerankerScorer        |
              |   text / code / sql /     | CodeHybridScorer          |
              |   calculation 全题型       | SQLStructureScorer        |
              |                           | CalculationScorer         |
              +---------------------------+---------------------------+
                                        |
                              ScoreAggregatorComponent
                                        |
                              ScoringResult（契约不变）
```

## 3. 库侧改动（`subjective-scoring` v0.1.12）

### 3.1 新增 `src/subjective_scoring/engines/llm_judge.py`

新增三个公开类型 + 一个异常族：

#### `LLMJudgeConfig`（pydantic BaseModel）

```python
LLMJudgeConfig(
    url="https://router.tumuer.me/v1",   # 兼容传完整 /chat/completions 地址
    api_key="...",
    model="...",
    timeout=90.0,        # LLM 判分时延高于 reranker
    max_retries=2,
    retry_backoff_seconds=1.0,
    temperature=0.0,
    max_tokens=2048,
    json_mode=True,      # 发送 response_format={"type":"json_object"}
)
```

#### `LLMJudgeClient`

- OpenAI 兼容 `POST {url}/chat/completions`，`Authorization: Bearer <api_key>`。
- 基于 `httpx`，构造器支持注入 `client`（测试用 `MockTransport`），未注入时自持并支持
  `close()` / 上下文管理；`httpx` 归属现有 `remote` extra，**不新增依赖**。
- 重试、超时、非 2xx、JSON 解析失败均抛库内异常。
- 安全约定（沿用 `rerankers/cohere.py`）：api_key 私有存储、不进 `repr`/异常；异常不
  整段回显响应体（网关可能回显考生作答内容）。

#### `LLMJudgeScorer`（实现 `ScorerProtocol.score(request) -> IntermediateScoreResult`）

- `name = "LLMJudgeScorer"`（成为结果 `track`，前端展示无需改动）。
- 构造参数：`config`、可选注入 `client`、`fallback: ScorerProtocol | None`（降级后端）。
- 评分行为见第 4、5 节。

#### 异常族

`LLMJudgeError` / `LLMJudgeRequestError` / `LLMJudgeResponseError`（含解析失败），导出
方式对齐 `RemoteRerankerError` 系列。

### 3.2 `src/subjective_scoring/models/schemas.py`

- 新增枚举 `JudgeBackend(str, Enum)`：`AUTO("auto")` / `LLM("llm")` / `RERANKER("reranker")`。
- `ScoringOptions` 新增字段：
  ```python
  judge_backend: JudgeBackend = JudgeBackend.AUTO
  ```
  请求级选择：`auto` 跟随服务级配置；`llm` 强制 LLM；`reranker` 显式单题回退经典引擎。

### 3.3 `src/subjective_scoring/service.py`

`SubjectiveScoringService.__init__` 新增参数：

```python
llm_judge: LLMJudgeConfig | None = None   # None = 行为与现状完全一致
judge_fallback: bool = True               # LLM 失败是否回退经典引擎
```

接线规则：

- `llm_judge=None`：**零行为变化**，现有注册逻辑原样保留。
- `llm_judge` 提供时：`TEXT / CODE / SQL / CALCULATION` 四个模式全部注册
  `LLMJudgeScorer(config=..., fallback=<该模式经典引擎> if judge_fallback else None)`；
  **不提供按题型分流的 `judge_modes` 参数**（对应决策 D1）。
- 经典引擎仍可用显式注入：`text_scorer=` / `code_scorer=` / `text_pair_scorer=` /
  `code_pair_scorer=` 等参数照常生效，作为 LLM 模式的降级后端或独立使用。
- `create_default_service()` 透传新参数。

### 3.4 导出、测试、文档、版本

- `src/subjective_scoring/__init__.py` 导出 `LLMJudgeConfig` / `LLMJudgeClient` /
  `LLMJudgeScorer` / `JudgeBackend` / 异常族。
- 新增 `tests/test_llm_judge.py`（全部 `httpx.MockTransport`，不打真实 API，见第 7 节）。
- 新增设计文档 `docs/superpowers/specs/2026-08-05-llm-judge-design.md`（仿
  `2026-07-12-cohere-compatible-reranker-design.md` 格式）。
- `README.md` 增加 LLM 分支说明；版本 bump `v0.1.12` 并打 tag。

## 4. LLMJudgeScorer 评分行为

### 4.1 确定性短路（与经典引擎一致，不消耗 API）

- 空答案 → `score=0, confidence=1.0`，`decision_reason="blank_answer"`。
- 与参考答案完全一致 → 满分，`decision_reason="exact_reference_match"`。

### 4.2 Prompt 组装（按题型）

| 题型 | 额外上下文 |
|---|---|
| text | 题干 + 评分点 + 参考答案 + 学生答案 |
| code | 同 text，另带 `code_language`（python/java/...） |
| sql | 同 code，`code_language=sql`（SQL 题经归一化后路由到 SQL 模式） |
| calculation | 同 text，另带 `calculation` 配置（步骤 / 期望值 / 单位 / 容差） |

System 指令约束：严格按评分点逐点评分、只输出 JSON、分数不超单点满分、总分不超题目满分。

### 4.3 输出 JSON 契约

```json
{
  "points": [
    {
      "point_id": "p1",
      "score": 5.0,
      "relation": "supported",
      "confidence": 0.95,
      "evidence": "学生答案中的对应片段",
      "reason": "判分理由"
    }
  ],
  "notes": "整题补充说明（可选）"
}
```

`relation` 取值与现有枚举一致：`supported` / `contradicted` / `unknown`。

### 4.4 解析与校验

- 优先 `response_format={"type":"json_object"}`；网关不支持时从 content 提取首个 `{`
  到末尾 `}`，兼容 markdown 代码块包裹。
- `point_id` 必须在请求评分点内，否则该条丢弃并告警；缺失的评分点按 0 分进入
  `missed_evidence`。
- 每点评分裁剪到 `[0, point.score]`；总分封顶 `max_score`；confidence 裁剪到 `[0,1]`。
- `IntermediateScoreResult.confidence` 按评分点分值加权平均。

### 4.5 结果映射与元数据

- matched / missed → `EvidenceItem`（含 `evidence` / `reason` / `relation` /
  `relation_confidence`），聚合器照常输出 `MatchedPoint` / `MissedPoint`。
- `metadata`：`model`、`judge_backend="llm"`、`decision_reason`、`latency_ms` 等。

### 4.6 开放题（`scoring_points=[]`）处理

- 请求评分点为空时，LLMJudgeScorer 构造**单隐式评分点**参与判分：
  `point_id="whole"`，文本为「整题综合评分」，`score=max_score`。
- Prompt 中明确告知模型：无显式评分点，需对整题作答给出综合分与理由。
- 隐式点 `relation` 由得分推断：`score>0` → `supported`，`score=0` → `unknown`。
- 置信度仍按现有阈值走聚合器：低置信照常进入人工复核队列，安全网不失效。

### 4.7 强制人工复核（`force_manual_review=True`）

- 请求失败 / 解析失败 / 校验失败，且无 `fallback` 时。
- `required` 或 `critical` 评分点被判 `unknown` 或 `contradicted` 时。

## 5. 请求级选择（决策 D5）

```python
# 服务级：全量切 LLM
service = SubjectiveScoringService(
    llm_judge=LLMJudgeConfig(url="...", api_key="...", model="..."),
    judge_fallback=True,
)

# 请求级：单题/单卷显式回退经典引擎
{"scoring_config": {"judge_backend": "reranker"}}
```

## 6. 降级链与人工兜底

```text
LLM judge 失败（超时 / 4xx/5xx / JSON 解析失败 / 校验失败）
  ├─ judge_fallback=True  -> 该题回退经典引擎（warning 注明降级原因）
  │     └─ 经典引擎再失败 -> 0 分 + force_manual_review
  └─ judge_fallback=False -> 0 分 + force_manual_review（严格模式，不静默降级）
```

聚合器现有逻辑保证：`force_manual_review=True` → `review_level=MANUAL_REQUIRED` →
整卷 `need_review` 上抛人工，不因 confidence 高而被自动通过掩盖。

## 7. 测试计划

### 7.1 库侧 `tests/test_llm_judge.py`

1. Prompt 包含题干 / 评分点 / 参考答案 / 学生答案；请求头与 payload 正确。
2. 合法响应 → 正确映射 matched / missed / confidence / track。
3. 分数越界裁剪、总分封顶、未知 point_id 丢弃、缺失评分点归 0。
4. 非 JSON / 损坏 JSON / HTTP 错误 → 有 fallback 时回退；无 fallback 时
   `force_manual_review=True` 且 0 分。
5. `judge_backend=reranker` → 不发起 HTTP，直接委托经典引擎。
6. 空答案 / 完全一致答案 → 确定性短路，不发 HTTP。
7. 服务级：`SubjectiveScoringService(llm_judge=...)` 四种题型路由均到
   `LLMJudgeScorer`；`llm_judge=None` 时现有路由行为不变。
8. api_key 不进 `repr` / 异常文本。
9. 开放题（`scoring_points=[]`）：自动构造单隐式评分点，LLM 返回整题分后正确映射
   matched / confidence / review 链路。
10. 现有测试套件（含 `test_remote_reranker.py`）保持全绿。

### 7.2 examSystem 侧 `tests/worker/`

- 新增 `test_llm_judge_bridge.py`：mock LLM 响应验证 bridge 构造、降级、异常上抛；
  需为 bridge 增加 client 可注入点（或 monkeypatch）；覆盖 LLM 模式下开放题
  `scoring_points=[]` 不再短路 `open_ended` 人工、正常走 LLM 判分。
- `test_grader_bridge_parity.py`：新增 LLM 分支（设 `LLM_JUDGE_*` 时真实跑一次），
  未设置时 skip 逻辑不变。
- `test_bridge_scoring_options.py` / `test_exact_dispatch.py`：未配置 LLM 环境变量时
  行为不变，保持绿色。

## 8. examSystem 侧改动

### 8.1 依赖

- `scoring_worker/pyproject.toml`：`subjective-scoring` tag `v0.1.11` → `v0.1.12`。
- `uv lock --project scoring_worker` 重锁 `uv.lock`；无新增 Python 依赖。

### 8.2 `scoring_worker/grader_bridge.py`（核心改动，约 40~60 行）

- 新增 `validate_llm_judge_config()`，仿 `validate_remote_reranker_config()`：读
  `LLM_JUDGE_URL / LLM_JUDGE_API_KEY / LLM_JUDGE_MODEL`（可选
  `LLM_JUDGE_TIMEOUT`、`LLM_JUDGE_FALLBACK`），缺任一即报错。
- `get_subjective_service()` 增加最优先分支：LLM 配置齐全 →
  `SubjectiveScoringService(llm_judge=LLMJudgeConfig(...), judge_fallback=...)`；
  未配置时维持现有 remote reranker / local 分支，行为零变化。
- `close_service()` 增加对 `LLMJudgeClient` 的 `close()`。
- 环境变量读取沿用现有模式（直接在 bridge 读，不动 `Config` dataclass）。

### 8.3 `scoring_worker/__main__.py`（preflight 适配）

- 新增 LLM 端点连通性检查（一次真实健康调用）。
- 降级检测语义适配：LLM 模式下检测「LLM judge 失败已回退 reranker」告警。
- LLM 模式 `allow_model_load=False`，不加载本地 CrossEncoder，相关判断同步调整。

### 8.4 开放题行为

**已确认（决策 D6）**：LLM 分支下开放题也走 LLM 整题判分。

- `grader_bridge`：LLM 模式激活时，`scoring_points=[]` 不再由
  `_is_open_ended_question()` 短路为 `open_ended` 人工，而是放行给
  `LLMJudgeScorer`（库内自动构造单隐式评分点，见 4.6）。
- 非 LLM 模式维持现状（开放题强制人工）。
- 低置信 / `force_manual_review` 仍正常上抛人工，不削弱兜底。

### 8.5 环境变量与文档

- `.env.example` 增加 `LLM_JUDGE_*` 示例。
- `README.md` / `DEPLOY.md` 增加 LLM 模式配置与降级说明。

### 8.6 Go / 前端

零改动：Go 只消费 `grading_detail` JSON，`MatchedPoint/MissedPoint/review_status`
契约不变。

## 9. 落地步骤与灰度

1. `subjective-scoring` 实现 + 测试 + 设计文档，打 tag `v0.1.12`。
2. examSystem 升级依赖 + bridge 接线 + mock 测试全绿。
3. 本机指向真实网关，用一份历史试卷跑对比（LLM vs 现 reranker），核对误差与
   `review_status` 分布。
4. 单考试轮次灰度，观察成本 / 时延 / 人工复核率。
5. 确认达标后切换默认；经典引擎保留随时可回退。

## 10. 待确认事项

已确认：开放题在 LLM 分支下也走 LLM 整题判分（D6）；`judge_fallback` 默认开启（D7）。

- **网关前置**（部署侧，不阻塞设计）：`router.tumuer.me` 需开通 chat/completions
  端点；在落地步骤第 1 步（库发 tag）之前由部署侧完成。
- **请求级 `judge_backend=reranker`**：确认保留（D5），作为显式单题例外，默认永远
  是 LLM。
- **成本 / 时延评估**：SQL/计算题从毫秒级变秒级、单卷调用量上升的影响，移至第 9 节
  灰度阶段（步骤 4）用真实数据验证，不阻塞设计定稿。

> 定稿范围：核心设计（D1–D7）与两仓库改动清单。实现细节以编码时为准，如与本文冲突
> 需回更本文并说明原因。

## 11. 相关文件

- 库：`src/subjective_scoring/engines/llm_judge.py`（新）、`models/schemas.py`、
  `service.py`、`__init__.py`、`tests/test_llm_judge.py`（新）。
- 应用：`scoring_worker/grader_bridge.py`、`scoring_worker/__main__.py`、
  `scoring_worker/pyproject.toml`、`.env.example`、`README.md`、`DEPLOY.md`。
- 背景分析：`主观题评分误差分析.md`、`评分系统迭代流程.md`。
