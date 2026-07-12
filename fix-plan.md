• 建议采用“先修复现有算法，再引入执行式判题”的两阶段方案。整个开发都在独立的 subjective-scoring 仓库完成，examSystem 只通过 GitHub Tag 升级依赖。

  仓库隔离

  建议目录结构：

  /Users/yhw/Code/Github/
  ├── examSystem/
  └── subjective-scoring/

  拉取时临时使用代理，不修改全局 Git 配置：

  cd /Users/yhw/Code/Github

  HTTP_PROXY=http://127.0.0.1:6152 \
  HTTPS_PROXY=http://127.0.0.1:6152 \
  git clone https://github.com/yhwyxy/subjective-scoring.git

  cd subjective-scoring
  git switch -c fix/scoring-correctness

  配置职责保持不变：

  - subjective-scoring：评分算法、云端 Reranker 客户端、判题接口。
  - examSystem：URL、API Key、模型 ID、是否启用云端等运行配置。
  - 不发布 PyPI，通过 GitHub Tag 安装。
  - 在新版本通过测试前，examSystem 继续锁定 v0.1.1。

  ———

  ## 阶段一：现有评分器止血

  目标版本：v0.1.2

  ### 执行状态（subjective-scoring）

  - [x] 建立 text/code/sql 共享基准数据与 MAE、分档、排序指标运行器。
  - [x] 文本否定改为局部语句检测，并加入无状态领域语义处理。
  - [x] 新增本地、云端和回退后端的可注入单调分段校准组件。
  - [x] SQL 增加单语句、顶层类型硬门槛和参考维度动态权重。
  - [x] 代码修复递归检测、解析失败限分、结构冲突封顶和评分点诊断。
  - [x] 版本号更新为 0.1.2，并补充公共文档与回归测试。
  - [x] 全量测试、基准指标和 v0.1.2 wheel/sdist 构建通过。
  - [ ] 提交变更后创建并推送 v0.1.2 Tag。
  - [ ] 发布 Tag 后在 examSystem 升级依赖并重新生成锁文件。

  ### 任务 1：迁移评分基准

  把 examSystem 中已有的合成评分案例整理到独立仓库，例如：

  tests/benchmarks/
  ├── text_cases.json
  ├── code_cases.json
  └── sql_cases.json

  测试用例至少覆盖：

  - REST 完整答案。
  - REST 正确改写。
  - REST 部分答案。
  - 含“无状态”“不保存会话”等正常否定词的答案。
  - 完全无关答案。
  - 正确和错误 Python 实现。
  - 正确 SELECT 和危险 DELETE SQL。

  验收要求：

  - 测试数据不依赖 examSystem。
  - 本地词法模式和云端 Reranker 使用同一组案例。
  - 输出每种题型的 MAE、分档命中率和排序准确率。

  ### 任务 2：修复文本否定误判

  涉及：

  src/subjective_scoring/engines/text_reranker.py

  修改方向：

  - 否定检测从“扫描整份答案”改为“评分点局部匹配”。
  - 区分概念术语和语义冲突，例如“无状态”不能被视为否定答案。
  - 只有否定词与评分点核心概念位于同一局部语句时才触发冲突。
  - 否定检测结果不再直接将得分清零，应返回诊断和置信度。
  - 增加可扩展的领域术语例外机制。

  验收案例：

  “REST 通信是无状态的”             -> 不触发冲突
  “服务端不保存客户端会话状态”       -> 不触发冲突
  “REST 是有状态的”                 -> 触发概念冲突
  “REST 不要求统一接口”             -> 对统一接口评分点触发冲突

  ### 任务 3：校准 Reranker 分数

  当前问题是直接执行：

  point_score = relevance_score * point_score

  Reranker 分数代表排序相关性，不是评分概率。

  修改方向：

  - 新增独立的分数校准组件。
  - 支持阈值分段或单调映射。
  - 本地 CrossEncoder 和云端 Reranker 分别配置校准参数。
  - 原始相似度、校准后得分、否定结果都写入诊断信息。
  - 低置信度答案进入人工复核，不静默给出极端分数。

  建议首先使用简单可解释的分段映射，不引入机器学习训练流程。

  验收目标：

  - 完整 REST 答案不低于满分的 80%。
  - 正确改写不低于 70%。
  - 部分答案保持在 30%～60%。
  - 无关答案不高于 20%。
  - 正确答案不得因为“无状态”被清零。

  ### 任务 4：修复 SQL 结构评分

  涉及：

  src/subjective_scoring/engines/sql_structure.py

  修改方向：

  - 首先比较顶层语句类型。
  - 参考答案是 SELECT 时，DELETE/UPDATE/INSERT/DDL 直接得 0。
  - 不再给“双方都缺少 WHERE/HAVING/LIMIT”固定分数。
  - 仅对题目真正要求或参考答案包含的结构维度分配权重。
  - 无法解析、多语句和危险语句返回明确诊断。
  - 为后续执行式 SQL 判题预留接口。

  关键验收：

  DELETE FROM users;

  对任何 SELECT 参考答案必须为 0 分。

  ### 任务 5：修复代码结构评分

  涉及：

  src/subjective_scoring/engines/code_hybrid.py

  修改方向：

  - 递归检测必须确认函数调用的是自身函数名。
  - 不再把普通函数调用判断为递归。
  - 学生代码无法解析时限制最高分。
  - 语义相似度很高但结构明显冲突时设置分数上限。
  - 让代码题评分点实际参与评分和诊断。
  - 明确标注：没有执行测试时，结果只是“静态估分”。

  阶段一不能真正判断代码行为，因此不要承诺仅靠 Reranker 识别所有错误代码。

  ### 任务 6：发布修复版本

  完成测试后：

  UV_CACHE_DIR=/private/tmp/uv-cache-subjective uv run --extra dev pytest
  git tag -a v0.1.2 -m "fix: improve scoring correctness"
  git push origin main
  git push origin v0.1.2

  然后在 examSystem 中更新：

  subjective-scoring = {
      git = "https://github.com/yhwyxy/subjective-scoring",
      tag = "v0.1.2"
  }

  执行：

  HTTP_PROXY=http://127.0.0.1:6152 \
  HTTPS_PROXY=http://127.0.0.1:6152 \
  UV_CACHE_DIR=/private/tmp/uv-cache-subjective \
  uv lock

  ———

  ## 阶段二：执行式判题

  目标版本建议：v0.2.0

  ### 任务 7：定义统一执行判题接口

  在库中定义抽象接口：

  CodeExecutionBackend
  SqlExecutionBackend
  ExecutionResult
  TestCaseResult

  评分库只负责编排测试、聚合分数和生成诊断。

  API URL、密钥、超时、并发数仍由 examSystem 通过 .env 注入。

  ### 任务 8：接入 Judge0 代码判题

  第一期限制范围：

  - 只支持 Python。
  - 只支持函数题。
  - 使用隐藏测试验证输入输出。
  - 行为测试占 80%～90%。
  - AST 和语义诊断占 10%～20%。
  - 超时 2～5 秒。
  - 内存 128～256 MB。
  - 禁止网络。
  - 不在 FastAPI 进程中直接运行学生代码。

  关键验收：

  def total(items):
      return sum(items)

  当题目要求其他行为时，即使代码和参考答案词汇相似，也必须由隐藏测试判为低分。

  ### 任务 9：实现 SQL 结果判题

  优先支持 SQLite 内存数据库：

  1. 加载题目隐藏建表和测试数据。
  2. 执行参考 SQL。
  3. 执行学生 SQL。
  4. 标准化列名、数据类型和行顺序。
  5. 比较结果集。
  6. 使用多组隐藏数据，防止写死答案。

  安全要求：

  - 第一版仅允许单条只读 SELECT。
  - 拒绝 DDL、DML、PRAGMA 和多语句。
  - 设置执行超时、最大返回行数和数据库大小限制。
  - MySQL 特有语法后续使用隔离的 MySQL/MariaDB 服务，不强行用 SQLite 模拟。



  ## 验收指标

  阶段一目标：

  工作流成功率       100%
  文本关键案例通过率  100%
  DELETE 对 SELECT     0 分
  分档命中率          >= 75%
  排序准确率          >= 95%
  整体 MAE            <= 1.5

  阶段二目标：

  代码隐藏测试判定准确率  >= 90%
  SQL 结果判定准确率      >= 95%
  危险 SQL 拒绝率         100%
  执行超时可正确终止       100%
  评分错误不影响服务进程   100%

  最终推荐发布顺序是：

  v0.1.2  文本否定、分数校准、SQL硬门槛、代码静态修复
  v0.2.0  统一执行判题接口、SQLite结果判题
  v0.3.0  Judge0代码判题和隐藏测试

  这样每一步都有独立价值，也不会让正在运行的 examSystem 被开发中的评分库影响。
