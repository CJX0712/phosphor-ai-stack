# 架构说明

作者：晨星 · Phosphor AI Stack v1.0.0

## 一、分层与依赖方向

```
  +------------------------------------------------------+
  |  接入层   api (FastAPI + SSE)      cli       web      |
  +------------------------------------------------------+
                       |  只调用 pipeline 暴露的方法
  +------------------------------------------------------+
  |  装配层   pipeline      唯一知道具体实现的地方         |
  +------------------------------------------------------+
                       |  依赖注入
  +------------------------------------------------------+
  |  编排层   orch (plan -> execute -> critique)          |
  |           agent (ReAct)     memory      observe       |
  +------------------------------------------------------+
                       |  只依赖 Protocol
  +------------------------------------------------------+
  |  能力层   retrieve   tools   llm    embed    store     |
  |           lexical    chunk   io                        |
  +------------------------------------------------------+
                       |
  +------------------------------------------------------+
  |  内核     core (types config errors events text)      |
  |           零依赖，不反向依赖任何其它模块               |
  +------------------------------------------------------+
```

规则：

1. `core` 不 import 任何其它 phosphor 包。
2. 能力层各包之间只通过 `core.types` 与各自的 Protocol 交互。
3. 只有 `pipeline.py` 知道 `HashingEmbedder`、`MemoryStore`、`MockLLM` 这些具体类。
4. 新增一个后端 = 实现 Protocol + 在对应 `factory.py` 注册一行，业务代码零改动。

## 二、模块职责与接口

| 模块 | 对外接口 | 默认实现 | 生产实现 |
|---|---|---|---|
| `io` | `load_path / load_text / load_dir -> Document` | txt/md/html | pypdf |
| `chunk` | `split_document(Document, cfg) -> list[Chunk]` | 标题继承 + 句边界滑窗 | 同（token 感知可选） |
| `embed` | `Embedder.embed(texts) -> list[vec]` | HashingEmbedder | fastembed / openai / ollama |
| `store` | `VectorStore.upsert / search / drop_document` | MemoryStore（精确） | FaissStore |
| `lexical` | `BM25.index / search` | Robertson IDF | rank-bm25（对照） |
| `retrieve` | `Retriever.index / search / drop_document` | HybridRetriever(RRF) | + 交叉编码器重排 |
| `tools` | `ToolRegistry.invoke(name, args) -> ToolResult` | 6 个内建工具 | 按需注册 |
| `llm` | `LLM.complete / stream` | MockLLM（抽取式） | ollama / openai / llamacpp |
| `agent` | `ReActAgent.run(query) -> AgentAnswer` | 确定性前置路由 + 有界循环 | 大模型驱动 |
| `orch` | `Orchestrator.run(query) -> OrchestratedResult` | HeuristicPlanner + LexicalCritic | 模型驱动 |
| `memory` | `Memory.append / recent / clear` | InProcMemory + JSON | 任意后端 |
| `observe` | `Metrics / Tracer` | 进程内 | OpenTelemetry |
| `eval` | `run_evaluation(factory)` | 内建黄金集 | 自定义集 |
| `api` | `create_app(config, pipeline)` | FastAPI | — |
| `cli` | `main(argv)` | argparse | — |

## 三、关键数据流

### 摄入

```
bytes -> io.load_* -> Document
      -> chunk.split_document -> [Chunk]  (标题栈快照 + 滑窗)
      -> embedder.embed([c.text]) -> [vec]
      -> retriever.index(chunks, vectors)
          +- dense:  store.upsert(ids, vecs, meta)
          +- sparse: bm25.index(ids, heading + text)
```

摄入前先 `drop_document`：upsert 不是 replace，重复摄入更短的文档会留下上一版的孤儿分块。

### 问答

```
query +- retriever.search -> [Scored] -> render_evidence（每块压成一行）
      +- 确定性路由（算术/单位）-> ToolResult -> Observation 注入
      +- agent: while i < max_iterations:
              messages = build_messages(query, evidence, observations)
              raw = llm.complete(messages)
              if Action 且工具存在 -> invoke -> Observation -> 下一轮
              else -> parse_final(raw) -> 收敛
```

### 编排

```
query -> planner.plan -> [SubTask] -> 每个子任务独立跑 agent.run
      -> critic.review(coverage, support) -> 不达标则重跑（最多 max_rounds）
      -> synthesize（单任务直接返回，多任务分条汇总）
```

## 四、失败模式表（真踩过的坑）

每行都是一个真实 bug：症状 -> 根因 -> 修法 -> 守护它的测试。

| 症状 | 根因 | 修法 | 守护测试 |
|---|---|---|---|
| 标题 + 紧随正文整块丢失，分块数为 0 | 段落按空行切分时"标题\n正文"被判为一个标题段落 | 标题行单独成块（只更新标题栈），其后的正文另成块 | `test_heading_line_does_not_swallow_body` |
| 首块继承到下一章标题 | heading_path 取 flush 时刻的栈，而块内容起始于更早的标题下 | 缓冲**起始时**快照 `para_path`，遇新标题先 flush 再更新栈 | `test_heading_is_inherited_by_body` |
| 测试语料周期性重复，正确实现被判错 | 段落等长导致分块边界总落在同一位置 | 语料刻意非均匀（`i % n` 变化长度） | `test_non_uniform_paragraphs_vary_chunk_boundaries` |
| 两篇文档时 BM25 把正确文档排最后 | 教科书 IDF 可为负，epsilon 地板扭曲排序 | Robertson 平滑 `ln(1+(N-n+0.5)/(n+0.5))`，恒非负 | `test_idf_is_never_negative`、`test_target_document_ranks_first_on_three_documents` |
| 每个 chunk 只有首行进入模型 | 证据块多行渲染，消费方逐行解析 | `render()` 把每块压成一行，解析时续行追加 | `test_evidence_rendering_is_single_line`、`test_parse_evidence_appends_continuation_lines` |
| 抽取式答案重复同一句三次 | 语料重复导致相同句子多次入选 | 抽取后按句子去重 | `test_answer_is_de_duplicated` |
| 抽取式模型把"观测"当成问题 | 取最后一条 user 消息，而观测也是 user 轮 | 从含上下文块的 user 轮里取 `Question:` 行 | `test_question_is_taken_from_the_question_turn` |
| 最后一条 `Action:` 被当成最终答案 | 单行兜底未排除协议行 | 兜底排除 `action:/thought:/observation:` 前缀 | `test_parse_final_rejects_protocol_lines` |
| 跨语言查询检索到但答案不对 | 稀疏侧扩展了，阅读器侧没扩展 | 阅读器同样走 `expand_query` | 黄金集跨语言用例 |
| 接地率被自家引用拉低 | `(source: doc#3)` 与证据零重叠，被当成无支撑断言 | 打分前 `strip_citations` | `test_grounding_strips_citations` |
| 恒值输入的秩相关算出 +1.0 | 常量秩相关在数学上未定义 | 任一输入为常量时返回 0.0 | `test_spearman_constant_input_is_zero` |
| 工具路由用例把接地率拉低 | 设计上零证据的用例计入平均 | 接地聚合排除 `retrieved == 0` 的用例 | `test_evaluation_uses_a_fresh_pipeline_each_time` + 门限 |
| 评估指标随机漂移 | 复用运行期已写入数据的单例管道 | 评估永远新建管道 | `run_evaluation(lambda: build_pipeline(...))` |
| 重复摄入更短文档后旧内容仍可检索 | upsert 不是 replace | 摄入前 `drop_document` | 运行时不变量 `reingest_shorter_replaces_chunks` |
| 全角算式 `１２×（３＋４）` 路由失败 | 只映射了全角运算符没映射全角数字 | 归一化表补齐 U+FF10–FF19；裁剪集绝不能含运算符字符 | `test_fullwidth_arithmetic` |
| 除零异常穿透成 `ZeroDivisionError` | 安全求值只归一化了语法错误 | `ZeroDivisionError` / `OverflowError` 一并归一为 `ToolError` | `test_division_by_zero_is_tool_error_not_exception` |
| 工厂函数造出的 app 全部 404 | 路由注册留给调用方 | 工厂内部完成注册 | HTTP E2E 全路由断言 |
| `IngestReport.__dict__` 不存在 | dataclass 用了 `slots=True` | 用 `dataclasses.asdict` | `test_ingest_and_search` |
| 本机 Ollama 调用被重置（WinError 10054） | httpx 默认 `trust_env=True`，本机流量被送进 SOCKS 代理 | 所有 localhost 客户端 `trust_env=False` | 代码约定 + E2E |
| 冷启动因重排模型下载失败而中断 | 交叉编码器懒加载，失败发生在第一次检索 | 构建时 warmup 探测，失败静默回退词法重排 | `build_reranker` 回退路径 |

## 五、为什么默认实现必须是真实现

如果默认实现是 stub，那么"默认配置"就是一个从未被真正执行过的配置，所有测试都在 fakes 上跑，生产路径第一次被执行是在生产环境。

Phosphor 的做法是给每个外部端口配一个**零依赖的真实实现**：

* 哈希投影嵌入是真嵌入 —— 余弦相似度随词面重合度变化，混合检索的两路分支因此都能被真正锻炼；
* BM25 是真 BM25 —— 黄金集的排序结论是排序能力的证据；
* 抽取式阅读器是真阅读器 —— 接地率衡量的是"答案是否来自证据"，而不是"答案是否为空字符串"。

代价（哈希嵌入没有语义泛化能力）在 `docs/SPEC.md` 里写清楚，并且换到真模型只是一个环境变量。
