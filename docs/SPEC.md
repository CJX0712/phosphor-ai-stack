# 接口契约与评估口径

作者：晨星 · Phosphor AI Stack v1.0.0

## 一、核心类型契约

| 类型 | 字段 | 不变式 |
|---|---|---|
| `Document` | id, title, text, source, meta | `id` 由内容派生，同内容同 id |
| `Chunk` | id, doc_id, text, order, start, end, heading_path, meta | `render()` 输出**单行**；`heading_path` 是块起始时的标题栈快照 |
| `Scored` | chunk, score, source, detail | `detail` 至少含 `dense_rank` 或 `sparse_rank` 之一 |
| `AgentAnswer` | answer, steps, evidence, tool_calls, iterations, converged, trace_id | 有观测即必须收敛；`iterations <= max_iterations` |
| `ToolResult` | call, output, ok, error, elapsed_ms | 失败也不抛异常，`ok=False` + `error` |
| `IngestReport` | doc_id, title, chunks, skipped, reason | 摄入前先 drop，`chunks` 反映本次结果而非累计 |
| `EvalReport` | 指标 + thresholds + violations | `ok == (not violations)` |

## 二、指标口径

| 指标 | 定义 | 为什么这样定 |
|---|---|---|
| 文档级命中率 | top-k 中是否出现正确文档（0/1 平均） | 头条指标。人工段落级标注不可用，把整篇文档的所有分块标为相关会低估块级召回 |
| 文档级 MRR | 正确文档首次出现排名的倒数 | 考察排名位置，比命中率更敏感 |
| 块级召回@k | 期望子串在 top-k 文本中的命中比例 | 诊断项，不作为头条指标 |
| 接地率 | 剥离出处标记后，答案词集与引用证据词集的余弦，达到 0.2 阈值记 1.0 | 引用是元数据不是断言，不剥离会把每条答案都判成无支撑 |
| 通过率 | 命中 + 答案含期望子串 + 接地 ≥ 0.5 | 三个维度同时成立才算通过 |

聚合规则：

* 检索类聚合（`doc_hit` / `mrr` / `recall`）**排除** `expect_no_retrieval` 用例（工具路由用例设计上零证据，计入会拉低平均值），单独统计 `retrieval_cases`。
* 接地率聚合排除 `retrieved == 0` 的用例。

## 三、评估基线（默认零依赖配置）

| 指标 | 实测 | 门限 |
|---|---|---|
| 文档级命中率 | 1.000 | 0.85 |
| 文档级 MRR | 1.000 | 0.60 |
| 块级召回@6 | 0.950 | 0.55 |
| 接地率 | 1.000 | 0.75 |
| 通过率 | 1.000 | 0.75 |
| p50 延迟 | ~11 ms（CPU，无 GPU） | — |

门限贴着基线设定：真实回归必然失败，浮点抖动不会误报。

### 两个环境的实测对照

同一台机器（Windows，AMD Ryzen，无 GPU）：

| 环境 | 依赖 | doc_hit | MRR | recall@6 | 接地率 | 通过率 | p50 |
|---|---|---|---|---|---|---|---|
| 开发环境 | 含 numpy / faiss / fastembed | 1.000 | 1.000 | 0.950 | 1.000 | 1.000 | ~11 ms |
| 干净房间 | 仅 requirements.txt（无 numpy） | 1.000 | 1.000 | 0.950 | 1.000 | 1.000 | ~26 ms |

**质量指标两环境完全一致**（1.000 / 1.000 / 0.950 / 1.000 / 1.000），差异只在延迟：numpy 向量化与否改变了计算耗时。

机制说明：numpy 路径与纯 Python 路径在末位浮点上存在差异，近似并列的候选顺序可能因此交换，而 RRF 把顺序直接变成分数，所以两边的**分数不逐位相同**，但各自内部确定、都高于门限。这就是文档里同时给出两列的原因 —— 只报一个数字会掩盖差异来源。

## 四、默认实现的已知边界

| 默认实现 | 能力 | 边界 |
|---|---|---|
| HashingEmbedder | 词面重合度近似语义相似度，双语可用 | **没有语义泛化**：同义改写、跨语言（未经词典扩展的概念）召回弱于真句向量 |
| MockLLM | 从证据中抽取并引用，拒绝编造 | **不是语言模型**：不改写、不推理、不做多跳综合，只用默认配置衡量检索与接地，不衡量生成质量 |
| LexicalReranker | IDF 加权重合度 + 短语加成 | 弱于交叉编码器，但零依赖、零延迟成本 |

切换到真实模型只需：

```bash
export PHOSPHOR_LLM_PROVIDER=ollama
export PHOSPHOR_EMBED_PROVIDER=fastembed
```

## 五、HTTP 契约

契约由活代码生成（`create_app().openapi()` -> `docs/openapi.yaml`），杜绝文档与实现漂移。

| 方法 | 路径 | 认证 |
|---|---|---|
| GET | `/health` | 否 |
| POST | `/api/v1/ingest` | 是（当 `PHOSPHOR_API_TOKEN` 非空） |
| GET | `/api/v1/documents` | 是 |
| DELETE | `/api/v1/documents/{doc_id}` | 是 |
| POST | `/api/v1/search` | 是 |
| POST | `/api/v1/ask` | 是 |
| POST | `/api/v1/stream` | 是 |
| GET | `/api/v1/tools` | 是 |
| POST | `/api/v1/tools/invoke` | 是 |
| POST | `/api/v1/evaluate` | 是 |
| GET | `/api/v1/metrics` | 是 |
| GET | `/api/v1/traces/{trace_id}` | 是 |
| GET | `/api/v1/config` | 是 |

## 六、配置契约

全部环境变量以 `PHOSPHOR_` 为前缀，由 `core/config.py` 统一读取，其它模块不得直接读 `os.environ`。

| 变量 | 默认 | 取值 |
|---|---|---|
| `PHOSPHOR_ENV` | `dev` | 任意字符串 |
| `PHOSPHOR_CHUNK_SIZE` | `700` | 正整数 |
| `PHOSPHOR_CHUNK_OVERLAP` | `120` | 非负整数，须 < size |
| `PHOSPHOR_EMBED_PROVIDER` | `hashing` | hashing / fastembed / openai / ollama |
| `PHOSPHOR_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 字符串 |
| `PHOSPHOR_EMBED_DIM` | `384` | 正整数 |
| `PHOSPHOR_STORE_PROVIDER` | `memory` | memory / faiss |
| `PHOSPHOR_STORE_PATH` | `.cache/index` | 路径 |
| `PHOSPHOR_TOP_K` | `6` | 正整数 |
| `PHOSPHOR_RERANK` | `true` | bool |
| `PHOSPHOR_RERANK_PROVIDER` | `auto` | auto / lexical / cross-encoder |
| `PHOSPHOR_LLM_PROVIDER` | `mock` | mock / ollama / openai / llamacpp |
| `PHOSPHOR_LLM_MODEL` | `qwen2.5:0.5b-instruct` | 字符串 |
| `PHOSPHOR_LLM_BASE_URL` | `http://127.0.0.1:11434` | URL |
| `PHOSPHOR_LLM_MODEL_PATH` | 空 | GGUF 路径（llamacpp） |
| `PHOSPHOR_AGENT_MAX_ITER` | `6` | 正整数 |
| `PHOSPHOR_ORCH_ENABLED` | `true` | bool |
| `PHOSPHOR_API_HOST` | `127.0.0.1` | 地址 |
| `PHOSPHOR_API_PORT` | `8080` | 端口 |
| `PHOSPHOR_API_TOKEN` | 空 | 非空则启用鉴权 |
| `PHOSPHOR_DATA_DIR` | `.cache/phosphor` | 路径 |
| `PHOSPHOR_TELEMETRY` | `true` | bool |

## 七、验证契约

`python scripts/verify.py` 八阶段，任一阶段失败即短路，报告落在 `.artifacts/verify-report.json`：

1. `p0_guard` — 全仓字符门禁
2. `imports` — 18 个模块
3. `pytest` — 单元测试
4. `http_e2e` — 22 条 HTTP 断言（含错误流）
5. `evaluation` — 黄金集门限
6. `runtime_invariants` — 9 条系统级不变量
7. `determinism` — 跨管道实例结果一致
8. `openapi_export` — 契约导出
