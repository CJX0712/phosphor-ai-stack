# Phosphor AI Stack

端到端检索增强智能体平台：摄入、分块、向量化、混合检索、重排、带工具的推理、多智能体编排、评估与服务。

作者：**晨星** · 版本 v1.0.0 · 许可 MIT

---

## 一、它是什么

Phosphor 是一套**可独立验证、可组合成完整链路**的 AI 系统，不是 demo 脚本。设计上有三条硬约束：

1. **不重复造轮子。** 检索用 FAISS 与 fastembed（ONNX，免 torch），服务用 FastAPI + Uvicorn，推理接 Ollama / llama.cpp / OpenAI 兼容端点。自研只发生在"集成层与评估层"。
2. **每个外部依赖都是一个 Protocol。** 默认注入零依赖实现，且这些默认实现是**真实现不是 stub**：哈希投影嵌入、Robertson-IDF BM25、精确余弦索引、词法重排、抽取式阅读器。因此"默认配置"就是"被持续验证的配置"。
3. **一键验证是硬要求。** `git clone && pip install -r requirements.txt && python scripts/verify.py` 在无 GPU、无 API Key、无网络、无数据库的环境下必须全绿。

## 二、能力矩阵

| 能力 | 默认实现（零依赖） | 生产实现（环境变量切换） |
|---|---|---|
| 文本向量化 | BLAKE2b 有符号哈希投影 + 中英双语 token | fastembed ONNX（bge-small-zh） / OpenAI / Ollama |
| 向量索引 | 内存精确余弦（参照基线） | FAISS Flat-IP |
| 稀疏检索 | 自写 BM25（Robertson 平滑 IDF） | rank-bm25 基准对照 |
| 融合 | RRF（k=60） | 权重可调 RRF |
| 重排 | 词法重排（IDF 加权 + 短语加成） | 交叉编码器 bge-reranker-base |
| 推理 | 抽取式确定性阅读器 | Ollama / llama.cpp GGUF / OpenAI 兼容 |
| 工具 | 计算器、时钟、单位换算、JSON、知识库检索 | HTTP 抓取（需显式开启） |
| 智能体 | ReAct + 确定性前置路由 | 任意 LLM 驱动 |
| 编排 | 启发式规划器 + 词法批评器 | 模型驱动规划器/批评器 |
| 记忆 | 进程内 + JSON 持久化 | 可替换为任意后端 |
| 观测 | 计数器/仪表/延迟分位数 + trace 时间线 | OpenTelemetry 导出 |
| 服务 | FastAPI + SSE | 反向代理 / 多副本 |

## 三、模块划分（单一职责）

```
core        类型 / 配置 / 错误 / 事件总线 / 文本与数值原语   （零依赖）
io          文档加载（txt md html pdf docx）
chunk       标题继承分块（滑窗 + 句边界对齐）
embed       嵌入 Protocol + hashing / fastembed / openai / ollama
store       向量存储 Protocol + memory（精确） / faiss
lexical     BM25 + 双语词典查询扩展
retrieve    稠密 / 稀疏 / 混合(RRF) / 重排
tools       工具注册表 + 沙箱 + 算术安全求值
llm         推理 Protocol + mock / ollama / openai / llamacpp
agent       ReAct 循环 + 提示构造 + 协议解析
orch        规划 -> 执行 -> 批评 -> 综合
memory      会话记忆（进程内 / JSON 持久化）
observe     指标 + trace 收集
eval        黄金语料 + 指标 + 门限回归
api         FastAPI + SSE + OpenAPI
cli         命令行
pipeline    装配根（唯一知道全部具体实现的地方）
```

依赖方向是单向的：`core` 不依赖任何人；上层只依赖 Protocol；`pipeline` 在最后把实现注入进去。任何一个模块都能单独 import、单独测试。

## 四、快速开始

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 一句话验证全链路（8 个阶段，任一失败即短路）
python scripts/verify.py

# 命令行
python -m phosphor.cli ask "Phosphor 的混合检索用什么融合策略？"
python -m phosphor.cli search "HNSW 索引"
python -m phosphor.cli eval
python -m phosphor.cli serve --port 8080

# Web 控制台（单文件，零外部依赖）
#   浏览器打开 web/index.html，地址填 http://127.0.0.1:8080
```

启动服务后：`http://127.0.0.1:8080/docs` 是自动生成的交互式 API 文档。

## 五、一键验证到底验证了什么

`scripts/verify.py` 分八个阶段，任一阶段失败立即停止并写入 `.artifacts/verify-report.json`：

| 阶段 | 内容 |
|---|---|
| 1 p0_guard | 全仓字符门禁：无 emoji/符号字面量、无 BOM、无制表符 |
| 2 imports | 18 个模块逐个 import |
| 3 pytest | 100 条单元测试 |
| 4 http_e2e | 进程内拉起真实 ASGI 服务，22 条 HTTP 断言（含错误流） |
| 5 evaluation | 黄金集评估，对照门限 |
| 6 runtime_invariants | 9 条运行时不变量（见下） |
| 7 determinism | 两次独立构建的管道，检索结果逐位一致 |
| 8 openapi_export | 从活应用导出契约，杜绝文档漂移 |

运行时不变量（单元测试覆盖不到的系统级性质）：

| 不变量 | 为什么重要 |
|---|---|
| 长文档分块数 > 1 | 分块器真的在分块 |
| 重复摄入更短文档后分块数 == 1 | upsert 不是 replace，孤儿分块会污染检索 |
| 标题被正文继承且正文未被吞掉 | "标题 + 紧随正文"被当成标题整块丢弃过 |
| 证据渲染为单行 | 多行渲染会被逐行解析，只保留首行 |
| 全角算式 `１２×（３＋４）` == 84 | 全角数字未映射会让算术路由失效 |
| 确定性路由答案为 84 | 小模型会把 `12*(3+4)` 算成 72 |

## 六、评估基线

黄金集 11 条用例（含 1 条跨语言、1 条纯工具路由），在默认零依赖配置下：

| 指标 | 实测 | 门限 |
|---|---|---|
| 文档级命中率 | 1.000 | 0.85 |
| 文档级 MRR | 1.000 | 0.60 |
| 块级召回@6 | 0.950 | 0.55 |
| 接地率 | 1.000 | 0.75 |
| 用例通过率 | 1.000 | 0.75 |
| p50 延迟 | ~11 ms | — |

门限贴着基线设定：真实回归必然挂，浮点抖动不会挂。基线数值同步写进 `docs/SPEC.md`，可追溯。

两个环境都跑过：

| 环境 | 依赖 | 质量指标 | p50 |
|---|---|---|---|
| 开发环境 | 含 numpy / faiss / fastembed | 1.000 / 1.000 / 0.950 / 1.000 / 1.000 | ~11 ms |
| 干净房间 | 仅 `requirements.txt`（无 numpy） | 与上面完全一致 | ~26 ms |

质量指标一致、只有延迟不同，差异来源（numpy 与否改变末位浮点，进而改变 RRF 的并列顺序）在 `docs/SPEC.md` 里写清了。

## 七、配置

全部走 `PHOSPHOR_` 前缀环境变量，无配置文件也能跑：

| 变量 | 默认 | 作用 |
|---|---|---|
| `PHOSPHOR_EMBED_PROVIDER` | `hashing` | `hashing` / `fastembed` / `openai` / `ollama` |
| `PHOSPHOR_STORE_PROVIDER` | `memory` | `memory` / `faiss` |
| `PHOSPHOR_LLM_PROVIDER` | `mock` | `mock` / `ollama` / `openai` / `llamacpp` |
| `PHOSPHOR_LLM_MODEL` | `qwen2.5:0.5b-instruct` | Ollama 模型名 |
| `PHOSPHOR_LLM_BASE_URL` | `http://127.0.0.1:11434` | Ollama / OpenAI 兼容端点 |
| `PHOSPHOR_RERANK_PROVIDER` | `auto` | `auto` / `lexical` / `cross-encoder` |
| `PHOSPHOR_TOP_K` | `6` | 检索条数 |
| `PHOSPHOR_API_TOKEN` | 空 | 设置后所有 `/api/v1/*` 需 `X-API-Token` |
| `PHOSPHOR_API_PORT` | `8080` | 服务端口 |

切到真实模型只需两个变量：

```bash
export PHOSPHOR_LLM_PROVIDER=ollama
export PHOSPHOR_LLM_MODEL=qwen2.5:7b-instruct
export PHOSPHOR_EMBED_PROVIDER=fastembed
```

## 八、HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 + 组件状态 |
| POST | `/api/v1/ingest` | 摄入 text / path / directory |
| GET | `/api/v1/documents` | 文档列表 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档及其分块 |
| POST | `/api/v1/search` | 混合检索 |
| POST | `/api/v1/ask` | ReAct 问答（`orchestrate: true` 走多智能体） |
| POST | `/api/v1/stream` | SSE 流式输出 |
| GET | `/api/v1/tools` | 工具清单（含 JSON Schema） |
| POST | `/api/v1/tools/invoke` | 调用工具 |
| POST | `/api/v1/evaluate` | 跑黄金集评估 |
| GET | `/api/v1/metrics` | 计数器 / 延迟分位数 |
| GET | `/api/v1/traces/{trace_id}` | 单次回答的完整时间线 |
| GET | `/api/v1/config` | 当前配置（密钥已脱敏） |

## 九、部署

```bash
# 容器（CPU，无需 GPU）
docker build -t phosphor:1.0.0 .
docker run -p 8080:8080 phosphor:1.0.0

# 或直接跑
python -m phosphor.cli serve --host 0.0.0.0 --port 8080
```

CPU 部署注意：嵌入与重排走 ONNX Runtime；`llama-cpp-python` 线程数需显式设置，默认按核数取值的策略在小模型上会让内存带宽成为瓶颈。

## 十、文档

| 文件 | 内容 |
|---|---|
| `ARCHITECTURE.md` | 架构、模块依赖、失败模式表（症状/根因/修法/守护测试） |
| `docs/SPEC.md` | 接口契约、指标口径、评估基线 |
| `docs/openapi.yaml` | 从活代码导出的 OpenAPI 契约 |
| `docs/decisions/ADR-*.md` | 5 份架构决策记录 |

## 十一、作者

**晨星**
