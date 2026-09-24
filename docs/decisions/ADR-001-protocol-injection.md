# ADR-001：所有外部能力走 Protocol + 依赖注入

作者：晨星 · 状态：已接受

## 背景

嵌入、向量库、LLM、工具、存储都是外部依赖。若业务代码直接 `import faiss`、`from llama_cpp import Llama`，则：

* 单元测试必须起真实服务或打全局 monkeypatch；
* 换后端要改业务代码；
* 离线环境下整条链路无法运行，于是"能跑"和"被测过"变成两件事。

## 决策

每个外部能力定义 `typing.Protocol`（`Embedder`、`VectorStore`、`Retriever`、`Reranker`、`LLM`、`Memory`、`Planner`、`Critic`），业务模块只依赖 Protocol，具体实现由 `pipeline.py` 在装配时注入。

## 后果

正面：

* 单元测试可注入 fake，100 条单测在无网络环境下 13 秒跑完；
* 换后端 = 实现 Protocol + 在 factory 注册一行；
* 默认配置即被持续验证的配置。

代价：

* 多一层间接，需要读 `pipeline.py` 才能知道运行时到底用了什么；
* Protocol 是结构化类型，实现类的签名漂移要等运行时才暴露 —— 用 `runtime_checkable` + 模块级 `_self_check()` 缓解。
