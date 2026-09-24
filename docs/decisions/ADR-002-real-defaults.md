# ADR-002：默认实现必须是真实现，不能是 stub

作者：晨星 · 状态：已接受

## 背景

常见做法是默认实现返回空字符串或抛 `NotImplementedError`，测试时用 fixture 换成真东西。结果是：默认配置从未真正执行过，测试在 fakes 上跑，生产路径第一次被执行是在生产环境。

## 决策

每个外部端口都配一个**零依赖的真实实现**：

| 端口 | 默认实现 | 真实之处 |
|---|---|---|
| 嵌入 | HashingEmbedder | BLAKE2b 有符号投影，余弦随词面重合度变化 |
| 稀疏检索 | BM25（Robertson IDF） | 完整的 tf-idf 打分与排序 |
| 向量索引 | MemoryStore | 精确余弦，可当近似索引的召回基线 |
| 重排 | LexicalReranker | IDF 加权重合度 + 短语加成 |
| 推理 | MockLLM | 抽取式阅读，带引用与去重，拒绝编造 |
| 规划/批评 | HeuristicPlanner / LexicalCritic | 真实拆分与真实支撑度打分 |

## 后果

* `git clone && python scripts/verify.py` 在裸 CPython 上全绿，无需下载任何模型；
* 黄金集评估在默认配置下给出有意义的数字（命中率 1.000 / MRR 1.000 / 接地率 1.000）；
* 代价是这些实现的能力边界必须写进文档（`docs/SPEC.md` 第四节），否则用户会误以为哈希嵌入等于句向量。
