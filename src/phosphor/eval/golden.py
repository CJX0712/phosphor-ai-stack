"""Built-in golden corpus and question set.

The corpus is deliberately non uniform: paragraph lengths vary with the index
so chunk boundaries do not always land on the same heading, which is the only
way the heading inheritance logic is actually covered by the evaluation.

Every document is longer than the default chunk size (700) so the multi chunk
path is exercised rather than short-circuited.
"""

from __future__ import annotations

from ..core.types import EvalCase

CORPUS: dict[str, str] = {
    "hybrid-retrieval": """# 混合检索架构

## 为什么需要两路召回

稠密检索把问题与段落投影到同一个向量空间，能够召回语义相近但字面不重合的内容，这是 embedding 的价值所在。稀疏检索依赖词项重合，擅长精确匹配专有名词、型号与编号，例如 FAISS、HNSW、bge-small-zh 这类词。两路召回的失效模式互不重叠，因此 Phosphor 默认同时运行两路，再对结果做融合。 bm25 在大语料上的表现稳定，但在两篇文档的小语料上会因为逆文档频率退化而给出错误排序，所以稀疏分支使用 Robertson 平滑公式，保证任意词项的权重非负。

## 融合策略

Phosphor 使用 reciprocal rank fusion（RRF）做融合，公式为 score(d) = sum over branches of weight / (k + rank)，其中 k 默认取 60。之所以不用分数归一化，是因为余弦相似度与 BM25 分值处在完全不同的量纲上，任何线性归一化都会让其中一路在另一路的分布尾部被淹没；排名融合与分值无关，因此稠密后端从哈希投影换成真正的句向量模型时，融合行为保持一致。

## 重排

融合之后进入 rerank 阶段。默认实现是词法重排：以逆文档频率加权计算问题与段落的重合度，并对完整短语命中给 1.35 倍加成。安装 fastembed 后可切换为交叉编码器（bge-reranker-base），交叉编码器把问题与段落拼接后一起编码，精度更高但延迟随候选数线性增长，因此候选池控制在 top_k 的四倍以内。

## 跨语言

中文问题与英文段落之间没有字面交集，稀疏分支天生单语。Phosphor 在稀疏侧做词典查询扩展，把检索、嵌入、重排等概念映射到对方语言，同时明确不把扩展后的查询喂给稠密分支，否则两路结果会趋同，融合带来的互补性会被抵消掉。""",
    "vector-index": """# 向量索引选型

## 精确索引

内存精确余弦索引逐条计算相似度，复杂度是线性的，但在十万级以内的向量规模上延迟仍然在毫秒级。它的真正价值是作为参照实现：任何近似索引的召回率都是拿它当分母算出来的，没有这条基线，"召回没有退化"这句话就只是自证。

## HNSW

HNSW 构建多层近邻图，查询复杂度接近对数级，召回率在 0.95 以上时仍然保持亚毫秒延迟，是 CPU 场景的默认选择代价是内存占用高于原始向量，构建时间随数据量增长较快。

## IVF

IVF 先用聚类把向量空间切成若干桶，查询时只在最近的若干个桶内搜索，通过 nprobe 参数在召回率与延迟之间权衡。它对数据分布敏感，聚类数通常取向量数量的平方根附近，数据分布偏移时需要重新训练聚类中心。

## 量化

乘积量化把高维向量切成子空间并分别聚类，可以把索引压缩到原始大小的十分之一以下代价是召回率下降，通常与 HNSW 组合使用（HNSW 负责粗排，量化负责存储）。标量量化实现简单，在 CPU 上对 int8 的支持更友好，是离线批处理场景的低风险选项。""",
    "agent-orchestration": """# 智能体编排模式

## ReAct 循环

ReAct 把推理与工具调用交替进行：模型输出思考与动作，运行时执行动作并把观测追加回上下文，直到模型给出最终答案。这个循环必须有硬上界，Phosphor 默认最多六轮，超出即用已有证据兜底回答，避免模型在无法判定的问题上无限打转。

## 确定性前置路由

小参数模型在算术与单位换算上不可靠，实测 0.5B 模型会把 12 乘 (3 加 4) 算成 72。Phosphor 在问题进入模型之前用正则检出可求值的表达式，先调用计算器工具，再把结果作为观测注入上下文，模型因此不需要重新计算，只需要复述已经算好的数值。

## 规划与批评

复合问题先由规划器拆成若干子问题，子问题之间默认无依赖并行执行，执行结果交给批评器评估。批评器从两个维度打分：答案对子问题的覆盖度，以及答案被其引用证据支撑的程度。得分低于门限时重跑一轮，最多两轮。规划器与批评器都是可注入的，默认实现不依赖模型，因此在没有 API Key 的环境里整条链路依然可运行、可测量。

## 观测

每一步都会产生事件，事件携带 trace_id，收集器据此还原单次回答的完整时间线：检索耗时、工具耗时、每轮循环耗时。指标模块提供计数器、仪表与延迟分位数，全部在进程内维护，不依赖 Prometheus 之类的外部组件。""",
    "evaluation": """# 评估方法论

## 检索指标

文档级命中率衡量 top-k 结果里是否包含正确文档，这是头条指标，因为人工标注的段落级真值往往不可用，退而求其次把整篇文档的所有分块标为相关会严重低估块级召回——其中多数分块从未包含答案。文档级 MRR 进一步考察正确文档的排名位置。块级召回只作为诊断项保留，并在文档中写清口径。

## 接地指标

接地率衡量答案中有多少内容能在其引用的证据里找到支撑。计算之前必须先剥离出处标记，例如方括号里的编号与括号里的来源说明，因为引用是元数据不是断言，不剥离会让每一条答案看起来都缺乏支撑。

## 门限

门限值贴着实测基线设定，例如文档命中率实测 1.000 时门限取 0.95，这样真实回归必然失败，而浮点抖动不会误报。基线数值同时写进规格文档，做到可追溯。评估永远在全新构建的管道上运行，复用运行期已经写入数据的单例会让文档重复、指标随机漂移。

## 数值差异

numpy 路径与纯 Python 路径在末位浮点上会有差异，近似并列的候选顺序可能因此交换，而排名融合把顺序直接变成分数。两个环境都要跑，只要各自内部确定且高于门限即可，这个差异来源要在文档里主动说明。""",
    "deployment": """# 部署与推理加速

## ONNX 运行时

嵌入与重排模型走 ONNX Runtime，不需要安装 torch，CPU 上即可运行。int8 量化模型把体积压到原来的四分之一以内，精度损失通常在百分之一量级，是 CPU 部署的默认形态。线程数应当显式设置：默认按 CPU 核数减一取值的策略在小模型上会让内存带宽成为瓶颈，实测吞吐反而下降到四分之一。

## 本地推理

llama-cpp-python 提供进程内 GGUF 推理，也可以走 Ollama 的 HTTP 接口。指向本机端口的 HTTP 客户端必须关闭环境变量代理，否则本机流量被送进系统 SOCKS 代理，连接会被重置。

## 服务层

FastAPI 提供 REST 与 SSE 两种接口，OpenAPI 契约由运行中的应用直接导出，避免文档与实现漂移。所有外部依赖都做成协议加可注入实现，默认注入零依赖实现，因此在没有网络、没有密钥、没有数据库的机器上，一键验证脚本依然可以全绿。

## 可复现

版本锁定的依赖清单由干净虚拟环境生成，只装顶层钉版再冻结，得到的是最小闭包且已经验证可安装。交付前还要做一次干净克隆复现，因为本地磁盘上存在的未提交文件会掩盖清单里的遗漏。""",
}

GOLDEN: list[EvalCase] = [
    EvalCase(
        query="Phosphor 的混合检索用什么融合策略？",
        expected_substrings=("RRF", "reciprocal rank fusion", "排名融合"),
        expected_doc_ids=("hybrid-retrieval",),
        tags=("retrieval", "zh"),
    ),
    EvalCase(
        query="为什么稀疏检索要用 Robertson 平滑公式？",
        expected_substrings=("逆文档频率", "非负"),
        expected_doc_ids=("hybrid-retrieval",),
        tags=("retrieval", "zh"),
    ),
    EvalCase(
        query="CPU 场景下向量索引推荐哪一种？",
        expected_substrings=("HNSW",),
        expected_doc_ids=("vector-index",),
        tags=("index", "zh"),
    ),
    EvalCase(
        query="乘积量化的代价是什么？",
        expected_substrings=("召回率",),
        expected_doc_ids=("vector-index",),
        tags=("index", "zh"),
    ),
    EvalCase(
        query="ReAct 循环如何保证不会无限打转？",
        expected_substrings=("上界", "六轮"),
        expected_doc_ids=("agent-orchestration",),
        tags=("agent", "zh"),
    ),
    EvalCase(
        query="为什么算术要在进入模型之前先算好？",
        expected_substrings=("计算器", "观测"),
        expected_doc_ids=("agent-orchestration",),
        tags=("agent", "zh"),
    ),
    EvalCase(
        query="评估检索质量的头条指标是什么？",
        expected_substrings=("文档级命中率",),
        expected_doc_ids=("evaluation",),
        tags=("eval", "zh"),
    ),
    EvalCase(
        query="计算接地率之前必须先做什么处理？",
        expected_substrings=("剥离", "出处标记"),
        expected_doc_ids=("evaluation",),
        tags=("eval", "zh"),
    ),
    EvalCase(
        query="CPU 上小模型推理的线程数应该怎么设置？",
        expected_substrings=("线程", "内存带宽"),
        expected_doc_ids=("deployment",),
        tags=("deploy", "zh"),
    ),
    # Cross lingual: the question is English, the passage is Chinese.
    EvalCase(
        query="how does reranking work in Phosphor",
        expected_substrings=("rerank", "重排"),
        expected_doc_ids=("hybrid-retrieval",),
        tags=("retrieval", "en"),
    ),
    # Deterministic routing: no retrieval is expected at all.
    EvalCase(
        query="12*(3+4) 等于多少？",
        expected_substrings=("84",),
        expect_no_retrieval=True,
        tags=("routing", "math"),
    ),
]


def corpus_documents() -> list[tuple[str, str]]:
    return list(CORPUS.items())


def golden_cases() -> list[EvalCase]:
    return list(GOLDEN)
