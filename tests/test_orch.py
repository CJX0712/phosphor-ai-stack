from phosphor.agent import ReActAgent
from phosphor.core.config import AgentConfig, OrchestrationConfig
from phosphor.core.types import Chunk, Scored
from phosphor.llm import MockLLM
from phosphor.orch import HeuristicPlanner, LexicalCritic, Orchestrator
from phosphor.orch.protocol import TaskResult
from phosphor.tools import ToolRegistry, register_builtins


class FakeRetriever:
    def __init__(self) -> None:
        self.chunks = [
            Chunk(id="c1", doc_id="d1", text="混合检索使用 reciprocal rank fusion 融合两路召回。"),
            Chunk(id="c2", doc_id="d2", text="HNSW 是 CPU 场景的默认向量索引。"),
        ]

    def search(self, query, k=6, trace_id=""):
        return [Scored(chunk=c, score=1.0, source="fake") for c in self.chunks][:k]

    def drop_document(self, doc_id):
        return 0


def build_agent() -> ReActAgent:
    retriever = FakeRetriever()
    tools = register_builtins(ToolRegistry(), retriever=retriever)
    return ReActAgent(MockLLM(), retriever, tools, AgentConfig())


def test_planner_keeps_single_question_intact():
    plan = HeuristicPlanner().plan("混合检索用什么融合策略？")
    assert plan.size == 1
    assert plan.strategy == "single"


def test_planner_splits_compound_question():
    plan = HeuristicPlanner().plan("融合策略是什么，以及向量索引推荐哪一种？")
    assert plan.size >= 2
    assert plan.strategy == "decompose"


def test_critic_rejects_empty_answers():
    from phosphor.orch.protocol import SubTask

    task = SubTask(id="t", question="问题")
    critique = LexicalCritic().review("问题", [TaskResult(task=task, answer="", ok=False)])
    assert not critique.accepted
    assert critique.issues


def test_critic_accepts_supported_answer():
    from phosphor.core.types import Chunk
    from phosphor.orch.protocol import SubTask

    task = SubTask(id="t", question="融合策略是什么")
    result = TaskResult(
        task=task,
        answer="使用 reciprocal rank fusion 融合",
        evidence=[Scored(chunk=Chunk(id="c", doc_id="d", text="reciprocal rank fusion 融合策略"), score=1.0)],
    )
    critique = LexicalCritic().review("融合策略是什么", [result])
    assert critique.score > 0


def test_orchestrator_runs_and_synthesises():
    orch = Orchestrator(build_agent(), OrchestrationConfig(max_rounds=2))
    result = orch.run("融合策略是什么，以及向量索引推荐哪一种？")
    assert result.answer
    assert result.rounds >= 1
    assert result.trace_id
