from phosphor.agent import ReActAgent, parse_action, parse_final
from phosphor.core.config import AgentConfig
from phosphor.core.types import Chunk, Scored
from phosphor.llm import MockLLM
from phosphor.tools import ToolRegistry, register_builtins, route_deterministic


class FakeRetriever:
    def __init__(self) -> None:
        self.chunks = [
            Chunk(id="c1", doc_id="d1", text="Phosphor 使用 reciprocal rank fusion 融合两路召回。"),
            Chunk(id="c2", doc_id="d2", text="HNSW 是 CPU 场景的默认索引。"),
        ]

    def search(self, query, k=6, trace_id=""):
        hits = [Scored(chunk=c, score=1.0, source="fake") for c in self.chunks]
        return hits[:k]

    def drop_document(self, doc_id):
        return 0


def test_parse_action_reads_tool_and_arguments():
    parsed = parse_action('Thought: x\nAction: calculator {"expression": "1+1"}')
    assert parsed is not None
    name, args = parsed
    assert name == "calculator" and args["expression"] == "1+1"


def test_parse_final_rejects_protocol_lines():
    text = "Thought: 思考\nAction: calculator\nObservation: 42"
    assert parse_final(text) == ""


def test_parse_final_reads_final_answer():
    assert parse_final("Final Answer: 答案是 42") == "答案是 42"


def test_agent_answers_from_evidence():
    tools = register_builtins(ToolRegistry(), retriever=FakeRetriever())
    agent = ReActAgent(MockLLM(), FakeRetriever(), tools, AgentConfig())
    answer = agent.run("Phosphor 使用什么融合策略？")
    assert "reciprocal rank fusion" in answer.answer
    assert answer.converged


def test_deterministic_routing_injects_tool_result():
    tools = register_builtins(ToolRegistry(), retriever=FakeRetriever())
    agent = ReActAgent(
        MockLLM(),
        FakeRetriever(),
        tools,
        AgentConfig(),
        deterministic_router=route_deterministic,
    )
    answer = agent.run("12*(3+4) 等于多少？")
    assert "84" in answer.answer
    assert answer.tool_calls >= 1


def test_iterations_are_bounded():
    class LoopLLM:
        name = "loop"
        model = "loop"

        def complete(self, messages, **kwargs):
            return 'Thought: 继续\nAction: echo {"text": "继续"}'

        def stream(self, messages, **kwargs):
            yield ""

    tools = register_builtins(ToolRegistry(), retriever=FakeRetriever())
    agent = ReActAgent(LoopLLM(), FakeRetriever(), tools, AgentConfig(max_iterations=3))
    answer = agent.run("任意问题")
    assert answer.iterations <= 3
    assert answer.answer
