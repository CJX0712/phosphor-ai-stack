from phosphor.agent.prompts import build_messages
from phosphor.core.types import Message
from phosphor.llm import MockLLM, parse_evidence
from phosphor.llm.mock import _deterministic_observation


def test_extractive_answer_uses_evidence():
    llm = MockLLM()
    messages = build_messages(
        "融合策略是什么？",
        "[id#c1] Phosphor 使用 reciprocal rank fusion 进行融合。无关句子没有任何重合。",
    )
    answer = llm.complete(messages)
    assert "reciprocal rank fusion" in answer


def test_answer_is_de_duplicated():
    llm = MockLLM(max_sentences=3)
    sentence = "同一个句子被重复写入语料。"
    messages = build_messages("重复", f"[id#c1] {sentence} {sentence} {sentence}")
    answer = llm.complete(messages)
    assert answer.count("同一个句子") == 1


def test_no_evidence_is_stated_explicitly():
    llm = MockLLM()
    answer = llm.complete(build_messages("问题？", ""))
    assert "no evidence" in answer


def test_parse_evidence_appends_continuation_lines():
    messages = build_messages("q", "[id#c1] 第一行\n第二行是续行\n[id#c2] 另一条")
    items = parse_evidence(messages)
    assert len(items) == 2
    assert "第二行是续行" in items[0][1]


def test_deterministic_observation_outranks_extraction():
    messages = build_messages("12*(3+4) 等于多少？", "[id#c1] 完全无关的证据内容。")
    messages.append(Message(role="user", content="Observation: calculator: 12*(3+4) = 84"))
    assert _deterministic_observation(messages) == "12*(3+4) = 84"
    assert "84" in MockLLM().complete(messages)


def test_question_is_taken_from_the_question_turn():
    """The last user turn is an observation, not the question."""
    llm = MockLLM()
    messages = build_messages("融合策略是什么？", "[id#c1] Phosphor 使用 RRF 融合。")
    messages.append(Message(role="user", content="Observation: 一些观测内容"))
    answer = llm.complete(messages)
    assert "RRF" in answer


def test_stream_emits_tokens():
    llm = MockLLM()
    messages = build_messages("融合", "[id#c1] Phosphor 使用 RRF 融合策略。")
    pieces = list(llm.stream(messages))
    assert pieces
