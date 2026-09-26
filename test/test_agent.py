"""Tests for ``LangGraphAgent`` entry points, with the graph and LLM faked out."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.types import Command

from app.core.langgraph import graph as graph_module
from app.core.langgraph.graph import LangGraphAgent
from app.schemas import (
    GraphState,
    Message,
)


def make_agent(llm_service=None, graph=None):
    agent = LangGraphAgent.__new__(LangGraphAgent)  # skip __init__: it binds real tools to the real LLM
    agent.llm_service = llm_service
    agent._graph = graph
    agent._connection_pool = None
    return agent


class FakeGraph:
    """Stands in for the compiled graph: records how it was entered."""

    def __init__(self, snapshot, final_messages=None):
        self.snapshot = snapshot
        self.final_messages = final_messages or [AIMessage(content="the answer")]
        self.inputs = []

    async def aget_state(self, config):
        return self.snapshot

    async def ainvoke(self, input=None, config=None, **kwargs):
        self.inputs.append(input)
        return {"messages": self.final_messages}


def snapshot(next_=(), tasks=(), values=None):
    return SimpleNamespace(next=next_, tasks=tasks, values=values or {})


@pytest.fixture(autouse=True)
def quiet_memory(monkeypatch):
    async def search(user_id, query):
        return ""

    async def add(*args, **kwargs):
        return None

    monkeypatch.setattr(graph_module, "memory_service", SimpleNamespace(search=search, add=add))


class TestGetResponse:
    def test_returns_the_assistant_reply(self):
        # regression: ``graph = await self._graph`` awaited a CompiledStateGraph -> TypeError on every /chat
        fake = FakeGraph(snapshot())
        agent = make_agent(graph=fake)

        result = asyncio.run(agent.get_response([Message(role="user", content="hi")], session_id="s1", user_id="1"))

        assert result == [Message(role="assistant", content="the answer")]

    def test_fresh_turn_sends_the_messages_as_input(self):
        fake = FakeGraph(snapshot())
        agent = make_agent(graph=fake)

        asyncio.run(agent.get_response([Message(role="user", content="hi")], session_id="s1"))

        assert fake.inputs[0]["messages"] == [{"role": "user", "content": "hi"}]
        assert fake.inputs[0]["tool_call_count"] == 0

    def test_dropped_connection_continues_from_the_checkpoint(self):
        crashed = snapshot(("chat",), (), {"messages": [HumanMessage(content="weather?")]})
        fake = FakeGraph(crashed)
        agent = make_agent(graph=fake)

        asyncio.run(agent.get_response([Message(role="user", content="weather?")], session_id="s1"))

        assert fake.inputs == [None]

    def test_ask_human_pause_resumes_with_the_users_answer(self):
        paused = snapshot(("ask",), (SimpleNamespace(interrupts=(SimpleNamespace(value="What city?"),)),))
        fake = FakeGraph(paused)
        agent = make_agent(graph=fake)

        asyncio.run(agent.get_response([Message(role="user", content="Paris")], session_id="s1"))

        assert len(fake.inputs) == 1
        assert isinstance(fake.inputs[0], Command) and fake.inputs[0].resume == "Paris"

    def test_new_question_after_a_crash_starts_a_new_turn(self):
        crashed = snapshot(("chat",), (), {"messages": [HumanMessage(content="weather?")]})
        fake = FakeGraph(crashed)
        agent = make_agent(graph=fake)

        asyncio.run(agent.get_response([Message(role="user", content="something else")], session_id="s1"))

        assert isinstance(fake.inputs[0], dict)


class TestToolCallBudget:
    """``_chat`` must stop tools at the budget it is *given* (workers get a smaller one)."""

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def get_llm(self):
            return SimpleNamespace()  # no model_name -> falls back to the default

        async def call(self, messages, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            return AIMessage(content="ok")

    def run_chat(self, tool_call_count, **chat_kwargs):
        llm = self.FakeLLM()
        agent = make_agent(llm_service=llm)
        state = GraphState(messages=[HumanMessage(content="q")], tool_call_count=tool_call_count)
        config = {"configurable": {"thread_id": "t"}, "metadata": {"username": "u"}}

        asyncio.run(agent._chat(state, config, **chat_kwargs))
        return llm.calls[0]

    def test_under_budget_lets_the_model_use_tools(self):
        call = self.run_chat(2, tool_call_limit=3)

        assert "model_name" not in call  # normal path keeps the tool-bound model
        assert "Do not call any more tools" not in call["messages"][-1]["content"]

    def test_at_the_workers_smaller_budget_tools_are_cut_off(self):
        # regression: _chat compared against a non-existent settings.MAX_TOOL_LIMIT and ignored this argument
        call = self.run_chat(3, tool_call_limit=3)

        assert "model_name" in call  # routed through the tool-less override path
        assert "Do not call any more tools" in call["messages"][-1]["content"]

    def test_default_budget_is_the_per_turn_setting(self):
        limit = graph_module.settings.MAX_TOOL_CALLS_PER_TURN

        assert "model_name" not in self.run_chat(limit - 1)
        assert "model_name" in self.run_chat(limit)


class TestClose:
    def test_close_shuts_the_pool_and_forgets_the_graph(self):
        closed = []

        class Pool:
            async def close(self):
                closed.append(True)

        agent = make_agent(graph=object())
        agent._connection_pool = Pool()

        asyncio.run(agent.close())

        assert closed == [True]
        assert agent._connection_pool is None and agent._graph is None

    def test_close_before_any_connection_is_a_no_op(self):
        asyncio.run(make_agent().close())
