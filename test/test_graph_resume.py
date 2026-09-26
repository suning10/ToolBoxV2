"""Tests for telling an ``ask_human`` pause apart from a run that died mid-flight.

``snapshot.next`` is non-empty in both cases, so resuming with ``Command(resume=...)``
on a dropped connection silently swallows the user's message. These tests pin the
discriminators in ``app/utils/graph.py`` and check the LangGraph behaviour they rely on.
"""

import asyncio
from types import SimpleNamespace
from typing import Annotated

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.graph.message import add_messages
from langgraph.types import (
    Command,
    interrupt,
)
from pydantic import BaseModel

from app.schemas import Message
from app.utils.graph import (
    is_orphaned_run_for,
    pending_interrupt_value,
)


def snapshot(next_=(), tasks=(), values=None):
    return SimpleNamespace(next=next_, tasks=tasks, values=values or {})


def task(*interrupt_values):
    return SimpleNamespace(interrupts=tuple(SimpleNamespace(value=v) for v in interrupt_values))


def user(text):
    return Message(role="user", content=text)


class TestPendingInterruptValue:
    def test_no_tasks(self):
        assert pending_interrupt_value(snapshot()) is None

    def test_tasks_without_interrupts_are_not_interrupts(self):
        assert pending_interrupt_value(snapshot(next_=("chat",), tasks=(task(),))) is None

    def test_returns_first_interrupt_value(self):
        assert pending_interrupt_value(snapshot(("ask",), (task("What city?"),))) == "What city?"

    def test_finds_interrupt_on_a_later_task(self):
        assert pending_interrupt_value(snapshot(("a", "b"), (task(), task("second")))) == "second"


class TestIsOrphanedRunFor:
    STORED = {"messages": [HumanMessage(content="hi"), AIMessage(content="hello"), HumanMessage(content="weather?")]}

    def test_same_user_message_with_unfinished_run_is_orphaned(self):
        assert is_orphaned_run_for(snapshot(("chat",), values=self.STORED), [user("weather?")]) is True

    def test_new_user_message_is_a_new_turn(self):
        assert is_orphaned_run_for(snapshot(("chat",), values=self.STORED), [user("something else")]) is False

    def test_completed_run_is_never_orphaned(self):
        assert is_orphaned_run_for(snapshot((), values=self.STORED), [user("weather?")]) is False

    def test_pending_interrupt_is_not_an_orphan(self):
        snap = snapshot(("ask",), (task("What city?"),), self.STORED)

        assert is_orphaned_run_for(snap, [user("weather?")]) is False

    def test_only_the_last_message_is_compared(self):
        snap = snapshot(("chat",), values=self.STORED)

        assert is_orphaned_run_for(snap, [user("unrelated earlier"), user("weather?")]) is True

    def test_last_message_must_come_from_the_user(self):
        snap = snapshot(("chat",), values=self.STORED)

        assert is_orphaned_run_for(snap, [Message(role="assistant", content="weather?")]) is False

    def test_no_human_message_stored(self):
        snap = snapshot(("chat",), values={"messages": [AIMessage(content="hello")]})

        assert is_orphaned_run_for(snap, [user("hello")]) is False

    @pytest.mark.parametrize("values", [None, {}, {"messages": []}])
    def test_empty_state(self, values):
        assert is_orphaned_run_for(snapshot(("chat",), values=values), [user("hi")]) is False

    def test_no_incoming_messages(self):
        assert is_orphaned_run_for(snapshot(("chat",), values=self.STORED), []) is False

    def test_structured_content_blocks_are_compared_as_text(self):
        stored = {"messages": [HumanMessage(content=[{"type": "text", "text": "weather?"}])]}

        assert is_orphaned_run_for(snapshot(("chat",), values=stored), [user("weather?")]) is True


# --- against a real LangGraph graph + checkpointer -------------------------------------------------


class State(BaseModel):
    messages: Annotated[list, add_messages] = []


def build_graph(calls: dict, fail_second_node_once: bool = False, ask: bool = False):
    """START -> first -> second -> END. ``second`` optionally crashes once or asks the human."""

    async def first(state: State):
        calls["first"] += 1
        return {"messages": [AIMessage(content="first done")]}

    async def second(state: State):
        calls["second"] += 1
        if ask:
            answer = interrupt("What city?")
            return {"messages": [AIMessage(content=f"city={answer}")]}
        if fail_second_node_once and calls["second"] == 1:
            raise ConnectionError("client went away")  # stands in for the generator being cancelled
        return {"messages": [AIMessage(content="second done")]}

    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile(checkpointer=InMemorySaver())


CONFIG = {"configurable": {"thread_id": "t1"}}
TURN = {"messages": [{"role": "user", "content": "weather?"}]}


def test_crashed_run_is_orphaned_and_resumes_from_checkpoint_without_rerunning_done_nodes():
    calls = {"first": 0, "second": 0}
    graph = build_graph(calls, fail_second_node_once=True)

    async def scenario():
        with pytest.raises(ConnectionError):
            await graph.ainvoke(TURN, CONFIG)

        crashed = await graph.aget_state(CONFIG)
        assert crashed.next == ("second",)  # the ambiguous signal the old code keyed on
        assert pending_interrupt_value(crashed) is None
        assert is_orphaned_run_for(crashed, [user("weather?")]) is True
        assert is_orphaned_run_for(crashed, [user("a different question")]) is False

        result = await graph.ainvoke(None, CONFIG)
        return result, await graph.aget_state(CONFIG)

    result, final = asyncio.run(scenario())

    assert result["messages"][-1].content == "second done"
    assert calls == {"first": 1, "second": 2}  # `first` was NOT re-run
    assert final.next == ()
    assert [m.content for m in final.values["messages"]] == ["weather?", "first done", "second done"]


def test_resume_command_on_a_crashed_run_does_not_consume_the_users_message():
    """Documents the original bug: this is what ``Command(resume=...)`` did on a dropped connection."""
    calls = {"first": 0, "second": 0}
    graph = build_graph(calls, fail_second_node_once=True)

    async def scenario():
        with pytest.raises(ConnectionError):
            await graph.ainvoke(TURN, CONFIG)
        await graph.ainvoke(Command(resume="weather?"), CONFIG)
        return await graph.aget_state(CONFIG)

    final = asyncio.run(scenario())

    # the resume value went nowhere: it is not in the conversation
    assert "weather?" not in [m.content for m in final.values["messages"][1:]]


def test_ask_human_pause_is_a_pending_interrupt_and_not_an_orphan():
    calls = {"first": 0, "second": 0}
    graph = build_graph(calls, ask=True)

    async def scenario():
        await graph.ainvoke(TURN, CONFIG)
        paused = await graph.aget_state(CONFIG)
        assert paused.next == ("second",)
        assert pending_interrupt_value(paused) == "What city?"
        assert is_orphaned_run_for(paused, [user("weather?")]) is False

        result = await graph.ainvoke(Command(resume="Paris"), CONFIG)
        return result, await graph.aget_state(CONFIG)

    result, final = asyncio.run(scenario())

    assert result["messages"][-1].content == "city=Paris"
    assert final.next == ()
    assert pending_interrupt_value(final) is None


def test_completed_turn_is_neither_interrupted_nor_orphaned():
    graph = build_graph({"first": 0, "second": 0})

    async def scenario():
        await graph.ainvoke(TURN, CONFIG)
        return await graph.aget_state(CONFIG)

    done = asyncio.run(scenario())

    assert pending_interrupt_value(done) is None
    assert is_orphaned_run_for(done, [user("weather?")]) is False
