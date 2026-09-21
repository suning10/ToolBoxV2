"""Tests for the pure helpers in ``app/utils/graph.py``."""

from pathlib import Path

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from app.core.config import settings
from app.schemas import Message
from app.schemas.graph import ToolCallRecord
from app.utils.graph import (
    _TIKTOKEN_ENCODING,
    _count_tokens_tiktoken,
    compute_call_signature,
    detect_cycle,
    dump_messages,
    extract_text_content,
    find_duplicate_call,
    prepare_messages,
    process_llm_response,
)


def record(name: str, args: dict, result: str = "ok") -> ToolCallRecord:
    return ToolCallRecord(name=name, args=args, signature=compute_call_signature(name, args), result=result)


def history_of(*labels: str) -> list[ToolCallRecord]:
    """Build a history where each distinct label is a distinct call signature."""
    return [record("tool", {"label": label}) for label in labels]


class TestComputeCallSignature:
    def test_is_a_16_char_hex_digest(self):
        signature = compute_call_signature("search", {"query": "cats"})

        assert len(signature) == 16
        int(signature, 16)  # raises if not hex

    def test_is_deterministic(self):
        assert compute_call_signature("search", {"q": "x"}) == compute_call_signature("search", {"q": "x"})

    def test_ignores_argument_key_order(self):
        assert compute_call_signature("t", {"a": 1, "b": 2}) == compute_call_signature("t", {"b": 2, "a": 1})

    def test_ignores_key_order_in_nested_arguments(self):
        first = compute_call_signature("t", {"outer": {"a": 1, "b": 2}})
        second = compute_call_signature("t", {"outer": {"b": 2, "a": 1}})

        assert first == second

    def test_differs_by_tool_name(self):
        assert compute_call_signature("a", {"q": "x"}) != compute_call_signature("b", {"q": "x"})

    def test_differs_by_argument_value(self):
        assert compute_call_signature("t", {"q": "x"}) != compute_call_signature("t", {"q": "y"})

    def test_argument_value_type_matters(self):
        assert compute_call_signature("t", {"n": 1}) != compute_call_signature("t", {"n": "1"})

    def test_handles_non_json_serialisable_arguments(self):
        assert len(compute_call_signature("t", {"path": Path("/tmp/x"), "tags": {"a"}})) == 16

    def test_empty_arguments(self):
        assert len(compute_call_signature("t", {})) == 16


class TestExtractTextContent:
    def test_plain_string_is_returned_as_is(self):
        assert extract_text_content("hello") == "hello"

    def test_list_of_strings_is_concatenated(self):
        assert extract_text_content(["foo", "bar"]) == "foobar"

    def test_text_blocks_are_concatenated(self):
        assert extract_text_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"

    def test_reasoning_blocks_are_dropped(self):
        content = [
            {"type": "reasoning", "id": "rs_1", "summary": [{"text": "thinking..."}]},
            {"type": "text", "text": "The answer."},
        ]

        assert extract_text_content(content) == "The answer."

    def test_unknown_block_types_are_dropped(self):
        assert extract_text_content([{"type": "image_url", "image_url": "x"}, {"type": "text", "text": "t"}]) == "t"

    def test_text_block_missing_text_contributes_nothing(self):
        assert extract_text_content([{"type": "text"}, "tail"]) == "tail"

    def test_mixed_strings_and_blocks(self):
        assert extract_text_content(["a", {"type": "text", "text": "b"}, "c"]) == "abc"

    def test_empty_inputs(self):
        assert extract_text_content("") == ""
        assert extract_text_content([]) == ""


class TestProcessLlmResponse:
    def test_structured_content_is_flattened_to_a_string(self):
        response = AIMessage(content=[{"type": "reasoning", "id": "r"}, {"type": "text", "text": "hi"}])

        result = process_llm_response(response)

        assert result.content == "hi"

    def test_returns_the_same_instance(self):
        response = AIMessage(content=[{"type": "text", "text": "hi"}])

        assert process_llm_response(response) is response

    def test_string_content_is_left_alone(self):
        response = AIMessage(content="already text")

        assert process_llm_response(response).content == "already text"

    def test_tool_calls_survive_normalisation(self):
        tool_call = {"name": "search", "args": {"q": "x"}, "id": "call_1", "type": "tool_call"}
        response = AIMessage(content=[{"type": "text", "text": "calling"}], tool_calls=[tool_call])

        assert process_llm_response(response).tool_calls == [tool_call]


class TestCountTokens:
    def test_empty_conversation_only_costs_reply_priming(self):
        assert _count_tokens_tiktoken([]) == 2

    def test_dict_message_counts_overhead_plus_string_values(self):
        message = {"role": "user", "content": "hello world"}
        expected = 4 + len(_TIKTOKEN_ENCODING.encode("user")) + len(_TIKTOKEN_ENCODING.encode("hello world")) + 2

        assert _count_tokens_tiktoken([message]) == expected

    def test_non_string_dict_values_are_ignored(self):
        assert _count_tokens_tiktoken([{"n": 5, "flag": True, "none": None}]) == 4 + 2

    def test_langchain_message_string_content(self):
        expected = 4 + len(_TIKTOKEN_ENCODING.encode("hello world")) + 2

        assert _count_tokens_tiktoken([HumanMessage(content="hello world")]) == expected

    def test_langchain_message_block_content(self):
        message = AIMessage(content=["plain", {"type": "text", "text": "block"}, {"type": "reasoning", "id": "r"}])
        expected = 4 + len(_TIKTOKEN_ENCODING.encode("plain")) + len(_TIKTOKEN_ENCODING.encode("block")) + 2

        assert _count_tokens_tiktoken([message]) == expected

    def test_count_grows_with_more_messages(self):
        one = _count_tokens_tiktoken([{"content": "hi"}])
        two = _count_tokens_tiktoken([{"content": "hi"}, {"content": "hi"}])

        assert two > one


class TestDumpMessages:
    def test_dumps_role_and_content(self):
        messages = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]

        assert dump_messages(messages) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_empty(self):
        assert dump_messages([]) == []


class TestPrepareMessages:
    def test_system_prompt_is_prepended(self):
        result = prepare_messages([Message(role="user", content="hi")], "You are helpful.")

        assert result[0] == Message(role="system", content="You are helpful.")

    def test_short_conversation_is_kept_intact_and_in_order(self):
        messages = [
            Message(role="user", content="first question"),
            Message(role="assistant", content="first answer"),
            Message(role="user", content="second question"),
        ]

        result = prepare_messages(messages, "sys")

        assert [m.content for m in result[1:]] == ["first question", "first answer", "second question"]

    def test_no_messages_yields_only_the_system_prompt(self):
        assert prepare_messages([], "sys") == [Message(role="system", content="sys")]

    def test_history_over_the_token_budget_is_trimmed_from_the_front(self, monkeypatch):
        monkeypatch.setattr(settings, "MAX_TOKENS", 60)
        messages = []
        for i in range(10):
            messages.append(Message(role="user", content=f"question number {i} " * 3))
            messages.append(Message(role="assistant", content=f"answer number {i} " * 3))

        result = prepare_messages(messages, "sys")

        kept = result[1:]
        assert 0 < len(kept) < len(messages)
        assert kept[0].content == messages[-len(kept)].content  # a suffix of the history survived
        assert kept[-1].content == messages[-1].content  # the newest message is always kept
        assert _count_tokens_tiktoken(kept) <= 60

    def test_trimmed_history_starts_on_a_human_message(self, monkeypatch):
        monkeypatch.setattr(settings, "MAX_TOKENS", 60)
        messages = []
        for i in range(10):
            messages.append(Message(role="user", content=f"question number {i} " * 3))
            messages.append(Message(role="assistant", content=f"answer number {i} " * 3))

        kept = prepare_messages(messages, "sys")[1:]

        assert isinstance(kept[0], BaseMessage)
        assert kept[0].type == "human"


class TestFindDuplicateCall:
    def test_empty_history(self):
        assert find_duplicate_call("search", {"query": "x"}, [], 0.9) is None

    def test_exact_repeat_is_found(self):
        prior = record("search", {"query": "python asyncio"})

        assert find_duplicate_call("search", {"query": "python asyncio"}, [prior], 0.9) is prior

    def test_exact_repeat_ignores_argument_key_order(self):
        prior = record("t", {"a": 1, "b": 2})

        assert find_duplicate_call("t", {"b": 2, "a": 1}, [prior], 0.99) is prior

    def test_near_duplicate_above_threshold_is_found(self):
        prior = record("search", {"query": "python asyncio tutorial"})

        assert find_duplicate_call("search", {"query": "python asyncio tutorials"}, [prior], 0.9) is prior

    def test_dissimilar_arguments_are_not_duplicates(self):
        prior = record("search", {"query": "weather in Paris"})

        assert find_duplicate_call("search", {"query": "stock price of AAPL"}, [prior], 0.9) is None

    def test_threshold_is_exclusive_of_a_near_match(self):
        prior = record("search", {"query": "python asyncio tutorial"})

        assert find_duplicate_call("search", {"query": "python asyncio tutorials"}, [prior], 1.0) is None

    def test_same_arguments_on_a_different_tool_are_not_duplicates(self):
        prior = record("search", {"query": "x"})

        assert find_duplicate_call("rag_search", {"query": "x"}, [prior], 0.0) is None

    def test_earliest_matching_entry_is_returned(self):
        first = record("search", {"query": "python asyncio"}, result="first")
        second = record("search", {"query": "python asyncio"}, result="second")

        assert find_duplicate_call("search", {"query": "python asyncio"}, [first, second], 0.9) is first

    def test_finds_match_behind_non_matching_entries(self):
        history = [
            record("other", {"query": "python asyncio"}),
            record("search", {"query": "totally different"}),
            record("search", {"query": "python asyncio"}),
        ]

        assert find_duplicate_call("search", {"query": "python asyncio"}, history, 0.9) is history[2]


class TestDetectCycle:
    def test_empty_and_single_call_history(self):
        assert detect_cycle([]) is None
        assert detect_cycle(history_of("A")) is None

    def test_no_repetition(self):
        assert detect_cycle(history_of("A", "B", "C", "D")) is None

    def test_same_call_twice_is_period_one(self):
        assert detect_cycle(history_of("A", "A")) == 1

    def test_alternating_calls_are_period_two(self):
        assert detect_cycle(history_of("A", "B", "A", "B")) == 2

    def test_three_step_loop_is_period_three(self):
        assert detect_cycle(history_of("A", "B", "C", "A", "B", "C")) == 3

    def test_incomplete_repetition_is_not_a_cycle(self):
        assert detect_cycle(history_of("A", "B", "C", "A", "B")) is None

    def test_only_the_tail_of_the_history_matters(self):
        assert detect_cycle(history_of("A", "B", "A", "B", "C", "D")) is None
        assert detect_cycle(history_of("X", "Y", "Z", "A", "B", "A", "B")) == 2

    def test_smallest_period_wins(self):
        assert detect_cycle(history_of("A", "A", "A", "A")) == 1

    def test_min_repeats_raises_the_bar(self):
        assert detect_cycle(history_of("A", "A"), min_repeats=3) is None
        assert detect_cycle(history_of("A", "A", "A"), min_repeats=3) == 1
        assert detect_cycle(history_of("A", "B", "A", "B"), min_repeats=3) is None
        assert detect_cycle(history_of("A", "B", "A", "B", "A", "B"), min_repeats=3) == 2

    def test_max_period_limits_the_pattern_length(self):
        assert detect_cycle(history_of("A", "B", "C", "A", "B", "C"), max_period=2) is None
        assert detect_cycle(history_of("A", "B", "A", "B"), max_period=1) is None

    def test_cycle_is_detected_by_signature_not_result(self):
        history = [record("t", {"q": "x"}, result="r1"), record("t", {"q": "x"}, result="r2")]

        assert detect_cycle(history) == 1
