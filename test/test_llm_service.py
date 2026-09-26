"""Tests for ``LLMService.bind_tools``."""

from app.services.llm.service import LLMService


class FakeLLM:
    """Mimics LangChain: ``bind_tools`` returns a NEW runnable, it does not mutate."""

    def __init__(self, tools=()):
        self.tools = tuple(tools)

    def bind_tools(self, tools):
        return FakeLLM(tools)


def make_service(llm):
    service = LLMService.__new__(LLMService)
    service._llm = llm
    service._bound_tools = []
    return service


def test_bind_tools_replaces_the_llm_with_the_tool_bound_one():
    # regression: called a misspelled ``bond_tools`` and would have discarded the result anyway
    original = FakeLLM()
    service = make_service(original)

    returned = service.bind_tools(["search", "ask_human"])

    assert returned is service
    assert service._llm is not original
    assert service._llm.tools == ("search", "ask_human")
    assert service._bound_tools == ["search", "ask_human"]


def test_bind_tools_without_an_llm_is_a_no_op():
    service = make_service(None)

    assert service.bind_tools(["search"]) is service
    assert service._llm is None
