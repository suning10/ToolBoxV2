"""This file contains the graph schema for the application."""
import operator
from typing import (
    Annotated,
    Literal,
)

from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    Field,
)


class ToolCallRecord(BaseModel):
    """One executed (or reused-from-cache) tool call, for repeat/cycle detection."""

    name: str
    args: dict
    signature: str
    result: str


class GraphState(BaseModel):
    """State definition for the LangGraph Agent/Workflow."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    long_term_memory: str = Field(default="", description="The long term memory of the conversation")
    knowledge_base: str = Field(
        default="", description="Retrieved knowledge-base context relevant to the latest user message"
    )
    tool_call_count: int = Field(default=0, description="Number of tool-call rounds executed this turn")
    subtasks: list[str] = Field(default_factory=list, description="Subtasks from the lead agent's decomposition")
    subtask_results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="Worker outputs, merged across parallel branches"
    )
    action_history: list[ToolCallRecord] = Field(
        default_factory=list, description="Tool calls executed this turn/worker-run, for repeat/cycle detection"
    )

class QueryPlan(BaseModel):
    """Structured output from the lead agent's decomposition step."""

    complexity: Literal["simple", "complex"] = Field(description="Whether the query needs one agent or a swarm")
    subtasks: list[str] = Field(
        default_factory=list, description="Independent sub-questions to research in parallel; empty when 'simple'"
    )