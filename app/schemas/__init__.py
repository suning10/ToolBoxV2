"""This file contains the schemas for the application."""

from app.schemas.auth import Token
from app.schemas.base import BaseResponse
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Message,
    StreamResponse,
)
from app.schemas.graph import (
    GraphState,
    QueryPlan,
    ToolCallRecord,
)
from app.schemas.rag import (
    DocumentCreate,
    DocumentResponse,
    GroupCreate,
    GroupMemberAdd,
    GroupMemberResponse,
    GroupResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

__all__ = [
    "Token",
    "BaseResponse",
    "ChatRequest",
    "ChatResponse",
    "Message",
    "StreamResponse",
    "GraphState",
    "QueryPlan",
    "ToolCallRecord",
    "DocumentCreate",
    "DocumentResponse",
    "GroupCreate",
    "GroupMemberAdd",
    "GroupMemberResponse",
    "GroupResponse",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]