"""This file contains the RAG (groups + documents) schemas for the application."""

import re
from datetime import datetime
from typing import (
    List,
    Optional,
)

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    field_validator,
)

from app.core.config import settings
from app.schemas.base import BaseResponse

class GroupCreate(BaseModel):
    """Request model for creating a group.

    Attributes:
        name: The unique group name.
    """
    name: str = Field(..., min_length=1, max_length=100, description="Unique group name")

    @classmethod
    @field_validator("name")
    def sanitize_name(selfcls, v:str) -> str:
        """Strip characters that have no place in a group name."""
        return re.sub(r'[<>{}[\]()\'"`]', "", v).strip()

class GroupResponse(BaseResponse):
    """Response model for a group, from the requesting user's perspective.

    Attributes:
        id: The group's ID.
        name: The group's name.
        role: The requesting user's role in this group ("admin" or "member").
    """
    id: int = Field(..., description="ID of the group")
    name: str = Field(..., description="Name of the group")
    role: str = Field(..., description="Role of the group")

# todo: change this later
class GroupMemberAdd(BaseModel):
    """Request model for adding a member to a group.

    Attributes:
        email: Email of the user to add.
        role: Role to grant — "admin" can ingest/delete documents and manage
            membership, "member" can only search/read.
    """
    email: EmailStr = Field(..., description="Email of the user to add.")
    role: str = Field(default="member", description="Role to add, admin or member")

    @field_validator(
        "role"
    )
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Restrict role to the two supported values."""
        if v not in ("admin", "member"):
            raise ValueError("role must be 'admin' or 'member'")
        return v

class GroupMemberResponse(BaseResponse):
    """Response model for a group member.

    Attributes:
        user_id: The member's user ID.
        email: The member's email.
        role: The member's role in the group.
    """
    user_id: int = Field(..., description="The member's user ID")
    email: str = Field(..., description="The member's email")
    role: str = Field(..., description="The member's role in the group")

class DocumentCreate(BaseModel):
    """Request model for ingesting a document into the knowledge base.

    Attributes:
        group_id: The group this document is scoped to. The requesting user
            must be an "admin" member of this group.
        title: Human-readable document title.
        content: The document's raw text content, chunked and embedded on ingest.
        source: Optional origin identifier (filename, URL, etc.).
    """
    group_id: int = Field(..., description="Group this document is scoped to")
    title: str = Field(..., min_length=1, max_length=200, description="Document title")
    content: str = Field(..., min_length=1, max_length=200_000, description="Raw text content to ingest")
    source: Optional[str] = Field(default=None, max_length=500, description="Optional origin identifier")

class DocumentResponse(BaseResponse):
    """Response model for a document.

    Attributes:
        id: The document's ID.
        group_id: The group this document is scoped to.
        owner_id: The user who uploaded the document.
        title: Document title.
        source: Optional origin identifier.
        chunk_count: Number of embedded chunks stored for this document.
        created_at: When the document was created.
    """
    id: str = Field(..., description="The document's ID")
    group_id: int = Field(..., description="Group this document is scoped to")
    owner_id: int = Field(..., description="The user who uploaded the document")
    title: str = Field(..., description="Document title")
    source: Optional[str] = Field(default=None, description="Optional origin identifier")
    chunk_count: int = Field(..., description="Number of embedded chunks stored for this document")
    created_at: datetime = Field(..., description="When the document was created")

class SearchRequest(BaseModel):
    """Request model for a direct knowledge-base search.

    Attributes:
        query: The search query text.
        group_id: Optional single group to restrict the search to. When
            omitted, all groups the requesting user belongs to are searched.
        top_k: Maximum number of chunks to return.
    """
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    group_id: Optional[int] = Field(default=None, description="restrict the search to")
    top_k: int = Field(default=settings.RAG_TOP_K, ge=1, le=20, description="Maximum number of chunks to return")

class SearchResult(BaseModel):
    """One matched chunk from a knowledge-base search.

    Attributes:
        document_id: The source document's ID.
        title: The source document's title.
        chunk_index: The chunk's position within its document.
        content: The chunk's text content.
        distance: Cosine distance to the query (lower is more relevant).
    """

    document_id: str = Field(..., description="The source document's ID")
    title: str = Field(..., description="The source document's title")
    chunk_index: int = Field(..., description="The chunk's position within its document")
    content: str = Field(..., description="The chunk's text content")
    distance: float = Field(..., description="Cosine distance to the query (lower is more relevant)")

class SearchResponse(BaseResponse):
    """Response model for a direct knowledge-base search.

    Attributes:
        results: Matched chunks, ordered by relevance.
    """

    results: List[SearchResult] = Field(..., description="Matched chunks, ordered by relevance")
