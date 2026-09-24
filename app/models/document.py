"""This file contains the document and document-chunk models for RAG."""

from typing import (
    TYPE_CHECKING,
    List,
    Optional,
)

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import (
    Field,
    Relationship,
)

from app.core.config import settings
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.group import Group


class Document(BaseModel, table=True):
    """A document ingested into the knowledge base, scoped to one group.

    Access control is group-based: a user can read a document only if they
    belong to its ``group_id``, and can only ingest/delete documents in
    groups where their membership role is "admin" (see app/services/groups.py
    and app/services/rag.py, which enforce this — never trust the LLM or
    request body for group/user identity).

    Attributes:
        id: The primary key (UUID string)
        group_id: Foreign key to the owning group — the access-control boundary
        owner_id: Foreign key to the user who uploaded the document
        title: Human-readable document title
        source: Optional origin (filename, URL, etc.)
        created_at: When the document was created
        group: Relationship to the owning group
        chunks: Relationship to this document's chunks
    """
    id: str = Field(primary_key=True)
    group_id: int = Field(foreign_key="group.id", index=True)
    owner_id: int = Field(foreign_key="user.id", index=True)
    title: str
    source: Optional[str] = Field(default=None)
    group: "Group" = Relationship(back_populates="documents")
    chunks: List["DocumentChunk"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

class DocumentChunk(BaseModel, table=True):
    """One embedded chunk of a document, used for pgvector similarity search.

    Attributes:
        id: The primary key
        document_id: Foreign key to the parent document
        chunk_index: Position of this chunk within the document
        content: The chunk's text content
        embedding: The chunk's embedding vector
        created_at: When the chunk was created
        document: Relationship to the parent document
    """

    id: int = Field(default=None, primary_key=True)
    document_id: str = Field(foreign_key="document.id", index=True)
    chunk_index: int
    content: str
    embedding: List[float] = Field(sa_column=Column(Vector(settings.RAG_EMBEDDING_DIMENSIONS)))
    document: "Document" = Relationship(back_populates="chunks")
