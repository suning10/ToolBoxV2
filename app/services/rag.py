"""RAG (retrieval-augmented generation) service: chunking, embedding, and access-scoped search.

Access control is enforced entirely in this module using group IDs resolved
server-side from the authenticated user (never from LLM- or request-supplied
group/user IDs): ``search`` only ever queries documents whose ``group_id`` is
one the caller belongs to, and ``ingest_document``/``delete_document`` require
the caller to be an "admin" member of the target group.
"""

import logging
import uuid
from typing import (
    List,
    Optional,
    Tuple,
)

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import (
    APIError,
    APITimeoutError,
    RateLimitError,
)
from pydantic import SecretStr
from sqlalchemy import func
from sqlmodel import (
    Session,
    col,
    select,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.cache import (
    cache_key,
    cache_service,
)
from app.core.config import settings
from app.core.logging import logger
from app.models.document import (
    Document,
    DocumentChunk,
)
from app.schemas.rag import SearchResult
from app.services.database import database_service
from app.services.group import group_service


class RAGService:
    """Service for ingesting documents into, and searching, the group-scoped knowledge base."""

    def __init__(self):
        """Reuse the shared database engine; embedder is created lazily."""
        self.engine  = database_service.engine
        self._embedder: Optional[OpenAIEmbeddings] = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        )

    def _get_embedder(self) -> OpenAIEmbeddings:
        if self._embedder is None:
            self._embedder = OpenAIEmbeddings(
                model=settings.RAG_EMBEDDER_MODEL,
                api_key=SecretStr(settings.OPENAI_API_KEY),
            )
        return self._embedder

    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with automatic retry on transient API errors."""
        return await self._get_embedder().aembed_documents(texts)

    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _embed_query(self, text: str) -> List[float]:
        """Embed a single query string with automatic retry on transient API errors."""
        return await self._get_embedder().aembed_query(text)

    async def ingest_document(
        self,
        requesting_user_id: int,
        group_id: int,
        title: str,
        content: str,
        source: Optional[str] = None,
    ) -> Tuple[Document, int]:
        """Chunk, embed, and store a document, scoped to a group.

        Args:
            requesting_user_id: The uploading user — recorded as owner and
                must be an "admin" member of ``group_id``.
            group_id: The group this document is scoped to.
            title: Document title.
            content: Raw text content to chunk and embed.
            source: Optional origin identifier.

        Returns:
            tuple[Document, int]: The created document and its chunk count.
            The count is returned separately rather than via ``document.chunks``
            because that relationship can't be lazy-loaded once this method's
            session closes.

        Raises:
            PermissionError: If the requesting user is not a group admin.
            ValueError: If the content produces no chunks.
        """
        if not await group_service.is_admin(requesting_user_id, group_id):
            logger.warning("rag_ingest_denied", user_id=requesting_user_id, group_id=group_id)
            raise PermissionError("only group admins can add documents to this group")
        # 1. split text into chunks
        chunks = self._splitter.split_text(content)
        if not chunks:
            raise ValueError("content must contain at least one chunk")
        # 2. embed the chunks
        embedding = await self._embed_documents(chunks)
        # 3. add to vector db, index, actual_content, title, group_id, embedding
        with Session(self.engine) as session:
            document = Document(
                id=str(uuid.uuid4()), group_id=group_id, owner_id=requesting_user_id, title=title, source=source
            )
            session.add(document)
            session.flush()

            for index, (chunk_text, embedding) in enumerate(zip(chunks, embedding, strict=True)):
                session.add(
                    DocumentChunk(document_id=document.id, chunk_index=index, content=chunk_text, embedding=embedding)
                )
            session.commit()
            session.refresh(document)

            logger.info(
                "document_ingested",
                document_id=document.id,
                group_id=group_id,
                owner_id=requesting_user_id,
                chunk_count=len(chunks),
            )
            return document, len(chunks)

    async def delete_document(self, document_id: str, requesting_user_id: int) -> None:
        """Delete a document and its chunks.

        Args:
            document_id: The document to delete.
            requesting_user_id: The user requesting deletion — must be the
                document's owner or an "admin" member of its group.

        Raises:
            LookupError: If the document doesn't exist.
            PermissionError: If the requesting user is not authorized.
        """
        with Session(self.engine) as session:
            document = session.get(Document, document_id)
            if document is None:
                raise LookupError("document not found")

            is_owner = document.owner_id == requesting_user_id
            if not is_owner and not await group_service.is_admin(requesting_user_id, document.group_id):
                logger.warning(
                    "rag_delete_denied",
                    user_id=requesting_user_id,
                    document_id=document_id,
                    group_id=document.group_id,
                )
                raise PermissionError("only the owner or a group admin can delete this document")

            session.delete(document)
            session.commit()
            logger.info("document_deleted", document_id=document_id, requesting_user_id=requesting_user_id)

    async def list_accessible_documents(self, user_id: int) -> List[Tuple[Document, int]]:
        """List every document in a group the user belongs to, with each document's chunk count."""
        accessible_group_ids = await group_service.user_group_ids(user_id)
        if not accessible_group_ids:
            return []

        with Session(self.engine) as session:
            # get document + chunk count by user_id and group by document id
            statement = (
                select(Document, func.count(col(DocumentChunk.id)))
                       .join(DocumentChunk, col(DocumentChunk.document_id) == col(Document.id), isouter=True)
                       .where(col(Document.group_id).in_(accessible_group_ids))
                       .group_by(col(DocumentChunk.id))
            )
            return list(session.exec(statement).all())

    async def search(
        self,
        user_id: int,
        query: str,
        group_id: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Search chunks in documents the user has group access to.

        Args:
            user_id: The searching user — results are filtered to documents
                whose group is one the user belongs to. This is resolved
                server-side and is the sole access-control boundary; callers
                must never let this be overridden by request/LLM input.
            query: The search query text.
            group_id: Optional single group to restrict to. Silently yields
                no results if the user isn't a member of it, rather than
                leaking whether the group exists.
            top_k: Maximum chunks to return.

        Returns:
            list[SearchResult]: Matched chunks ordered by relevance (closest first).
        """
        # 1. get available contents from vector db by user id
        accessible_group_ids = await group_service.user_group_ids(user_id)
        if group_id is not None and group_id not in accessible_group_ids:
            accessible_group_ids.append(group_id)
        if not accessible_group_ids:
            return []
        # 2. embed the query
        query_embedding = await self._embed_query(query)
        # 3. calculate cosine_distance by ini a pgvector comparator
        distance = col(DocumentChunk.embedding).cosine_distance(query_embedding)
        # 4 run sql to retrieve top k
        # 4.1 join Document to make sure user have access of the file
        # 4.2 achieve this by looking at document.group_id is in accessible_group_ids
        with Session(self.engine) as session:
            statement = (
                select(DocumentChunk, Document, distance.label("distance"))
                .join(Document, col(DocumentChunk.document_id) == col(Document.id))
                .where(col(Document.group_id).in_(accessible_group_ids))
                .where(distance <= settings.RAG_MAX_DISTANCE)
                .order_by(distance)
                .limit(top_k or settings.RAG_TOP_K)
            )
            rows = session.exec(statement).all()
            # todo: add logic of reranking
            return [
                SearchResult(
                    document_id=chunk.document_id,
                    title=document.title,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    distance=result_distance,
                )
                for chunk, document, result_distance in rows
            ]

    @staticmethod
    def _format_chunks(results: List[SearchResult]) -> str:
        """Join results into citation-friendly text, or "" when there are none."""
        return "\n\n".join(f"[{r.title} — chunk {r.chunk_index}]\n{r.content}" for r in results)

    async def format_results(self, results: List[SearchResult]) -> str:
        """Format search results as a citation-friendly string for the LLM, with a fallback message."""
        return self._format_chunks(results) or "No relevant documents found in your accessible knowledge base."

    async def search_context(self, user_id: Optional[str], query: str) -> str:
        """Search the user's accessible knowledge base and return formatted context for the system prompt.

        search in cache first to speed up

        Mirrors ``app.services.memory.MemoryService.search``'s contract: no-op
        (returns "") when there's no authenticated ``user_id``, on search
        failure, or when nothing relevant is found — callers apply their own
        fallback wording, matching how long-term memory is surfaced. Only
        non-empty results are cached, since an empty hit isn't worth serving stale.

        Args:
            user_id: The searching user, as a string (session metadata carries
                it this way). ``None`` short-circuits — anonymous sessions
                have no group memberships to search.
            query: The user's latest message, used as the search query.

        Returns:
            str: Formatted, citation-friendly excerpts, or "" if none apply.
        """
        if user_id is None:
            return ""
        try:
            # search in cache first
            key = cache_key("rag", str(user_id), query)
            cached = await cache_service.get(key)
            if cached is not None:
                logger.debug("rag_search_context_cache_hit", user_id=user_id)
                return cached

            results = await self.search(int(user_id), query, settings.RAG_TOP_K)
            formatted = self._format_chunks(results)

            if formatted:
                await cache_service.set(key, formatted)

            return formatted
        except Exception as e:
            logger.exception("rag_search_context_failed", error=str(e), user_id=user_id, query=query)
            return ""

rag_service = RAGService()