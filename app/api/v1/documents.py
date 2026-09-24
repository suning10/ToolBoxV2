"""Document endpoints for the RAG knowledge base.

All access control is resolved from the authenticated user (``get_current_user``)
and enforced in ``app.services.rag.RAGService`` / ``app.services.group.GroupService``
— request bodies never carry a user identity that could be spoofed to read or
write another user's documents.
"""

from typing import List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from app.api.v1.auth import get_current_user
from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import logger
from app.models.user import User
from app.schemas.rag import (
    DocumentCreate,
    DocumentResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.rag import rag_service
from app.utils.sanitization import sanitize_string

router = APIRouter()

@router.post("/documents", response_model=DocumentResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["documents"][0])
async def ingest_document(request: Request, document: DocumentCreate, user: User = Depends(get_current_user)):
    """Ingest a document into the knowledge base.

    Args:
        request: The FastAPI request object for rate limiting.
        document: The document to chunk, embed, and store.
        user: The authenticated user — must be an admin of ``document.group_id``.

    Returns:
        DocumentResponse: The created document.

    Raises:
        HTTPException: 403 if the user isn't a group admin, 422 on bad input.
    """
    try:
        sanitized_title = sanitize_string(document.title)
        sanitized_source = sanitize_string(document.source) if document.source else None

        created, chunk_count = await rag_service.ingest_document(
            requesting_user_id=user.id,
            group_id=document.group_id,
            title=sanitized_title,
            content=document.content,
            source=sanitized_source,
        )

        logger.info("document_ingest_requested", document_id=created.id, user_id=user.id, group_id=document.group_id)
        return DocumentResponse(
            id=created.id,
            group_id=created.group_id,
            owner_id=created.owner_id,
            title=created.title,
            source=created.source,
            chunk_count=chunk_count,
            created_at=created.created_at,
        )
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))


@router.get("/documents", response_model=List[DocumentResponse])
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["documents"][0])
async def list_documents(request: Request, user: User = Depends(get_current_user)):
    """List every document the authenticated user has group access to.

    Args:
        request: The FastAPI request object for rate limiting.
        user: The authenticated user.

    Returns:
        List[DocumentResponse]: Accessible documents.
    """
    documents = await rag_service.list_accessible_documents(user.id)
    return [
        DocumentResponse(
            id=d.id,
            group_id=d.group_id,
            owner_id=d.owner_id,
            title=d.title,
            source=d.source,
            chunk_count=chunk_count,
            created_at=d.created_at,
        )
        for d, chunk_count in documents
    ]

@router.delete("/documents/{document_id}")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["documents"][0])
async def delete_document(request: Request, document_id: str, user: User = Depends(get_current_user)):
    """Delete a document and its chunks.

    Args:
        request: The FastAPI request object for rate limiting.
        document_id: The document to delete.
        user: The authenticated user — must be the owner or a group admin.

    Returns:
        dict: A confirmation message.

    Raises:
        HTTPException: 404 if not found, 403 if not authorized.
    """
    try:
        sanitized_document_id = sanitize_string(document_id)
        await rag_service.delete_document(sanitized_document_id, user.id)
        return {"message": "Document deleted successfully"}
    except LookupError as le:
        raise HTTPException(status_code=404, detail=str(le))
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))


@router.post("/documents/search", response_model=SearchResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["documents_search"][0])
async def search_documents(request: Request, search: SearchRequest, user: User = Depends(get_current_user)):
    """Search the knowledge base directly (outside the chat agent).

    Args:
        request: The FastAPI request object for rate limiting.
        search: The search query and optional scoping.
        user: The authenticated user — results are scoped to their groups.

    Returns:
        SearchResponse: Matched chunks ordered by relevance.
    """
    results = await rag_service.search(user.id, search.query, group_id=search.group_id, top_k=search.top_k)
    return SearchResponse(results=results)