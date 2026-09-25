import json
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse

from app.api.v1.auth import get_current_session
from app.core.config import settings
from app.core.langgraph.graph import LangGraphAgent
from app.core.limiter import limiter
from app.core.logging import logger
from app.core.metrics import llm_stream_duration_seconds
from app.models.session import Session
from app.schemas.chat import (
    ActiveRunResponse,
    ChatRequest,
    ChatResponse,
)
from app.services.run_stream import (
    START_ID,
    StreamEvent,
    is_valid_event_id,
    is_valid_run_id,
    run_stream_service,
)
from app.services.session_naming import maybe_name_session

router = APIRouter()
agent = LangGraphAgent()

RUN_ID_HEADER = "X-Run-Id"
# no-cache + no proxy buffering: a buffering proxy would hold tokens back and make resuming pointless
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(event: StreamEvent) -> str:
    """Format one follower event as SSE. Buffered events carry an ``id:`` so clients can resume from them."""
    if event.payload is None:
        return ": keep-alive\n\n"
    data = f"data: {json.dumps(event.payload)}\n\n"
    return f"id: {event.id}\n{data}" if event.id else data


def _follow_response(run_id: str, after_id: str) -> StreamingResponse:
    """Stream a run's events after ``after_id``. Disconnecting only stops *following*; the run keeps going."""

    async def body():
        async for event in run_stream_service.follow(run_id, after_id):
            yield _sse(event)

    return StreamingResponse(
        body(), media_type="text/event-stream", headers={**SSE_HEADERS, RUN_ID_HEADER: run_id}
    )


def _resume_point(last_event_id: Optional[str]) -> str:
    """Validate the ``Last-Event-ID`` header; absent means "from the beginning"."""
    if not last_event_id:
        return START_ID
    if not is_valid_event_id(last_event_id):
        raise HTTPException(status_code=400, detail="Invalid Last-Event-ID")
    return last_event_id


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.

    Returns:
        ChatResponse: The processed chat response.

    Raises:
        HTTPException: 409 if a streamed response is still being generated for this session, or 500 if
            there's an error processing the request.
    """
    try:
        if await run_stream_service.get_active_run(session.id) is not None:
            raise HTTPException(status_code=409, detail="A streaming response is still in progress for this session")
        logger.info(
            "chat_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )
        # 1. give session a name
        if settings.SESSION_NAMING_ENABLED:
            maybe_name_session(session.id, session.name, chat_request.messages)

        result = await agent.get_response(
            chat_request.messages,
            session_id=session.id,
            user_id=str(session.user_id),
            username=session.username
        )
        logger.info("chat_request_processed", session_id=session.id)
        return ChatResponse(messages=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("chat_request_failed", session_id=session.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
    last_event_id: Optional[str] = Header(default=None),
):
    """Start a chat run and stream its events as SSE; the run survives the client disconnecting.

    The graph runs in the background and buffers every event. Each event carries an ``id:``; if the connection
    drops, re-attach with ``GET /chat/stream/{run_id}`` (``run_id`` is in the ``X-Run-Id`` response header)
    and ``Last-Event-ID`` set to the last id received. Re-sending the *same* message while its run is still in
    flight is also safe: it attaches to the existing run instead of starting a second one.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.
        last_event_id: Only used when attaching to an in-flight run; a new run always streams from its start.

    Returns:
        StreamingResponse: ``text/event-stream`` of ``StreamResponse`` events, ending with ``done: true``.

    Raises:
        HTTPException: 400 for a malformed ``Last-Event-ID``; 409 (with the in-flight ``run_id``) if the session
            is busy with a different message; 500 on an unexpected error.
    """
    try:
        after_id = _resume_point(last_event_id)
        logger.info(
            "stream_chat_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )

        # Capture plain values now: the background run outlives this request (and the auth DB session).
        session_id, user_id, username = session.id, str(session.user_id), session.username
        messages = chat_request.messages

        async def agent_stream():
            with llm_stream_duration_seconds.labels(model=agent.llm_service.get_llm().get_name()).time():
                async for chunk in agent.get_stream_response(messages, session_id, user_id=user_id, username=username):
                    yield chunk

        handle = await run_stream_service.start(session_id, messages[-1].content, agent_stream)
        if handle.outcome == "conflict":
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "A response is already being generated for this session",
                    "run_id": handle.run_id,
                },
            )
        if handle.outcome == "started":
            after_id = START_ID  # a header left over from an earlier run must not skip this one's events
            if settings.SESSION_NAMING_ENABLED:
                maybe_name_session(session_id, session.name, messages)

        return _follow_response(handle.run_id, after_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "stream_chat_request_failed",
            session_id=session.id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))

# Declared before "/chat/stream/{run_id}" so "active" is not captured as a run id.
@router.get("/chat/stream/active", response_model=ActiveRunResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_resume"][0])
async def get_active_stream(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Report the run still generating a response for this session (e.g. after a page reload).

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        ActiveRunResponse: The in-flight run to re-attach to.

    Raises:
        HTTPException: 404 if nothing is being generated.
    """
    run_id = await run_stream_service.get_active_run(session.id)
    if run_id is None:
        raise HTTPException(status_code=404, detail="No response is in progress for this session")
    return ActiveRunResponse(run_id=run_id)

@router.get("/chat/stream/{run_id}")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_resume"][0])
async def resume_chat_stream(
    request: Request,
    run_id: str,
    session: Session = Depends(get_current_session),
    last_event_id: Optional[str] = Header(default=None),
):
    """Re-attach to a run and continue from the last event the client received.

    Works while the run is generating and for a while after it finishes (the buffered tail is replayed), so a
    client that missed the end of the stream can still collect it.

    Args:
        request: The FastAPI request object for rate limiting.
        run_id: The run to follow, from the ``X-Run-Id`` header of the original response.
        session: The current session from the auth token.
        last_event_id: Last event id already received; omit to replay from the start.

    Returns:
        StreamingResponse: ``text/event-stream`` of the events after ``last_event_id``.

    Raises:
        HTTPException: 400 for a malformed ``Last-Event-ID``; 404 if the run is unknown, expired, or belongs to
            another session.
    """
    after_id = _resume_point(last_event_id)
    if not is_valid_run_id(run_id) or await run_stream_service.get_run(run_id, session.id) is None:
        raise HTTPException(status_code=404, detail="Run not found or expired")
    logger.info("stream_resume_requested", session_id=session.id, run_id=run_id, after_id=after_id)
    return _follow_response(run_id, after_id)

@router.get("/messages", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def get_session_messages(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Get all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        ChatResponse: All messages in the session.

    Raises:
        HTTPException: If there's an error retrieving the messages.
    """
    try:
        messages = await agent.get_chat_history(session.id)
        return ChatResponse(messages=messages)
    except Exception as e:
        logger.exception("get_messages_failed", session_id=session.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/messages")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def clear_chat_history(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Clear all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        dict: A message indicating the chat history was cleared.
    """
    try:
        await agent.clear_chat_history(session.id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.exception("clear_chat_history_failed", session_id=session.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
