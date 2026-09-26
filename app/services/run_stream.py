"""Resumable chat streaming: generation is decoupled from the HTTP connection.

Without this, the client owns the graph run's lifetime: a disconnect cancels ``graph.astream`` mid-node, the
tokens already sent are lost, and the user has to start over. Here the graph runs in a background task that
appends every chunk to a per-run event buffer, and HTTP responses are just *followers* of that buffer:

    POST /chat/stream            start a run (or re-attach to the identical in-flight one), follow from the start
    GET  /chat/stream/{run_id}   re-attach, following from the ``Last-Event-ID`` the client last saw
    GET  /chat/stream/active     "is a run in flight for my session?" (e.g. after a page reload)

Buffer backends (same interface, chosen like ``app.core.cache``):

* ``ValkeyRunBackend``   -- Redis Streams. Event ids are stream ids, so "everything after X" is exactly
                            ``XREAD``; runs survive being followed from a different app instance.
* ``InMemoryRunBackend`` -- per-process fallback for development. Re-attach only works on the process that
                            owns the run, so run more than one worker/instance only with Valkey configured.

One run per session at a time is enforced with a lease (``SET NX PX``) that the producer renews while it
works. If the producer dies without finishing (crash, deploy), the lease expires and followers are told the
run was interrupted instead of hanging; the LangGraph checkpoint is untouched, so re-sending the message
continues from the last completed node (see ``is_orphaned_run_for``).
"""

import abc
import asyncio
import contextlib
import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import (
    AsyncIterator,
    Callable,
    Literal,
    NamedTuple,
    Optional,
)

from app.core.cache import (
    REDIS_AVAILABLE,
    Redis,
)
from app.core.config import settings
from app.core.logging import logger
from app.schemas.chat import StreamResponse

try:
    from redis.exceptions import WatchError
except ImportError:  # redis is an optional dependency, see app.core.cache

    class WatchError(Exception):  # type: ignore[no-redef]
        """Placeholder so ``except WatchError`` is valid when redis isn't installed."""


START_ID = "0"
_EVENT_ID = re.compile(r"[0-9]+(-[0-9]+)?")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
INTERRUPTED_MESSAGE = "The response was interrupted before it finished. Please try again."


def is_valid_event_id(value: str) -> bool:
    """Whether ``value`` is an event id this module could have issued (``"0"`` means "from the start")."""
    return bool(_EVENT_ID.fullmatch(value))


def is_valid_run_id(value: str) -> bool:
    """Whether ``value`` has the shape of a run id (guards against arbitrary strings reaching the backend)."""
    return bool(_RUN_ID.fullmatch(value))


def _payload(content: str, done: bool) -> dict:
    """One SSE event body, in the same shape the endpoint has always sent."""
    return StreamResponse(content=content, done=done).model_dump(mode="json")


def _now() -> float:
    return time.monotonic()


class StreamEvent(NamedTuple):
    """One thing a follower should send to the client.

    ``id`` and ``payload`` both set: a buffered event. ``payload`` only: a synthetic terminal event that is not
    in the buffer (producer lost). Neither: a keep-alive.
    """

    id: Optional[str]
    payload: Optional[dict]


class RunBackend(abc.ABC):
    """Storage for run events, run metadata and the per-session lease."""

    name: str
    lock_ttl: float

    async def initialize(self) -> None:
        """Connect / prepare the backend."""

    async def close(self) -> None:
        """Release backend resources."""

    @abc.abstractmethod
    async def acquire(self, session_id: str, run_id: str, trigger_hash: str) -> Optional[str]:
        """Claim the session's single run slot and create the run.

        Returns:
            ``None`` if this call now owns the slot, otherwise the ``run_id`` currently holding it.
        """

    @abc.abstractmethod
    async def renew(self, session_id: str, run_id: str) -> bool:
        """Extend the lease (and the data's lifetime) if ``run_id`` still holds it."""

    @abc.abstractmethod
    async def append(self, run_id: str, payload: dict) -> str:
        """Append an event and return its id (ids sort in append order)."""

    @abc.abstractmethod
    async def finish(self, session_id: str, run_id: str, status: str, terminal: dict) -> None:
        """Append the terminal event, record ``status``, start the retention clock, release the lease."""

    @abc.abstractmethod
    async def read(self, run_id: str, after_id: str, block_ms: int) -> list[tuple[str, dict]]:
        """Events strictly after ``after_id``; waits up to ``block_ms`` for one if there are none (0 = don't wait)."""

    @abc.abstractmethod
    async def get_meta(self, run_id: str) -> Optional[dict]:
        """``{"session_id", "trigger_hash", "status"}`` or ``None`` if unknown / expired."""

    @abc.abstractmethod
    async def active_run(self, session_id: str) -> Optional[str]:
        """The run currently holding the session's slot, if it is still running."""

    @abc.abstractmethod
    async def is_alive(self, run_id: str) -> bool:
        """Running *and* still holding its lease (a dead producer stops renewing)."""


class InMemoryRunBackend(RunBackend):
    """Process-local backend. Event ids are ``"1"``, ``"2"``, ... per run."""

    name = "memory"

    def __init__(
        self,
        lock_ttl: Optional[float] = None,
        retention: Optional[int] = None,
        max_run: Optional[int] = None,
    ):
        self.lock_ttl = settings.STREAM_LOCK_TTL_SECONDS if lock_ttl is None else lock_ttl
        self._retention = settings.STREAM_RUN_TTL_SECONDS if retention is None else retention
        self._max_run = settings.STREAM_RUN_MAX_SECONDS if max_run is None else max_run
        self._events: dict[str, list[dict]] = {}
        self._meta: dict[str, dict] = {}
        self._expires_at: dict[str, float] = {}
        self._locks: dict[str, tuple[str, float]] = {}
        self._conds: dict[str, asyncio.Condition] = {}

    def _purge(self) -> None:
        now = _now()
        for run_id in [r for r, expires in self._expires_at.items() if expires <= now]:
            for table in (self._events, self._meta, self._expires_at, self._conds):
                table.pop(run_id, None)
        for session_id in [s for s, (_, expires) in self._locks.items() if expires <= now]:
            del self._locks[session_id]

    def _holder(self, session_id: str) -> Optional[str]:
        held = self._locks.get(session_id)
        return held[0] if held is not None and held[1] > _now() else None

    def _cond(self, run_id: str) -> asyncio.Condition:
        if run_id not in self._conds:
            self._conds[run_id] = asyncio.Condition()
        return self._conds[run_id]

    async def acquire(self, session_id, run_id, trigger_hash):
        self._purge()
        holder = self._holder(session_id)
        if holder is not None:
            return holder
        self._locks[session_id] = (run_id, _now() + self.lock_ttl)
        self._events[run_id] = []
        self._meta[run_id] = {"session_id": session_id, "trigger_hash": trigger_hash, "status": "running"}
        self._expires_at[run_id] = _now() + self._max_run + self._retention
        return None

    async def renew(self, session_id, run_id):
        if self._holder(session_id) != run_id:
            return False
        self._locks[session_id] = (run_id, _now() + self.lock_ttl)
        self._expires_at[run_id] = _now() + self._max_run + self._retention
        return True

    async def append(self, run_id, payload):
        if run_id not in self._events:
            raise LookupError(f"run {run_id} is unknown or expired")
        self._events[run_id].append(payload)
        cond = self._cond(run_id)
        async with cond:
            cond.notify_all()
        return str(len(self._events[run_id]))

    async def finish(self, session_id, run_id, status, terminal):
        await self.append(run_id, terminal)
        self._meta[run_id]["status"] = status
        self._expires_at[run_id] = _now() + self._retention
        if self._holder(session_id) == run_id:
            del self._locks[session_id]

    async def read(self, run_id, after_id, block_ms):
        after = int(after_id)

        def pending() -> list[tuple[str, dict]]:
            return [(str(i), p) for i, p in enumerate(self._events.get(run_id, [])[after:], start=after + 1)]

        entries = pending()
        if entries or not block_ms or run_id not in self._events:
            return entries
        cond = self._cond(run_id)
        try:
            async with cond:
                await asyncio.wait_for(
                    cond.wait_for(lambda: len(self._events.get(run_id, [])) > after), timeout=block_ms / 1000
                )
        except TimeoutError:
            return []
        return pending()

    async def get_meta(self, run_id):
        meta = self._meta.get(run_id)
        if meta is None or self._expires_at.get(run_id, 0) <= _now():
            return None
        return dict(meta)

    async def active_run(self, session_id):
        holder = self._holder(session_id)
        meta = await self.get_meta(holder) if holder else None
        return holder if meta is not None and meta["status"] == "running" else None

    async def is_alive(self, run_id):
        meta = await self.get_meta(run_id)
        return meta is not None and meta["status"] == "running" and self._holder(meta["session_id"]) == run_id


class ValkeyRunBackend(RunBackend):
    """Redis/Valkey backend: one Stream per run (``run:{id}:events``), a hash for metadata, a lease key per session.

    Uses its own client and pool: followers hold a connection for the length of each blocking ``XREAD``, which
    must not starve the cache's pool.
    """

    name = "valkey"

    def __init__(
        self,
        client=None,
        lock_ttl: Optional[float] = None,
        retention: Optional[int] = None,
        max_run: Optional[int] = None,
    ):
        self._client = client
        self._owns_client = client is None
        self.lock_ttl = settings.STREAM_LOCK_TTL_SECONDS if lock_ttl is None else lock_ttl
        self._retention = settings.STREAM_RUN_TTL_SECONDS if retention is None else retention
        self._max_run = settings.STREAM_RUN_MAX_SECONDS if max_run is None else max_run

    @staticmethod
    def _events_key(run_id: str) -> str:
        return f"run:{run_id}:events"

    @staticmethod
    def _meta_key(run_id: str) -> str:
        return f"run:{run_id}:meta"

    @staticmethod
    def _lock_key(session_id: str) -> str:
        return f"session:{session_id}:active_run"

    @property
    def _lock_ms(self) -> int:
        return max(int(self.lock_ttl * 1000), 1)

    @property
    def _run_ttl(self) -> int:
        """Seconds to keep a *running* run's keys alive between renewals (EXPIRE needs an integer)."""
        return math.ceil(self._max_run + self._retention)

    @property
    def _retention_ttl(self) -> int:
        return max(math.ceil(self._retention), 1)

    async def initialize(self) -> None:
        if self._client is None:
            self._client = Redis(
                host=settings.VALKEY_HOST,
                port=settings.VALKEY_PORT,
                db=settings.VALKEY_DB,
                password=settings.VALKEY_PASSWORD or None,
                max_connections=settings.VALKEY_MAX_CONNECTIONS,
                decode_responses=True,
            )
        await self._client.ping()

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    async def _refresh_ttls(self, run_id: str, seconds: int) -> None:
        await self._client.expire(self._events_key(run_id), seconds)
        await self._client.expire(self._meta_key(run_id), seconds)

    async def acquire(self, session_id, run_id, trigger_hash):
        lock = self._lock_key(session_id)
        holder = None
        for _ in range(3):  # the holder's lease can expire between our SET and GET
            if await self._client.set(lock, run_id, nx=True, px=self._lock_ms):
                await self._client.hset(
                    self._meta_key(run_id),
                    mapping={"session_id": session_id, "trigger_hash": trigger_hash, "status": "running"},
                )
                await self._client.expire(self._meta_key(run_id), self._run_ttl)
                return None
            holder = await self._client.get(lock)
            if holder is not None:
                return holder
        raise RuntimeError(f"could not acquire or inspect the run lease for session {session_id}")

    async def _lease_op(self, session_id: str, run_id: str, renew: bool) -> bool:
        """Compare-and-set on the lease: only act if ``run_id`` still holds it (WATCH/MULTI, so no Lua needed)."""
        lock = self._lock_key(session_id)
        async with self._client.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(lock)
                if await pipe.get(lock) != run_id:
                    return False
                pipe.multi()
                if renew:
                    pipe.pexpire(lock, self._lock_ms)
                else:
                    pipe.delete(lock)
                await pipe.execute()
                return True
            except WatchError:
                return False  # it changed under us (expired and re-acquired): no longer ours

    async def renew(self, session_id, run_id):
        if not await self._lease_op(session_id, run_id, renew=True):
            return False
        await self._refresh_ttls(run_id, self._run_ttl)
        return True

    async def append(self, run_id, payload):
        return await self._client.xadd(self._events_key(run_id), {"p": json.dumps(payload)})

    async def finish(self, session_id, run_id, status, terminal):
        await self.append(run_id, terminal)
        await self._client.hset(self._meta_key(run_id), "status", status)
        await self._refresh_ttls(run_id, self._retention_ttl)
        await self._lease_op(session_id, run_id, renew=False)

    async def read(self, run_id, after_id, block_ms):
        response = await self._client.xread(
            {self._events_key(run_id): after_id}, count=100, block=block_ms if block_ms > 0 else None
        )
        if not response:
            return []
        return [(event_id, json.loads(fields["p"])) for event_id, fields in response[0][1]]

    async def get_meta(self, run_id):
        meta = await self._client.hgetall(self._meta_key(run_id))
        return dict(meta) if meta else None

    async def active_run(self, session_id):
        holder = await self._client.get(self._lock_key(session_id))
        meta = await self.get_meta(holder) if holder else None
        return holder if meta is not None and meta["status"] == "running" else None

    async def is_alive(self, run_id):
        meta = await self.get_meta(run_id)
        if meta is None or meta["status"] != "running":
            return False
        return await self._client.get(self._lock_key(meta["session_id"])) == run_id


def _create_backend() -> RunBackend:
    if settings.VALKEY_HOST and REDIS_AVAILABLE:
        return ValkeyRunBackend()
    return InMemoryRunBackend()


Outcome = Literal["started", "attached", "conflict"]
StreamFactory = Callable[[], AsyncIterator[str]]


@dataclass(frozen=True)
class RunHandle:
    """Result of ``start``.

    ``started``: a new run was created. ``attached``: the session already has a run in flight for this exact
    message, so ``run_id`` is that run (an idempotent retry). ``conflict``: the session is busy with a
    different message; ``run_id`` is the in-flight run.
    """

    run_id: str
    outcome: Outcome


class RunStreamService:
    """Starts background chat runs and lets any number of HTTP responses follow them."""

    def __init__(self, backend: Optional[RunBackend] = None):
        self._backend = backend if backend is not None else _create_backend()
        self._tasks: set[asyncio.Task] = set()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    async def initialize(self) -> None:
        """Connect the backend; if Valkey is configured but unreachable, degrade to the in-process buffer."""
        try:
            await self._backend.initialize()
        except Exception as e:
            if isinstance(self._backend, InMemoryRunBackend):
                raise
            logger.warning("run_stream_valkey_unavailable_falling_back_to_memory", error=str(e))
            with contextlib.suppress(Exception):
                await self._backend.close()
            self._backend = InMemoryRunBackend()
        logger.info("run_stream_initialized", backend=self._backend.name)

    async def shutdown(self) -> None:
        """Cancel in-flight runs (each records an "interrupted" terminal event first) and close the backend."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._backend.close()

    async def start(self, session_id: str, trigger: str, stream_factory: StreamFactory) -> RunHandle:
        """Start a run for ``session_id`` unless one is already in flight.

        Args:
            session_id: The chat session (LangGraph thread). Only one run per session at a time.
            trigger: The user message that starts the run; an identical retry while the run is in flight
                attaches to it instead of conflicting.
            stream_factory: Called once, inside the background task, to get the async iterator of text chunks.

        Returns:
            RunHandle: What happened and which run to follow.
        """
        run_id = uuid.uuid4().hex
        trigger_hash = hashlib.sha256(trigger.encode()).hexdigest()

        holder = await self._backend.acquire(session_id, run_id, trigger_hash)
        if holder is not None:
            meta = await self._backend.get_meta(holder)
            same = meta is not None and meta["session_id"] == session_id and meta["trigger_hash"] == trigger_hash
            logger.info("run_stream_slot_busy", session_id=session_id, active_run_id=holder, attached=same)
            return RunHandle(holder, "attached" if same else "conflict")

        task = asyncio.create_task(self._produce(session_id, run_id, stream_factory), name=f"chat-run-{run_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.info("run_stream_started", session_id=session_id, run_id=run_id)
        return RunHandle(run_id, "started")

    async def _produce(self, session_id: str, run_id: str, stream_factory: StreamFactory) -> None:
        """Drive the graph to completion regardless of who is listening, recording every chunk."""
        status, terminal = "done", _payload("", True)
        lease = asyncio.create_task(self._keep_lease(session_id, run_id))
        deadline = asyncio.timeout(settings.STREAM_RUN_MAX_SECONDS)
        try:
            async with deadline:
                async for chunk in stream_factory():
                    await self._backend.append(run_id, _payload(chunk, False))
        except asyncio.CancelledError:
            status, terminal = "interrupted", _payload(INTERRUPTED_MESSAGE, True)
            raise
        except Exception as e:
            if isinstance(e, TimeoutError) and deadline.expired():
                logger.error("run_stream_time_limit_exceeded", run_id=run_id, limit=settings.STREAM_RUN_MAX_SECONDS)
                message = f"The response took longer than {settings.STREAM_RUN_MAX_SECONDS}s and was stopped."
            else:
                logger.exception("run_stream_failed", run_id=run_id, session_id=session_id, error=str(e))
                message = str(e)
            status, terminal = "error", _payload(message, True)
        finally:
            lease.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await lease
            try:
                await self._backend.finish(session_id, run_id, status, terminal)
            except Exception as e:
                logger.exception("run_stream_finish_failed", run_id=run_id, error=str(e))

    async def _keep_lease(self, session_id: str, run_id: str) -> None:
        """Renew the session lease while the producer is alive, so a crash is detectable by its expiry."""
        interval = max(self._backend.lock_ttl / 3, 0.01)
        while True:
            await asyncio.sleep(interval)
            try:
                if not await self._backend.renew(session_id, run_id):
                    logger.warning("run_stream_lease_lost", run_id=run_id)
                    return
            except Exception as e:
                logger.warning("run_stream_lease_renew_failed", run_id=run_id, error=str(e))

    async def follow(self, run_id: str, after_id: str = START_ID) -> AsyncIterator[StreamEvent]:
        """Yield the run's events after ``after_id`` until its terminal event.

        Args:
            run_id: The run to follow (may already be finished: the tail is replayed).
            after_id: The last event id the client already has, or ``"0"`` for everything.

        Yields:
            StreamEvent: buffered events, keep-alives while idle, and -- if the producer died without
            finishing -- one synthetic terminal event.
        """
        wait_ms = max(int(settings.STREAM_HEARTBEAT_SECONDS * 1000), 1)
        last = after_id
        first_pass = True  # don't block before knowing the run is still going: a finished/expired run must end at once
        while True:
            entries = await self._backend.read(run_id, last, 0 if first_pass else wait_ms)
            if not entries:
                if await self._backend.is_alive(run_id):
                    if not first_pass:
                        yield StreamEvent(None, None)  # nothing new for a heartbeat: keep the connection warm
                    first_pass = False
                    continue
                # finished, expired, or dead. Re-read once: the terminal event may have landed just now.
                entries = await self._backend.read(run_id, last, 0)
                if not entries:
                    meta = await self._backend.get_meta(run_id)
                    if meta is not None and meta["status"] == "running":
                        logger.warning("run_stream_producer_lost", run_id=run_id)
                        yield StreamEvent(None, _payload(INTERRUPTED_MESSAGE, True))
                    return
            first_pass = False
            for event_id, payload in entries:
                last = event_id
                yield StreamEvent(event_id, payload)
                if payload.get("done"):
                    return

    async def get_run(self, run_id: str, session_id: str) -> Optional[dict]:
        """The run's metadata, but only if it belongs to ``session_id`` (others get ``None``, never a leak)."""
        if not is_valid_run_id(run_id):
            return None
        meta = await self._backend.get_meta(run_id)
        return meta if meta is not None and meta["session_id"] == session_id else None

    async def get_active_run(self, session_id: str) -> Optional[str]:
        """The run id currently in flight for the session, if any."""
        return await self._backend.active_run(session_id)


run_stream_service = RunStreamService()
