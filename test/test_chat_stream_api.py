"""HTTP-level tests for resumable chat streaming (``app/api/v1/chatbot.py``).

Starlette's ``TestClient`` buffers whole responses, so it can't model a client that drops mid-stream. These tests
drive the ASGI app directly and deliver a real ``http.disconnect`` after N body chunks, which is what makes
Starlette cancel the streaming response.
"""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1 import chatbot
from app.api.v1.auth import get_current_session
from app.core.limiter import limiter
from app.services.run_stream import (
    InMemoryRunBackend,
    RunStreamService,
)

SESSION_A = SimpleNamespace(id="session-a", user_id=1, username="ann", name="")
SESSION_B = SimpleNamespace(id="session-b", user_id=2, username="bob", name="")
PREFIX = "/api/v1/chatbot"
QUESTION = {"messages": [{"role": "user", "content": "What is the weather?"}]}


class FakeAgent:
    """Streams ``first``, waits on ``gate``, then streams ``second`` -- like a graph with a slow step in the middle."""

    def __init__(self, first, second=(), gate=None, fail_with=None):
        self.first, self.second, self.gate, self.fail_with = list(first), list(second), gate, fail_with
        self.calls = 0
        self.llm_service = SimpleNamespace(get_llm=lambda: SimpleNamespace(get_name=lambda: "fake-model"))

    async def get_stream_response(self, messages, session_id, user_id=None, username=None):
        self.calls += 1
        for chunk in self.first:
            yield chunk
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_with is not None:
            raise self.fail_with
        for chunk in self.second:
            yield chunk


class Response:
    def __init__(self, status, headers, chunks):
        self.status, self.headers, self.chunks = status, headers, chunks

    @property
    def text(self):
        return "".join(self.chunks)

    @property
    def json(self):
        return json.loads(self.text)

    @property
    def events(self):
        """SSE events as ``{"id": ..., "data": {...}}``; comment lines (keep-alives) are dropped."""
        events = []
        for block in self.text.split("\n\n"):
            fields = dict(line.split(": ", 1) for line in block.splitlines() if line and not line.startswith(":"))
            if "data" in fields:
                events.append({"id": fields.get("id"), "data": json.loads(fields["data"])})
        return events

    @property
    def contents(self):
        return [event["data"]["content"] for event in self.events]

    @property
    def last_id(self):
        return self.events[-1]["id"]


async def call(app, method, path, body=None, headers=None, disconnect_after=None):
    """Run one request through the ASGI app, optionally dropping the client after N body chunks."""
    raw = json.dumps(body).encode() if body is not None else b""
    request_headers = [(b"host", b"test"), (b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]
    request_headers += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "path": path,
        "raw_path": path.encode(), "query_string": b"", "headers": request_headers, "scheme": "http",
        "server": ("test", 80), "client": ("127.0.0.1", 5000), "root_path": "", "app": app,
    }
    sent_request = False
    disconnected = asyncio.Event()
    state = {"status": None, "headers": {}, "chunks": []}

    async def receive():
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            state["status"] = message["status"]
            state["headers"] = {k.decode().lower(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            if message.get("body"):
                state["chunks"].append(message["body"].decode())
            if disconnect_after is not None and len(state["chunks"]) >= disconnect_after:
                disconnected.set()

    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return Response(state["status"], state["headers"], state["chunks"])


@pytest.fixture
def harness(monkeypatch):
    """A FastAPI app with only the chatbot router, a fresh in-memory run service, and switchable auth."""
    limiter.reset()
    current = {"session": SESSION_A}
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(chatbot.router, prefix=PREFIX)
    app.dependency_overrides[get_current_session] = lambda: current["session"]

    service = RunStreamService(InMemoryRunBackend(lock_ttl=5))
    named = []
    monkeypatch.setattr(chatbot, "run_stream_service", service)
    monkeypatch.setattr(chatbot, "maybe_name_session", lambda *args: named.append(args))

    def use_agent(agent):
        monkeypatch.setattr(chatbot, "agent", agent)
        return agent

    return SimpleNamespace(app=app, service=service, use_agent=use_agent, current=current, named=named)


def run(coro):
    return asyncio.run(coro)


def stream(h, body=QUESTION, **kwargs):
    return call(h.app, "POST", f"{PREFIX}/chat/stream", body, **kwargs)


def resume(h, run_id, **kwargs):
    return call(h.app, "GET", f"{PREFIX}/chat/stream/{run_id}", **kwargs)


class TestStreaming:
    def test_events_carry_ids_a_run_id_header_and_end_with_done(self, harness):
        agent = harness.use_agent(FakeAgent(["The ", "weather ", "is sunny"]))

        async def scenario():
            response = await stream(harness)
            await harness.service.shutdown()
            return response

        response = run(scenario())

        assert response.status == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-run-id"])
        assert response.headers["x-accel-buffering"] == "no"
        assert "no-cache" in response.headers["cache-control"]
        assert response.contents == ["The ", "weather ", "is sunny", ""]
        assert [e["data"]["done"] for e in response.events] == [False, False, False, True]
        ids = [e["id"] for e in response.events]
        assert all(ids) and ids == sorted(ids, key=int)  # every event has an id, in order
        assert agent.calls == 1

    def test_event_data_keeps_the_original_schema(self, harness):
        harness.use_agent(FakeAgent(["hi"]))

        async def scenario():
            response = await stream(harness)
            await harness.service.shutdown()
            return response

        data = run(scenario()).events[0]["data"]

        assert set(data) >= {"content", "done", "request_id"}

    def test_a_new_run_names_the_session(self, harness):
        harness.use_agent(FakeAgent(["hi"]))

        async def scenario():
            await stream(harness)
            await harness.service.shutdown()

        run(scenario())

        assert len(harness.named) == 1 and harness.named[0][0] == "session-a"


class TestDisconnectAndResume:
    def test_reconnecting_after_a_drop_continues_from_the_last_event_with_no_gaps_or_repeats(self, harness):
        gate = asyncio.Event()
        agent = harness.use_agent(FakeAgent(["Hel", "lo ", "wor"], ["ld", "!"], gate))

        async def scenario():
            dropped = await stream(harness, disconnect_after=2)  # the network dies after two events
            run_id = dropped.headers["x-run-id"]
            still_running = await harness.service.get_active_run("session-a")
            gate.set()  # the graph finishes while nobody is connected
            resumed = await resume(harness, run_id, headers={"Last-Event-ID": dropped.last_id})
            await harness.service.shutdown()
            return dropped, run_id, still_running, resumed

        dropped, run_id, still_running, resumed = run(scenario())

        # The client may have received a few more buffered events than it asked to stop at (they were already in
        # flight when it dropped) -- what matters is that "what it got" + "what resume sends" is exactly the stream.
        assert dropped.contents[:2] == ["Hel", "lo "]
        assert dropped.events[-1]["data"]["done"] is False  # it really dropped mid-run
        assert still_running == run_id  # the disconnect did not cancel the run
        assert dropped.contents + resumed.contents == ["Hel", "lo ", "wor", "ld", "!", ""]
        assert "".join(dropped.contents + resumed.contents) == "Hello world!"
        assert agent.calls == 1  # one generation, not one per connection

    def test_resuming_twice_in_a_row_from_successive_ids(self, harness):
        gate = asyncio.Event()
        harness.use_agent(FakeAgent(["a", "b"], ["c", "d"], gate))

        async def scenario():
            first = await stream(harness, disconnect_after=1)
            gate.set()
            run_id = first.headers["x-run-id"]
            second = await resume(harness, run_id, headers={"Last-Event-ID": first.last_id}, disconnect_after=1)
            third = await resume(harness, run_id, headers={"Last-Event-ID": second.last_id})
            await harness.service.shutdown()
            return first, second, third

        first, second, third = run(scenario())

        assert first.contents + second.contents + third.contents == ["a", "b", "c", "d", ""]

    def test_a_finished_run_can_still_be_collected_after_the_stream_ended(self, harness):
        harness.use_agent(FakeAgent(["a", "b", "c"]))

        async def scenario():
            complete = await stream(harness)
            replay_all = await resume(harness, complete.headers["x-run-id"])
            replay_tail = await resume(
                harness, complete.headers["x-run-id"], headers={"Last-Event-ID": complete.events[0]["id"]}
            )
            await harness.service.shutdown()
            return complete, replay_all, replay_tail

        complete, replay_all, replay_tail = run(scenario())

        assert replay_all.contents == complete.contents == ["a", "b", "c", ""]
        assert replay_tail.contents == ["b", "c", ""]

    def test_resuming_from_the_final_event_returns_immediately_and_empty(self, harness):
        harness.use_agent(FakeAgent(["a"]))

        async def scenario():
            complete = await stream(harness)
            after_end = await resume(harness, complete.headers["x-run-id"], headers={"Last-Event-ID": complete.last_id})
            await harness.service.shutdown()
            return after_end

        after_end = run(scenario())

        assert after_end.status == 200 and after_end.events == []


class TestRetryingTheSameMessage:
    def test_resending_the_same_message_attaches_to_the_run_in_flight(self, harness):
        gate = asyncio.Event()
        agent = harness.use_agent(FakeAgent(["a", "b"], ["c"], gate))

        async def scenario():
            first = await stream(harness, disconnect_after=1)
            # naive client: just re-POST the same message while the run is still going
            retrying = asyncio.create_task(stream(harness, headers={"Last-Event-ID": first.last_id}))
            await asyncio.sleep(0.05)
            gate.set()
            retry = await retrying
            await harness.service.shutdown()
            return first, retry

        first, retry = run(scenario())

        assert retry.headers["x-run-id"] == first.headers["x-run-id"]
        assert first.contents + retry.contents == ["a", "b", "c", ""]
        assert agent.calls == 1  # no second generation on the same thread
        assert len(harness.named) == 1  # nor a second naming attempt

    def test_a_different_message_while_a_run_is_in_flight_is_a_409_naming_the_run(self, harness):
        gate = asyncio.Event()
        harness.use_agent(FakeAgent(["a"], ["b"], gate))
        other = {"messages": [{"role": "user", "content": "Something else entirely"}]}

        async def scenario():
            first = await stream(harness, disconnect_after=1)
            conflict = await stream(harness, other)
            gate.set()
            await harness.service.shutdown()
            return first, conflict

        first, conflict = run(scenario())

        assert conflict.status == 409
        assert conflict.json["detail"]["run_id"] == first.headers["x-run-id"]

    def test_a_stale_last_event_id_does_not_make_a_brand_new_run_skip_its_events(self, harness):
        harness.use_agent(FakeAgent(["a", "b"]))

        async def scenario():
            response = await stream(harness, headers={"Last-Event-ID": "999999"})
            await harness.service.shutdown()
            return response

        assert run(scenario()).contents == ["a", "b", ""]

    def test_the_session_is_free_again_once_the_run_ends(self, harness):
        agent = harness.use_agent(FakeAgent(["a"]))

        async def scenario():
            first = await stream(harness)
            second = await stream(harness)  # same message again, after the first completed: a genuine new turn
            await harness.service.shutdown()
            return first, second

        first, second = run(scenario())

        assert second.headers["x-run-id"] != first.headers["x-run-id"]
        assert agent.calls == 2


class TestOtherEndpointsAgree:
    def test_non_streaming_chat_is_refused_while_a_stream_is_generating(self, harness):
        gate = asyncio.Event()
        harness.use_agent(FakeAgent(["a"], ["b"], gate))

        async def scenario():
            await stream(harness, disconnect_after=1)
            blocked = await call(harness.app, "POST", f"{PREFIX}/chat", QUESTION)
            gate.set()
            await harness.service.shutdown()
            return blocked

        blocked = run(scenario())

        assert blocked.status == 409
        assert "in progress" in blocked.json["detail"]

    def test_active_run_endpoint(self, harness):
        gate = asyncio.Event()
        harness.use_agent(FakeAgent(["a"], ["b"], gate))
        active = f"{PREFIX}/chat/stream/active"

        async def scenario():
            idle = await call(harness.app, "GET", active)
            first = await stream(harness, disconnect_after=1)
            running = await call(harness.app, "GET", active)
            gate.set()
            await resume(harness, first.headers["x-run-id"])  # wait for completion
            finished = await call(harness.app, "GET", active)
            await harness.service.shutdown()
            return idle, first, running, finished

        idle, first, running, finished = run(scenario())

        assert idle.status == 404
        assert running.status == 200 and running.json["run_id"] == first.headers["x-run-id"]
        assert finished.status == 404

    def test_active_is_not_mistaken_for_a_run_id(self, harness):
        async def scenario():
            return await call(harness.app, "GET", f"{PREFIX}/chat/stream/active")

        response = run(scenario())

        assert response.status == 404
        assert "No response is in progress" in response.json["detail"]


class TestErrorsAndAccess:
    def test_a_graph_failure_arrives_as_a_terminal_event(self, harness):
        harness.use_agent(FakeAgent(["partial"], fail_with=RuntimeError("model exploded")))

        async def scenario():
            response = await stream(harness)
            await harness.service.shutdown()
            return response

        response = run(scenario())

        assert response.status == 200
        assert response.contents == ["partial", "model exploded"]
        assert response.events[-1]["data"]["done"] is True

    def test_resuming_an_unknown_run_is_a_404(self, harness):
        async def scenario():
            return await resume(harness, "f" * 32)

        assert run(scenario()).status == 404

    @pytest.mark.parametrize("run_id", ["not-a-run-id", "A" * 32, "..%2f..", "a" * 31])
    def test_malformed_run_ids_are_a_404(self, harness, run_id):
        async def scenario():
            return await resume(harness, run_id)

        assert run(scenario()).status == 404

    def test_another_sessions_run_looks_like_it_does_not_exist(self, harness):
        harness.use_agent(FakeAgent(["secret answer"]))

        async def scenario():
            mine = await stream(harness)
            harness.current["session"] = SESSION_B
            theirs = await resume(harness, mine.headers["x-run-id"])
            harness.current["session"] = SESSION_A
            await harness.service.shutdown()
            return theirs

        theirs = run(scenario())

        assert theirs.status == 404
        assert "secret answer" not in theirs.text

    @pytest.mark.parametrize("bad_id", ["abc", "-1", "1-", "$", "0-0; DEL x", "١٢"])
    def test_malformed_last_event_id_is_a_400_on_both_endpoints(self, harness, bad_id):
        harness.use_agent(FakeAgent(["a"]))

        async def scenario():
            post = await stream(harness, headers={"Last-Event-ID": bad_id})
            started = await stream(harness)
            get = await resume(harness, started.headers["x-run-id"], headers={"Last-Event-ID": bad_id})
            await harness.service.shutdown()
            return post, get

        post, get = run(scenario())

        assert post.status == 400 and get.status == 400

    def test_two_sessions_can_stream_at_the_same_time(self, harness):
        gate = asyncio.Event()
        agent = harness.use_agent(FakeAgent(["a"], ["b"], gate))

        async def scenario():
            a = await stream(harness, disconnect_after=1)
            harness.current["session"] = SESSION_B
            b = await stream(harness, disconnect_after=1)
            gate.set()
            await harness.service.shutdown()
            return a, b

        a, b = run(scenario())

        assert a.headers["x-run-id"] != b.headers["x-run-id"]
        assert agent.calls == 2
