"""Tests for ``app/services/run_stream.py``: run buffering, resume, leases and failure handling.

Every backend test runs against both the in-memory backend and the Valkey backend (on ``fakeredis``, so the
Redis-Streams code path is exercised without a server).
"""

import asyncio
import contextlib

import fakeredis
import pytest

from app.core.config import settings
from app.services import run_stream
from app.services.run_stream import (
    INTERRUPTED_MESSAGE,
    InMemoryRunBackend,
    RunStreamService,
    StreamEvent,
    ValkeyRunBackend,
    is_valid_event_id,
    is_valid_run_id,
)


def make_backend(kind, **kwargs):
    kwargs.setdefault("lock_ttl", 5)
    if kind == "memory":
        return InMemoryRunBackend(**kwargs)
    client = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    return ValkeyRunBackend(client=client, **kwargs)


@pytest.fixture(params=["memory", "valkey"])
def kind(request):
    return request.param


def run(coro):
    return asyncio.run(coro)


def content(events):
    return [e.payload["content"] for e in events if e.payload is not None]


async def collect(service, run_id, after_id="0"):
    return [event async for event in service.follow(run_id, after_id)]


class TestBackend:
    def test_acquire_gives_one_run_per_session(self, kind):
        async def scenario():
            backend = make_backend(kind)
            first = await backend.acquire("s1", "run-a", "h")
            second = await backend.acquire("s1", "run-b", "h")
            other_session = await backend.acquire("s2", "run-c", "h")
            return first, second, other_session

        assert run(scenario()) == (None, "run-a", None)

    def test_events_get_increasing_ids_and_read_returns_only_later_ones(self, kind):
        async def scenario():
            backend = make_backend(kind)
            await backend.acquire("s1", "r1", "h")
            ids = [await backend.append("r1", {"content": c, "done": False}) for c in "abc"]
            everything = await backend.read("r1", "0", 0)
            after_first = await backend.read("r1", ids[0], 0)
            after_last = await backend.read("r1", ids[-1], 0)
            return ids, everything, after_first, after_last

        ids, everything, after_first, after_last = run(scenario())

        assert [payload["content"] for _, payload in everything] == ["a", "b", "c"]
        assert [event_id for event_id, _ in everything] == ids
        assert [payload["content"] for _, payload in after_first] == ["b", "c"]
        assert after_last == []

    def test_blocking_read_wakes_up_when_an_event_arrives(self, kind):
        async def scenario():
            backend = make_backend(kind)
            await backend.acquire("s1", "r1", "h")
            reader = asyncio.create_task(backend.read("r1", "0", 2000))
            await asyncio.sleep(0.05)
            await backend.append("r1", {"content": "late", "done": False})
            return await asyncio.wait_for(reader, 1)

        assert [payload["content"] for _, payload in run(scenario())] == ["late"]

    def test_blocking_read_times_out_empty(self, kind):
        async def scenario():
            backend = make_backend(kind)
            await backend.acquire("s1", "r1", "h")
            return await backend.read("r1", "0", 50)

        assert run(scenario()) == []

    def test_finish_appends_terminal_frees_the_slot_and_records_status(self, kind):
        async def scenario():
            backend = make_backend(kind)
            await backend.acquire("s1", "r1", "h")
            running = (await backend.active_run("s1"), await backend.is_alive("r1"))
            await backend.finish("s1", "r1", "done", {"content": "", "done": True})
            return (
                running,
                await backend.active_run("s1"),
                await backend.is_alive("r1"),
                (await backend.get_meta("r1"))["status"],
                [payload["done"] for _, payload in await backend.read("r1", "0", 0)],
                await backend.acquire("s1", "r2", "h"),  # slot is free again
            )

        running, active_after, alive_after, status, done_flags, reacquire = run(scenario())

        assert running == ("r1", True)
        assert (active_after, alive_after, status, done_flags, reacquire) == (None, False, "done", [True], None)

    def test_meta_carries_session_and_trigger_hash(self, kind):
        async def scenario():
            backend = make_backend(kind)
            await backend.acquire("s1", "r1", "hash-1")
            return await backend.get_meta("r1"), await backend.get_meta("nope")

        meta, missing = run(scenario())

        assert meta == {"session_id": "s1", "trigger_hash": "hash-1", "status": "running"}
        assert missing is None

    def test_a_run_that_stops_renewing_is_no_longer_alive(self, kind):
        async def scenario():
            backend = make_backend(kind, lock_ttl=0.2)
            await backend.acquire("s1", "r1", "h")
            before = await backend.is_alive("r1")
            await asyncio.sleep(0.35)
            return before, await backend.is_alive("r1"), await backend.active_run("s1")

        assert run(scenario()) == (True, False, None)

    def test_renew_keeps_the_lease_alive(self, kind):
        async def scenario():
            backend = make_backend(kind, lock_ttl=0.3)
            await backend.acquire("s1", "r1", "h")
            for _ in range(4):
                await asyncio.sleep(0.15)
                assert await backend.renew("s1", "r1") is True
            return await backend.is_alive("r1")

        assert run(scenario()) is True

    def test_only_the_lease_holder_can_renew(self, kind):
        async def scenario():
            backend = make_backend(kind)
            await backend.acquire("s1", "r1", "h")
            stranger_renew = await backend.renew("s1", "not-the-holder")
            return stranger_renew, await backend.active_run("s1")

        assert run(scenario()) == (False, "r1")

    def test_finishing_a_stale_run_does_not_release_the_new_holders_slot(self, kind):
        async def scenario():
            backend = make_backend(kind, lock_ttl=0.15)
            await backend.acquire("s1", "old", "h")
            await asyncio.sleep(0.3)  # old's lease expires without a renew
            assert await backend.acquire("s1", "new", "h") is None
            await backend.finish("s1", "old", "done", {"content": "", "done": True})  # late, from the old run
            return await backend.active_run("s1")

        assert run(scenario()) == "new"


class TestValkeyBackendDetails:
    def test_events_are_stored_in_a_redis_stream_and_expire_after_finish(self):
        async def scenario():
            backend = make_backend("valkey", retention=90)
            await backend.acquire("s1", "r1", "h")
            await backend.append("r1", {"content": "a", "done": False})
            await backend.finish("s1", "r1", "done", {"content": "", "done": True})
            client = backend._client
            return (
                await client.type("run:r1:events"),
                await client.xlen("run:r1:events"),
                0 < await client.ttl("run:r1:events") <= 90,
                0 < await client.ttl("run:r1:meta") <= 90,
                await client.exists("session:s1:active_run"),
            )

        assert run(scenario()) == ("stream", 2, True, True, 0)

    def test_lease_is_a_key_with_a_ttl_and_the_run_id_as_value(self):
        async def scenario():
            backend = make_backend("valkey", lock_ttl=30)
            await backend.acquire("s1", "r1", "h")
            client = backend._client
            return await client.get("session:s1:active_run"), 0 < await client.pttl("session:s1:active_run") <= 30000

        assert run(scenario()) == ("r1", True)


class TestValidation:
    @pytest.mark.parametrize("value", ["0", "1", "1790345254065-0", "17-3"])
    def test_valid_event_ids(self, value):
        assert is_valid_event_id(value)

    @pytest.mark.parametrize("value", ["", "abc", "-1", "1-", "1-2-3", "$", ">", "1 ", "1\n", "0-0; DEL x", "١٢"])
    def test_invalid_event_ids(self, value):
        assert not is_valid_event_id(value)

    def test_run_ids_must_be_32_lowercase_hex(self):
        assert is_valid_run_id("a" * 32)
        for bad in ("", "a" * 31, "A" * 32, "g" * 32, "a" * 32 + "\n", "../" + "a" * 29):
            assert not is_valid_run_id(bad)


def scripted(first, second=(), gate=None, fail_with=None):
    """A stream factory that yields ``first``, waits on ``gate``, then yields ``second``."""
    calls = []

    def factory():
        async def gen():
            calls.append(1)
            for chunk in first:
                yield chunk
            if gate is not None:
                await gate.wait()
            if fail_with is not None:
                raise fail_with
            for chunk in second:
                yield chunk

        return gen()

    factory.calls = calls
    return factory


class TestResume:
    def test_reattaching_after_a_disconnect_yields_exactly_the_remainder(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            gate = asyncio.Event()
            handle = await service.start("s1", "hi", scripted(["Hel", "lo ", "wor"], ["ld", "!"], gate))

            seen, last_id = [], "0"
            async for event in service.follow(handle.run_id):  # the client that then "drops"
                if event.payload is None:
                    continue
                seen.append(event.payload["content"])
                last_id = event.id
                if len(seen) == 2:
                    break
            gate.set()
            rest = content(await collect(service, handle.run_id, last_id))
            await service.shutdown()
            return handle, seen, rest

        handle, seen, rest = run(scenario())

        assert handle.outcome == "started"
        assert seen == ["Hel", "lo "]
        assert rest == ["wor", "ld", "!", ""]  # the remainder, then the empty terminal event
        assert "".join(seen + rest) == "Hello world!"

    def test_the_run_keeps_going_with_nobody_listening(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            gate = asyncio.Event()
            handle = await service.start("s1", "hi", scripted(["a"], ["b"], gate))
            await asyncio.sleep(0.05)  # no follower attached at all
            gate.set()
            events = await collect(service, handle.run_id)  # first follower shows up after it already finished
            await service.shutdown()
            return content(events)

        assert run(scenario()) == ["a", "b", ""]

    def test_a_finished_runs_tail_can_be_replayed(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            handle = await service.start("s1", "hi", scripted(["a", "b", "c"]))
            full = await collect(service, handle.run_id)
            replay_from_first = await collect(service, handle.run_id, full[0].id)
            await service.shutdown()
            return content(full), content(replay_from_first)

        full, replay = run(scenario())

        assert full == ["a", "b", "c", ""]
        assert replay == ["b", "c", ""]

    def test_following_from_the_terminal_event_ends_immediately(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            handle = await service.start("s1", "hi", scripted(["a"]))
            full = await collect(service, handle.run_id)
            after_terminal = await asyncio.wait_for(collect(service, handle.run_id, full[-1].id), 1)
            await service.shutdown()
            return after_terminal

        assert run(scenario()) == []

    def test_event_payloads_use_the_existing_stream_response_shape(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            handle = await service.start("s1", "hi", scripted(["a"]))
            events = await collect(service, handle.run_id)
            await service.shutdown()
            return events

        token, terminal = run(scenario())

        assert (token.payload["content"], token.payload["done"]) == ("a", False)
        assert (terminal.payload["content"], terminal.payload["done"]) == ("", True)
        assert "request_id" in token.payload


class TestOneRunPerSession:
    def test_identical_message_while_in_flight_attaches_instead_of_starting_a_second_run(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            gate = asyncio.Event()
            factory = scripted(["a"], ["b"], gate)
            first = await service.start("s1", "same message", factory)
            second = await service.start("s1", "same message", factory)
            gate.set()
            events = await collect(service, second.run_id)
            await service.shutdown()
            return first, second, len(factory.calls), content(events)

        first, second, generations, events = run(scenario())

        assert (first.outcome, second.outcome) == ("started", "attached")
        assert second.run_id == first.run_id
        assert generations == 1
        assert events == ["a", "b", ""]

    def test_a_different_message_conflicts_and_names_the_run_in_flight(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            gate = asyncio.Event()
            first = await service.start("s1", "question one", scripted(["a"], [], gate))
            second = await service.start("s1", "question two", scripted(["x"]))
            gate.set()
            await collect(service, first.run_id)
            await service.shutdown()
            return first, second

        first, second = run(scenario())

        assert second.outcome == "conflict"
        assert second.run_id == first.run_id

    def test_other_sessions_are_independent(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            gate = asyncio.Event()
            one = await service.start("s1", "hi", scripted(["a"], [], gate))
            two = await service.start("s2", "hi", scripted(["b"], [], gate))
            gate.set()
            await collect(service, one.run_id), await collect(service, two.run_id)
            await service.shutdown()
            return one, two

        one, two = run(scenario())

        assert (one.outcome, two.outcome) == ("started", "started")
        assert one.run_id != two.run_id

    def test_a_finished_run_frees_the_session_for_the_next_message(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            first = await service.start("s1", "one", scripted(["a"]))
            await collect(service, first.run_id)
            second = await service.start("s1", "two", scripted(["b"]))
            await collect(service, second.run_id)
            await service.shutdown()
            return first, second

        first, second = run(scenario())

        assert second.outcome == "started"
        assert second.run_id != first.run_id

    def test_simultaneous_starts_create_exactly_one_run(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            gate = asyncio.Event()
            factory = scripted(["a"], [], gate)
            handles = await asyncio.gather(*[service.start("s1", "same", factory) for _ in range(8)])
            gate.set()
            await collect(service, handles[0].run_id)
            await service.shutdown()
            return handles, len(factory.calls)

        handles, generations = run(scenario())

        assert sorted(h.outcome for h in handles) == ["attached"] * 7 + ["started"]
        assert len({h.run_id for h in handles}) == 1
        assert generations == 1

    def test_active_run_tracks_the_run_in_flight(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            gate = asyncio.Event()
            idle = await service.get_active_run("s1")
            handle = await service.start("s1", "hi", scripted(["a"], [], gate))
            running = await service.get_active_run("s1")
            gate.set()
            await collect(service, handle.run_id)
            finished = await service.get_active_run("s1")
            await service.shutdown()
            return handle, idle, running, finished

        handle, idle, running, finished = run(scenario())

        assert (idle, running, finished) == (None, handle.run_id, None)

    def test_get_run_only_reveals_a_session_its_own_runs(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            handle = await service.start("s1", "hi", scripted(["a"]))
            await collect(service, handle.run_id)
            result = (
                await service.get_run(handle.run_id, "s1"),
                await service.get_run(handle.run_id, "someone-else"),
                await service.get_run("f" * 32, "s1"),
                await service.get_run("not a run id", "s1"),
            )
            await service.shutdown()
            return result

        own, foreign, unknown, malformed = run(scenario())

        assert own["session_id"] == "s1"
        assert foreign is None and unknown is None and malformed is None


class TestFailures:
    def test_an_error_in_the_graph_becomes_a_terminal_event_and_frees_the_session(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            handle = await service.start("s1", "hi", scripted(["partial"], fail_with=RuntimeError("llm exploded")))
            events = await collect(service, handle.run_id)
            after = await service.start("s1", "retry", scripted(["ok"]))
            await collect(service, after.run_id)
            meta = await service._backend.get_meta(handle.run_id)
            await service.shutdown()
            return events, after, meta

        events, after, meta = run(scenario())

        assert content(events) == ["partial", "llm exploded"]
        assert events[-1].payload["done"] is True
        assert meta["status"] == "error"
        assert after.outcome == "started"

    def test_shutdown_records_an_interrupted_terminal_event(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            gate = asyncio.Event()  # never set: the run is stuck when the app shuts down
            handle = await service.start("s1", "hi", scripted(["a"], [], gate))
            await asyncio.sleep(0.05)
            await service.shutdown()
            return await service._backend.read(handle.run_id, "0", 0), await service._backend.active_run("s1")

        events, active = run(scenario())

        assert [p["content"] for _, p in events] == ["a", INTERRUPTED_MESSAGE]
        assert events[-1][1]["done"] is True
        assert active is None

    def test_a_run_over_its_time_limit_is_stopped_with_a_message(self, kind, monkeypatch):
        monkeypatch.setattr(settings, "STREAM_RUN_MAX_SECONDS", 0.1)

        async def scenario():
            service = RunStreamService(make_backend(kind))
            handle = await service.start("s1", "hi", scripted(["a"], [], asyncio.Event()))
            events = await asyncio.wait_for(collect(service, handle.run_id), 5)
            active = await service.get_active_run("s1")
            await service.shutdown()
            return events, active

        events, active = run(scenario())

        assert content(events)[0] == "a"
        assert "took longer than" in content(events)[-1]
        assert events[-1].payload["done"] is True
        assert active is None

    def test_a_timeout_raised_inside_the_graph_is_not_mistaken_for_the_run_limit(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            handle = await service.start("s1", "hi", scripted([], fail_with=TimeoutError("upstream timed out")))
            events = await collect(service, handle.run_id)
            await service.shutdown()
            return content(events)

        assert run(scenario()) == ["upstream timed out"]


class TestLiveness:
    def test_followers_get_keepalives_while_the_run_is_quiet(self, kind, monkeypatch):
        monkeypatch.setattr(settings, "STREAM_HEARTBEAT_SECONDS", 0.05)

        async def scenario():
            service = RunStreamService(make_backend(kind))
            gate = asyncio.Event()
            handle = await service.start("s1", "hi", scripted(["a"], ["b"], gate))
            got = []
            async for event in service.follow(handle.run_id):
                got.append(event)
                if sum(1 for e in got if e.payload is None) >= 2:
                    gate.set()
            await service.shutdown()
            return got

        got = run(scenario())

        assert sum(1 for e in got if e == StreamEvent(None, None)) >= 2
        assert content(got) == ["a", "b", ""]

    def test_lease_is_renewed_so_a_long_silent_step_is_not_mistaken_for_a_crash(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind, lock_ttl=0.3))
            gate = asyncio.Event()
            handle = await service.start("s1", "hi", scripted(["a"], ["b"], gate))
            await asyncio.sleep(0.9)  # three lease lifetimes with no events at all (e.g. a slow tool call)
            still_active = await service.get_active_run("s1")
            gate.set()
            events = await collect(service, handle.run_id)
            await service.shutdown()
            return still_active, handle.run_id, content(events)

        still_active, run_id, events = run(scenario())

        assert still_active == run_id
        assert events == ["a", "b", ""]

    def test_followers_are_told_when_the_producer_died_instead_of_hanging(self, kind, monkeypatch):
        monkeypatch.setattr(settings, "STREAM_HEARTBEAT_SECONDS", 0.05)

        async def scenario():
            backend = make_backend(kind, lock_ttl=0.2)
            service = RunStreamService(backend)
            # A run whose producer vanished (e.g. the instance was killed): a lease and one event, no renewals.
            await backend.acquire("s1", "r1", "h")
            await backend.append("r1", {"content": "a", "done": False})
            events = await asyncio.wait_for(collect(service, "r1"), 5)
            meta = await backend.get_meta("r1")
            return events, meta

        events, meta = run(scenario())

        assert content(events)[0] == "a"
        assert events[-1].id is None  # synthetic: not in the buffer
        assert events[-1].payload["content"] == INTERRUPTED_MESSAGE and events[-1].payload["done"] is True
        assert any(e == StreamEvent(None, None) for e in events)  # it waited (keep-alives) before giving up
        assert meta["status"] == "running"  # followers don't rewrite history

    def test_a_dead_producers_session_can_start_a_new_run_once_the_lease_expires(self, kind):
        async def scenario():
            backend = make_backend(kind, lock_ttl=0.15)
            service = RunStreamService(backend)
            await backend.acquire("s1", "ghost", "h")
            blocked = await service.start("s1", "retry", scripted(["x"]))
            await asyncio.sleep(0.3)
            recovered = await service.start("s1", "retry", scripted(["y"]))
            events = await collect(service, recovered.run_id)
            await service.shutdown()
            return blocked, recovered, content(events)

        blocked, recovered, events = run(scenario())

        assert blocked.outcome == "conflict"
        assert recovered.outcome == "started"
        assert events == ["y", ""]

    def test_following_an_expired_run_ends_quietly(self, kind):
        async def scenario():
            service = RunStreamService(make_backend(kind))
            return await asyncio.wait_for(collect(service, "f" * 32), 1)

        assert run(scenario()) == []


class TestInMemoryRetention:
    def test_finished_runs_are_purged_after_the_retention_period(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(run_stream, "_now", lambda: now[0])

        async def scenario():
            backend = InMemoryRunBackend(lock_ttl=30, retention=60, max_run=600)
            await backend.acquire("s1", "r1", "h")
            await backend.finish("s1", "r1", "done", {"content": "", "done": True})
            kept = await backend.get_meta("r1")
            now[0] += 61
            await backend.acquire("s1", "r2", "h")  # any acquire purges
            return kept, await backend.get_meta("r1"), "r1" in backend._events

        kept, expired, still_buffered = run(scenario())

        assert kept is not None
        assert expired is None
        assert still_buffered is False


class TestInitialize:
    def test_unreachable_valkey_falls_back_to_the_in_memory_buffer(self):
        class DeadClient:
            async def ping(self):
                raise ConnectionError("valkey is down")

        async def scenario():
            service = RunStreamService(ValkeyRunBackend(client=DeadClient()))
            await service.initialize()
            handle = await service.start("s1", "hi", scripted(["a"]))
            events = await collect(service, handle.run_id)
            await service.shutdown()
            return service.backend_name, content(events)

        assert run(scenario()) == ("memory", ["a", ""])

    def test_backend_is_chosen_from_configuration(self, monkeypatch):
        monkeypatch.setattr(settings, "VALKEY_HOST", "")
        assert run_stream._create_backend().name == "memory"
        monkeypatch.setattr(settings, "VALKEY_HOST", "valkey.internal")
        assert run_stream._create_backend().name == "valkey"


def test_no_tasks_are_left_running_after_shutdown():
    async def scenario():
        service = RunStreamService(make_backend("memory"))
        await service.start("s1", "hi", scripted(["a"], [], asyncio.Event()))
        await service.shutdown()
        with contextlib.suppress(Exception):
            await asyncio.sleep(0)
        return len(service._tasks)

    assert run(scenario()) == 0
