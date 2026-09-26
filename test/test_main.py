"""Tests for the FastAPI app assembled in ``app/main.py``."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app import main


class Recorder:
    """Fake service that records lifecycle calls into a shared, ordered log."""

    def __init__(self, name, log, fail_on=()):
        self.name, self.log, self.fail_on = name, log, set(fail_on)

    async def _step(self, action):
        self.log.append(f"{self.name}.{action}")
        if action in self.fail_on:
            raise RuntimeError(f"{self.name} {action} failed")

    async def initialize(self):
        await self._step("initialize")

    async def create_graph(self):
        await self._step("create_graph")

    async def shutdown(self):
        await self._step("shutdown")

    async def close(self):
        await self._step("close")


def run_lifespan(monkeypatch, cache_fail=(), memory_fail=(), agent_fail=(), stream_fail=()):
    log = []
    monkeypatch.setattr(main, "cache_service", Recorder("cache", log, cache_fail))
    monkeypatch.setattr(main, "run_stream_service", Recorder("stream", log, stream_fail))
    monkeypatch.setattr(main, "memory_service", Recorder("memory", log, memory_fail))
    monkeypatch.setattr(main, "agent", Recorder("agent", log, agent_fail))

    async def scenario():
        async with main.lifespan(main.app):
            log.append("serving")

    asyncio.run(scenario())
    return log


class TestLifespan:
    def test_initialises_services_before_serving_and_closes_after(self, monkeypatch):
        log = run_lifespan(monkeypatch)

        assert log == [
            "cache.initialize",
            "stream.initialize",
            "memory.initialize",
            "agent.create_graph",
            "serving",
            "stream.shutdown",  # first: in-flight runs record their "interrupted" event while the buffer is still open
            "agent.close",
            "cache.close",
        ]

    def test_cache_failure_does_not_stop_startup(self, monkeypatch):
        log = run_lifespan(monkeypatch, cache_fail={"initialize"})

        assert "serving" in log

    def test_memory_warmup_failure_does_not_stop_startup(self, monkeypatch):
        log = run_lifespan(monkeypatch, memory_fail={"initialize"})

        assert "serving" in log and "agent.create_graph" in log

    def test_cache_is_closed_even_if_agent_close_fails(self, monkeypatch):
        log = []
        monkeypatch.setattr(main, "cache_service", Recorder("cache", log))
        monkeypatch.setattr(main, "run_stream_service", Recorder("stream", log))
        monkeypatch.setattr(main, "memory_service", Recorder("memory", log))
        monkeypatch.setattr(main, "agent", Recorder("agent", log, {"close"}))

        async def scenario():
            async with main.lifespan(main.app):
                pass

        with pytest.raises(RuntimeError, match="agent close failed"):
            asyncio.run(scenario())

        assert "cache.close" in log

    def test_agent_and_cache_are_closed_even_if_stopping_the_runs_fails(self, monkeypatch):
        log = []
        monkeypatch.setattr(main, "cache_service", Recorder("cache", log))
        monkeypatch.setattr(main, "run_stream_service", Recorder("stream", log, {"shutdown"}))
        monkeypatch.setattr(main, "memory_service", Recorder("memory", log))
        monkeypatch.setattr(main, "agent", Recorder("agent", log))

        async def scenario():
            async with main.lifespan(main.app):
                pass

        with pytest.raises(RuntimeError, match="stream shutdown failed"):
            asyncio.run(scenario())

        assert "agent.close" in log and "cache.close" in log

    def test_graph_creation_failure_is_not_swallowed(self, monkeypatch):
        with pytest.raises(RuntimeError, match="create_graph failed"):
            run_lifespan(monkeypatch, agent_fail={"create_graph"})


@pytest.fixture
def client():
    return TestClient(main.app)  # not used as a context manager, so lifespan (DB, Redis) is skipped


class TestRoutes:
    def test_root(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert response.json()["name"] == main.settings.PROJECT_NAME

    @pytest.mark.parametrize("healthy, status, label", [(True, 200, "healthy"), (False, 503, "degraded")])
    def test_health_reflects_database_state(self, client, monkeypatch, healthy, status, label):
        async def db_health():
            return healthy

        monkeypatch.setattr(main.database_service, "health_check", db_health)

        response = client.get("/health")

        assert response.status_code == status
        assert response.json()["status"] == label
        assert response.json()["components"]["database"] == ("healthy" if healthy else "unhealthy")

    def test_validation_errors_are_flattened_to_a_422(self, client):
        response = client.post(f"{main.settings.API_V1_STR}/auth/register", json={"email": "not-an-email"})

        body = response.json()
        assert response.status_code == 422
        assert body["detail"] == "Validation error"
        assert {error["field"] for error in body["errors"]} >= {"email", "password"}
        assert all(set(error) == {"field", "message"} for error in body["errors"])

    def test_v1_routes_are_mounted_under_the_api_prefix(self, client):
        assert client.get(f"{main.settings.API_V1_STR}/health").status_code == 200

    def test_metrics_endpoint_is_exposed(self, client):
        assert client.get("/metrics").status_code == 200

    def test_cors_does_not_combine_wildcard_origin_with_credentials(self):
        cors = next(m for m in main.app.user_middleware if m.cls.__name__ == "CORSMiddleware")

        if "*" in main.settings.ALLOWED_ORIGINS:
            assert cors.kwargs["allow_credentials"] is False

    def test_cors_exposes_the_run_id_header_browsers_need_to_re_attach(self):
        cors = next(m for m in main.app.user_middleware if m.cls.__name__ == "CORSMiddleware")

        assert "X-Run-Id" in cors.kwargs["expose_headers"]

    def test_stream_resume_endpoints_are_mounted_and_require_auth(self, client):
        prefix = f"{main.settings.API_V1_STR}/chatbot/chat/stream"

        for path in (f"{prefix}/active", f"{prefix}/{'a' * 32}"):
            assert client.get(path).status_code in (401, 403)  # found, but rejected for lack of credentials
