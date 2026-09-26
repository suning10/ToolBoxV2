"""Tests for ``app/core/cache.py``."""

import asyncio

import pytest

from app.core import cache
from app.core.cache import (
    InMemoryCacheService,
    ValkeyCacheService,
    cache_key,
)


def run(coro):
    return asyncio.run(coro)


class TestInMemoryCacheService:
    def test_set_then_get_uses_the_default_ttl(self):
        # regression: ``set`` read a non-existent ``_default_ttl`` and raised AttributeError
        service = InMemoryCacheService(default_ttl=60)

        async def scenario():
            await service.set("k", "v")
            return await service.get("k")

        assert run(scenario()) == "v"

    def test_missing_key(self):
        assert run(InMemoryCacheService().get("nope")) is None

    def test_entry_expires_after_its_ttl(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(cache.time, "monotonic", lambda: now[0])
        service = InMemoryCacheService(default_ttl=60)

        async def scenario():
            await service.set("k", "v", ttl=10)
            before = await service.get("k")
            now[0] += 11
            return before, await service.get("k")

        assert run(scenario()) == ("v", None)

    def test_expired_entry_is_evicted(self, monkeypatch):
        now = [0.0]
        monkeypatch.setattr(cache.time, "monotonic", lambda: now[0])
        service = InMemoryCacheService(default_ttl=5)

        async def scenario():
            await service.set("k", "v")
            now[0] += 6
            await service.get("k")

        run(scenario())

        assert "k" not in service._cache

    def test_explicit_ttl_overrides_default(self, monkeypatch):
        now = [0.0]
        monkeypatch.setattr(cache.time, "monotonic", lambda: now[0])
        service = InMemoryCacheService(default_ttl=5)

        async def scenario():
            await service.set("k", "v", ttl=100)
            now[0] += 50
            return await service.get("k")

        assert run(scenario()) == "v"

    def test_delete_and_close(self):
        service = InMemoryCacheService()

        async def scenario():
            await service.set("a", "1")
            await service.set("b", "2")
            await service.delete("a")
            await service.delete("never-existed")  # must not raise
            after_delete = (await service.get("a"), await service.get("b"))
            await service.close()
            return after_delete, await service.get("b")

        assert run(scenario()) == ((None, "2"), None)


class TestValkeyCacheServiceWithoutAClient:
    """If Valkey is down at startup the client stays None; the app must degrade, not crash."""

    def test_every_operation_is_a_safe_no_op(self):
        service = ValkeyCacheService(default_ttl=60)

        async def scenario():
            await service.set("k", "v")
            await service.delete("k")
            value = await service.get("k")
            await service.close()
            return value

        assert run(scenario()) is None


class TestCacheKey:
    def test_is_deterministic_and_prefixed(self):
        assert cache_key("rag", "1", "hello") == cache_key("rag", "1", "hello")
        assert cache_key("rag", "1", "hello").startswith("rag:")

    def test_differs_by_parts_and_prefix(self):
        assert cache_key("rag", "1", "a") != cache_key("rag", "1", "b")
        assert cache_key("rag", "1", "a") != cache_key("memory", "1", "a")

    def test_does_not_leak_raw_parts_into_the_key(self):
        assert "secret-query" not in cache_key("rag", "1", "secret-query")


@pytest.mark.parametrize("service", [InMemoryCacheService(), ValkeyCacheService()])
def test_both_backends_share_the_same_interface(service):
    for method in ("initialize", "get", "set", "delete", "close"):
        assert asyncio.iscoroutinefunction(getattr(service, method))
