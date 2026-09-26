"""Tests for the background title generation in ``app/services/session_naming.py``."""

import asyncio
from types import SimpleNamespace

import pytest

from app.core.metrics import session_names_generated_total
from app.services import session_naming


def counter(status):
    return session_names_generated_total.labels(status=status)._value.get()


class FakeDatabase:
    def __init__(self):
        self.updates = []

    async def update_session_name(self, session_id, name):
        self.updates.append((session_id, name))


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeDatabase()
    monkeypatch.setattr(session_naming, "database_service", db)
    return db


def test_generated_title_is_persisted_against_the_session(monkeypatch, fake_db):
    async def fake_call(*args, **kwargs):
        return SimpleNamespace(title="Trip to Paris")

    monkeypatch.setattr(session_naming, "llm_service", SimpleNamespace(call=fake_call))
    before = counter("success"), counter("error")

    asyncio.run(session_naming._persist_session_name("sess-1", "plan a trip to paris"))

    assert fake_db.updates == [("sess-1", "Trip to Paris")]
    # regression: the success log referenced a stray module-level ``result``, which raised and
    # sent every successful run down the error path
    assert (counter("success"), counter("error")) == (before[0] + 1, before[1])


def test_llm_failure_is_counted_and_never_raised(monkeypatch, fake_db):
    async def boom(*args, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(session_naming, "llm_service", SimpleNamespace(call=boom))
    before = counter("error")

    asyncio.run(session_naming._persist_session_name("sess-1", "hello"))

    assert fake_db.updates == []
    assert counter("error") == before + 1


def test_no_stray_imports_left_behind():
    source = session_naming.__loader__.get_source(session_naming.__name__)

    assert "envs." not in source
