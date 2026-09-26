"""``DatabaseService`` must read the pool settings ``Settings`` actually defines."""

import pytest

from app.core.config import settings
from app.services import database


def test_engine_is_built_from_the_postgres_pool_settings(monkeypatch):
    # regression: the service read settings.POOL_SIZE / POOL_MAX_OVERFLOW, which don't exist
    captured = {}
    monkeypatch.setattr(database, "create_engine", lambda url, **kwargs: captured.update(url=url, **kwargs) or object())
    monkeypatch.setattr(settings, "POSTGRES_POOL_SIZE", 7)
    monkeypatch.setattr(settings, "POSTGRES_MAX_OVERFLOW", 3)

    database.DatabaseService()

    assert captured["pool_size"] == 7
    assert captured["max_overflow"] == 3
    assert captured["pool_pre_ping"] is True
    assert captured["url"].startswith("postgresql://")


@pytest.mark.parametrize("name", ["POSTGRES_POOL_SIZE", "POSTGRES_MAX_OVERFLOW"])
def test_pool_settings_exist(name):
    assert isinstance(getattr(settings, name), int)
