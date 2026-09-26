"""Tests for the auth dependencies in ``app/api/v1/auth.py``."""

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.v1 import auth


def creds(token):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


@pytest.mark.parametrize("dependency", [auth.get_current_user, auth.get_current_session])
@pytest.mark.parametrize("token", ["notajwt", "only.two", "has space.in.it"])
def test_malformed_token_is_a_422_not_a_server_error(dependency, token):
    # regression: the handler logged ``str(ve)`` / ``str(e)`` with the wrong name -> NameError -> 500
    with pytest.raises(HTTPException) as caught:
        asyncio.run(dependency(creds(token)))

    assert caught.value.status_code == 422
    assert caught.value.detail == "Invalid token format"


@pytest.mark.parametrize("dependency", [auth.get_current_user, auth.get_current_session])
def test_well_formed_but_invalid_token_is_a_401(dependency):
    with pytest.raises(HTTPException) as caught:
        asyncio.run(dependency(creds("aaaa.bbbb.cccc")))

    assert caught.value.status_code == 401
