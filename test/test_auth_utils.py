"""Tests for the JWT helpers in ``app/utils/auth.py``."""

import base64
import json
from datetime import (
    UTC,
    datetime,
    timedelta,
)

import pytest
from jose import jwt

from app.core.config import settings
from app.schemas.auth import Token
from app.utils.auth import (
    create_access_token,
    verify_token,
)

SECRET = "unit-test-secret-not-for-production"


@pytest.fixture(autouse=True)
def jwt_settings(monkeypatch):
    # The dev .env leaves JWT_SECRET_KEY empty, so pin our own for deterministic tests.
    monkeypatch.setattr(settings, "JWT_SECRET_KEY", SECRET)
    monkeypatch.setattr(settings, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(settings, "JWT_ACCESS_TOKEN_EXPIRE_DAYS", 7)


def decode(token: str) -> dict:
    return jwt.decode(token, SECRET, algorithms=["HS256"])


def b64(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


class TestCreateAccessToken:
    def test_returns_a_token_model(self):
        token = create_access_token("thread-1")

        assert isinstance(token, Token)
        assert token.access_token.count(".") == 2

    def test_subject_is_the_thread_id(self):
        assert decode(create_access_token("thread-1").access_token)["sub"] == "thread-1"

    def test_default_expiry_uses_configured_number_of_days(self):
        before = datetime.now(UTC)

        token = create_access_token("thread-1")

        after = datetime.now(UTC)
        assert before + timedelta(days=7) <= token.expires_at <= after + timedelta(days=7)

    def test_custom_expiry_overrides_default(self):
        before = datetime.now(UTC)

        token = create_access_token("thread-1", expires_delta=timedelta(minutes=5))

        assert before + timedelta(minutes=5) <= token.expires_at <= before + timedelta(minutes=5, seconds=5)

    def test_exp_claim_matches_reported_expiry(self):
        token = create_access_token("thread-1", expires_delta=timedelta(hours=1))

        assert decode(token.access_token)["exp"] == int(token.expires_at.timestamp())

    def test_contains_issued_at_and_token_id_claims(self):
        claims = decode(create_access_token("thread-1").access_token)

        assert claims["iat"] <= datetime.now(UTC).timestamp()
        assert claims["jti"].startswith("thread-1-")

    def test_token_is_signed_with_the_configured_secret(self):
        token = create_access_token("thread-1").access_token

        with pytest.raises(jwt.JWTError):
            jwt.decode(token, "some-other-secret", algorithms=["HS256"])


class TestVerifyToken:
    def test_round_trip_returns_the_thread_id(self):
        token = create_access_token("thread-42").access_token

        assert verify_token(token) == "thread-42"

    def test_expired_token_returns_none(self):
        token = create_access_token("thread-1", expires_delta=timedelta(seconds=-30)).access_token

        assert verify_token(token) is None

    def test_token_signed_with_another_secret_returns_none(self):
        forged = jwt.encode({"sub": "thread-1", "exp": datetime.now(UTC) + timedelta(days=1)}, "attacker", "HS256")

        assert verify_token(forged) is None

    def test_token_using_a_different_algorithm_returns_none(self):
        token = jwt.encode({"sub": "thread-1", "exp": datetime.now(UTC) + timedelta(days=1)}, SECRET, "HS512")

        assert verify_token(token) is None

    def test_tampered_payload_returns_none(self):
        header, _, signature = create_access_token("thread-1").access_token.split(".")
        forged_payload = b64({"sub": "admin", "exp": int((datetime.now(UTC) + timedelta(days=1)).timestamp())})

        assert verify_token(f"{header}.{forged_payload}.{signature}") is None

    def test_unsigned_alg_none_token_is_rejected(self):
        header = b64({"alg": "none", "typ": "JWT"})
        payload = b64({"sub": "admin", "exp": int((datetime.now(UTC) + timedelta(days=1)).timestamp())})

        assert verify_token(f"{header}.{payload}.bogus") is None

    def test_valid_signature_but_no_subject_returns_none(self):
        token = jwt.encode({"exp": datetime.now(UTC) + timedelta(days=1)}, SECRET, "HS256")

        assert verify_token(token) is None

    @pytest.mark.parametrize("token", ["", None, 123, b"a.b.c", ["a.b.c"]])
    def test_non_string_or_empty_input_raises(self, token):
        with pytest.raises(ValueError, match="non-empty string"):
            verify_token(token)

    @pytest.mark.parametrize(
        "token",
        [
            "notajwt",
            "only.two",
            "one.two.three.four",
            "has space.in.it",
            "a..c",
            "a.b.",
            "a.b.c\n",
            "a.b.c;drop table",
            "ünïcode.b.c",
        ],
    )
    def test_malformed_tokens_raise_before_decoding(self, token):
        with pytest.raises(ValueError, match="format is invalid"):
            verify_token(token)

    def test_well_formed_but_garbage_token_returns_none(self):
        assert verify_token("aaaa.bbbb.cccc") is None
