"""Rate limiting configuration for the application.

This module configures rate limiting using slowapi, with default limits
defined in the application settings. Rate limits are applied based on
remote IP addresses.

When Valkey is configured, uses it as a distributed storage backend
so rate limits work correctly across multiple app instances.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.logging import logger

# Build storage URI for Valkey if configured
_storage_uri = None
if settings.VALKEY_HOST:
    _password_part = f":{settings.VALKEY_PASSWORD}@" if settings.VALKEY_PASSWORD else ""
    _storage_uri = f"redis://{_password_part}{settings.VALKEY_HOST}:{settings.VALKEY_PORT}/{settings.VALKEY_DB}"
    logger.info("rate_limiter_using_valkey", host=settings.VALKEY_HOST, port=settings.VALKEY_PORT)

# Initialize rate limiter (uses in-memory storage if no Valkey)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=settings.RATE_LIMIT_DEFAULT,  # pyright: ignore[reportArgumentType]
    storage_uri=_storage_uri,
)
