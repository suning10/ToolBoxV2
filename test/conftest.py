"""Shared pytest setup.

The LLM registry builds its ``ChatOpenAI`` clients at import time, and those refuse to
construct without a key. Tests never call the API (they patch the LLM), so a dummy is fine;
``setdefault`` keeps a real key if one is exported.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
# Keep Langfuse quiet and offline during tests.
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
