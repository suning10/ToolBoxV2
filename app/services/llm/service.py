"""LLM service with retries, circular fallback, and optional structured output."""

import asyncio
import logging
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from openai import (
    APIError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import logger
from app.services.llm.registry import LLMRegistry

T = TypeVar("T", bound=BaseModel)

class LLMService:
    """Service for managing LLM calls with retries and circular fallback.

    Two distinct execution paths:

    - **Default path** (no model_name / response_format / model_kwargs): uses
      ``self._llm`` which is the tool-bound agent model. Circular fallback
      updates ``self._llm`` so tool bindings are preserved across retries.

    - **One-off path** (any override provided): resolves a fresh, local
      ``Runnable`` for the call without ever touching ``self._llm``, so
      concurrent default-path calls are never affected.
    """
    def __init__(self):
        self._llm: Any = None
        self._current_model_index: int = 0
        self._bound_tools: List = []

        all_names = LLMRegistry.get_all_names()
        try:
            self._current_model_index = all_names.index(settings.DEFAULT_LLM_MODEL)
            self._llm = LLMRegistry.get(settings.DEFAULT_LLM_MODEL)
            logger.info(
                "llm_service_initialized",
                default_model=settings.DEFAULT_LLM_MODEL,
                model_index=self._current_model_index,
                total_models=len(all_names),
                environment=settings.ENVIRONMENT.value,
            )
        except Exception as e:
            self._current_model_index = 0
            self._llm = LLMRegistry.LLMS[0]["llm"]
            logger.warning(
                "default_model_not_found_using_first",
                requested=settings.DEFAULT_LLM_MODEL,
                using=all_names[0] if all_names else "none",
                error=str(e),
            )

    @overload
    async def call(
        self,
        messages: LanguageModelInput,
        model_name: Optional[str] = ...,
        response_format: None = ...,
        **model_kwargs: Any,
    ) -> BaseMessage: ...

    @overload
    async def call(
        self,
        messages: LanguageModelInput,
        model_name: Optional[str] = ...,
        *,
        response_format: Type[T],
        **model_kwargs: Any,
    ) -> T: ...
    # public API
    async def call(
            self,
            messages: LanguageModelInput,
            model_name: Optional[str] = None,
            response_format: Optional[Type[BaseModel]] = None,
            **model_kwargs: Any,
    ) -> Union[BaseMessage, T]:
        """Call the LLM with retries and circular fallback.

                Args:
                    messages: Conversation messages to send.
                    model_name: Override the model. ``None`` uses the current default.
                    response_format: Pydantic schema for structured output. When
                        provided the call chains ``.with_structured_output(schema)``
                        and returns a validated instance of that schema instead of a
                        raw ``BaseMessage``.
                    **model_kwargs: Extra kwargs forwarded to ``LLMRegistry.get`` when
                        constructing a one-off model instance (e.g. ``temperature``,
                        ``max_tokens``, ``reasoning``).

                Returns:
                    ``BaseMessage`` when ``response_format`` is ``None``, otherwise a
                    validated instance of ``response_format``.

                Raises:
                    RuntimeError: When all models fail after retries or the total
                        timeout budget is exceeded.
                """
