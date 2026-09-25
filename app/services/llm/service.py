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

      ** Strunture
      call -> _call_with_fallback
                (determine deafult_target(current), override_target(switch to idx), default_advance(switch to next model,
                override_target(idx + 1)
                this handles default + customized fallback logic
            -> _fallback_loop -> _call_with_retry (real logic to invoke llm)
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
        try:
            return await asyncio.wait_for(
                self._call_with_fallback(messages, model_name, response_format, **model_kwargs),
                timeout=settings.LLM_TOTAL_TIMEOUT,
        )
        except asyncio.TimeoutError:
            logger.exception("LLM Timeout", timeout =settings.LLM_TOTAL_TIMEOUT)
            raise RuntimeError(f"LLM Timeout: {settings.LLM_TOTAL_TIMEOUT}")

    def get_llm(self):
        return self._llm

    def bind_tools(self, tools: List) -> "LLMService":

        if self._llm:
            self._bound_tools = tools
            self._llm = self._llm.bind_tools(tools)
            logger.debug("tools_bound_to_llm", tool_count=len(tools))
        return self

    async def _call_with_fallback(
            self,
            messages: LanguageModelInput,
            model_name: Optional[str],
            response_format: Optional[Type[BaseModel]],
            model_kwargs: dict,
    ) -> Union[BaseMessage, BaseModel]:
        """Build path-specific strategies and delegate to the shared fallback loop.

        One-off path (any override set):
            ``get_target`` builds a fresh registry instance each attempt.
            ``advance`` increments a local index — ``self._llm`` is never touched.

        Default path (no overrides):
            ``get_target`` returns ``self._llm`` (tool-bound).
            ``advance`` calls ``_switch_to_next_model`` so bindings persist.
        """

        """
        strategy pattern to add flexibility 
        """
        def _override_target(idx: int) -> Any:
            """
            Replace target by LLM of idx
            :param idx:
            :return: LLM
            """
            base = LLMRegistry.get(LLMRegistry.LLMS[idx]["name"])
            return base.with_structured_output(response_format) if response_format else base

        def _default_target(_: int) -> Any:
            return self._llm

        def _default_advance(_: int) -> Optional[int]:
            """
            default + 1, use _switch_to_next_model
            :param _:
            :return: next LLM Model
            """
            return self._current_model_index if self._switch_to_next_model() else None


        # logic of fallback to another model
        if model_name or response_format or model_kwargs: # case 1, a model name is given
            all_llm_names = LLMRegistry.get_all_names()
            if model_name and model_name not in all_llm_names: # edge case: model name is not in registry
                raise ValueError(
                    f"model '{model_name}' not found in registry. available models: {', '.join(all_llm_names)}"
                )

            start =  all_llm_names.index(model_name) if model_name else self._current_model_index
            total = len(all_llm_names)
            get_target: Callable[[int], Any] =  _override_target # parameter of int and return any

            def _override_advance(idx: int) -> Optional[int]:
                return (idx + 1) % total
            advance: Callable[[int], Optional[int]] = _override_advance # parameter of int, returns int or nothing
        else:
            start = self._current_model_index
            get_target = _default_target
            advance = _default_advance

        return await self._fallback_loop(messages, start, get_target, advance)



    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _invoke_with_retry(self, llm: Any, messages: LanguageModelInput) -> Any:
        """Invoke an LLM runnable with automatic per-model retry logic.

        Args:
            llm: Any LangChain ``Runnable`` (plain model or structured-output chain).
            messages: Messages to send.

        Returns:
            The runnable's response (``BaseMessage`` or a ``BaseModel`` instance).

        Raises:
            OpenAIError: Propagated after all retry attempts are exhausted.
        """
        try:
            response = await llm.ainvoke(messages)
            logger.debug("llm_call_successful")
            return response
        except (RateLimitError, APITimeoutError, APIError) as e:
            logger.warning(
                "llm_call_failed_retrying",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            raise
        except OpenAIError as e:
            logger.error(
                "llm_call_failed",
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

    def _switch_to_next_model(self) -> bool:
        """Advance the default model to the next entry in the registry (circular).

        Mutates ``self._llm`` and ``self._current_model_index`` so tool bindings
        survive model switches on the default agent path.

        Returns:
            ``True`` on success, ``False`` if the switch failed.
        """
        try:
            next_index = (self._current_model_index + 1) % len(LLMRegistry.LLMS)
            next_llm = LLMRegistry.get_model_at_index(next_index)
            logger.warning(
                "switching_to_next_model",
                from_index=self._current_model_index,
                to_index=next_index,
                to_model=next_llm["name"],
            )
            self._current_model_index = next_index
            self._llm = next_llm["llm"]
            # bind the tools to new switched model
            if self._bound_tools:
                self._llm = self._llm.bind_tools(self._bound_tools)
            logger.info("model_switched", new_model=next_llm["name"], new_index=next_index)
            return True
        except Exception as e:
            logger.error("model switch failed", error = str(e))
            return False

    async def _fallback_loop(self,
                             messages,
                             start,
                             get_target,
                             advance) -> Any:

        """Shared fallback loop — try each model in turn until one succeeds.

        Args:
            messages: Messages to send.
            start: Registry index to begin from.
            get_target: Returns the ``Runnable`` to invoke for a given index.
            advance: Returns the next index to try, or ``None`` to stop.

        Returns:
            The first successful response.

        Raises:
            RuntimeError: When all models have been exhausted.
        """
        total = len(LLMRegistry.LLMS)
        current = start
        models_tried = 0
        last_error: Optional[Exception] = None

        for model_tried in range(0, total):
            current_name = LLMRegistry.LLMS[current]["name"]
            try:
                return await self._invoke_with_retry(get_target(current), messages)
            except OpenAIError as e:
                last_error = e
                logger.error(
                    "llm_call_failed_after_retries",
                    model=current_name,
                    models_tried=models_tried,
                    total_models=total,
                    error=str(e),
                )
                if models_tried >= total:
                    logger.error(
                        "all_models_failed", models_tried=models_tried, starting_model=LLMRegistry.LLMS[start]["name"]
                    )
                    break
                next_idx = advance(current) # dynamically decide what model to be used
                if next_idx is None:
                    logger.error("failed_to_switch_to_next_model")
                    break
                current = next_idx

            raise RuntimeError(
                f"failed to get response from llm after trying {models_tried} models. last error: {str(last_error)}"
            )

llm_service = LLMService()