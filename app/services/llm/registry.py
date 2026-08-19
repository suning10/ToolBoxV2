"""LLM model registry with pre-initialized instances."""
from typing import Dict, Any, List

from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import SecretStr
from app.core.config import settings, Environment
import langchain_openai

from app.core.logging import logger

_TOKEN_LIMIT: Dict[str, Any] = {"max_completion_tokens": settings.MAX_TOKENS}
# use secretStr to avoid store sensitive information in logging
_API_KEY = SecretStr(settings.OPENAI_API_KEY)

class LLMRegistry:

    """Registry of available LLM models with pre-initialized instances.

    This class maintains a list of LLM configurations and provides
    methods to retrieve them by name with optional argument overrides.
    """

    LLMS: List[Dict[str, Any]] = [
        {
            "name": "gpt-5-mini",
            "llm": langchain_openai.ChatOpenAI(
                model = "gpt-5-mini",
                api_key = _API_KEY,
                model_kwargs=_TOKEN_LIMIT,
                reasoning={"effort": "low"}
            ),
        },
        {
            "name": "gpt-5.4",
            "llm": langchain_openai.ChatOpenAI(
                model="gpt-5",
                api_key=_API_KEY,
                model_kwargs=_TOKEN_LIMIT,
                reasoning={"effort": "medium"},
            ),
        },
        {
            "name": "gpt-5.4-nano",
            "llm": langchain_openai.ChatOpenAI(
                model="gpt-5.4-nano",
                api_key=_API_KEY,
                model_kwargs=_TOKEN_LIMIT,
                reasoning={"effort": "low"},
            ),
        },
        {
            "name": "gpt-5",
            "llm": langchain_openai.ChatOpenAI(
                model="gpt-5",
                api_key=_API_KEY,
                model_kwargs=_TOKEN_LIMIT,
                top_p=0.95 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
                presence_penalty=0.1 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.0,
                frequency_penalty=0.1 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.0,
            ),
        },
    ]

    # factory method to get LLM
    # override **kwargs as needed
    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseChatOpenAI:
        """Get an LLM by name with optional argument overrides.

        When kwargs are provided a fresh ChatOpenAI instance is returned with
        those overrides applied, leaving the shared registry entry untouched.

        Args:
            model_name: Name of the model to retrieve.
            **kwargs: Optional arguments to override default model configuration.

        Returns:
            BaseChatModel instance.

        Raises:
            ValueError: If model_name is not found in LLMS.
        """
        model = next((m for m in cls.LLMS if m["name"] == model_name), None)

        if not model:
            available_models = ",".join([m["name"] for m in cls.LLMS ])
            raise ValueError(f"model name {model_name} not found in {available_models}")

        if kwargs:
            logger.debug("creating_llm_with_custom_args", model_name=model_name, custom_arags = list(kwargs.keys()))

        logger.debug("using_default_llm_instance", model_name=model_name)
        return model["llm"]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Return all registered model names in order.

        Returns:
            List of model name strings.
        """
        return [e["name"] for e in cls.LLMS]

    @classmethod
    def get_model_at_index(cls, index: int) -> Dict[str, Any]:
        """Return the model entry at a specific index, wrapping to 0 if out of range.

        Args:
            index: Index into LLMS.

        Returns:
            Model entry dict.
        """
        if 0 <= index < len(cls.LLMS):
            return cls.LLMS[index]
        return cls.LLMS[0]


