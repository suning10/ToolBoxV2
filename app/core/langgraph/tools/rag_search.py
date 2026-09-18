"""Knowledge-base search tool for LangGraph.

Searches the group-scoped RAG knowledge base. Access control is resolved
from the authenticated session's ``user_id``, injected automatically via
``RunnableConfig`` — this parameter is excluded from the tool's schema shown
to the LLM, so the model can never supply or override whose access applies.
"""

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.core.logging import logger
from app.services.rag import rag_service


@tool
async def rag_search(query: str, config: RunnableConfig) -> str:
    """Search the user's group-accessible knowledge base for relevant document excerpts.

    Use this for questions about internal documents, policies, or other
    knowledge-base content — not for general web knowledge (use web search
    for that) or facts about the user themselves (use long-term memory for
    that). Results are automatically scoped to documents the current user
    can access through their group memberships.

    Args:
        query: The search query describing what to look for.
        config: Injected automatically by the runtime — never populated by
            the LLM. Supplies the authenticated ``user_id`` that scopes access.

    Returns:
        str: Formatted excerpts from accessible documents, or a message
            saying nothing relevant was found.
    """
    user_id = config.get("metadata", {}).get("user_id")
    if user_id is None:
        logger.warning("rag_search_unauthenticated")
        return "No knowledge base access: this session is not authenticated."

    results = await rag_service.search(int(user_id), query)
    logger.info("rag_search_completed", user_id=user_id, result_count=len(results))
    return await rag_service.format_results(results)