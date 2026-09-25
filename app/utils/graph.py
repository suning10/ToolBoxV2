import hashlib
import json
from typing import Optional
import difflib

import tiktoken
from langchain_core.messages import BaseMessage
from langchain_core.messages import trim_messages as _trim_messages

from app.core.config import settings
from app.core.logging import logger
from app.schemas import Message
from app.schemas.graph import ToolCallRecord

# Cache tiktoken encoding at module level — thread-safe and reusable
try:
    _TIKTOKEN_ENCODING = tiktoken.encoding_for_model(settings.DEFAULT_LLM_MODEL)
except KeyError:
    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens_tiktoken(messages: list) -> int:
    """Count tokens locally using tiktoken — no API call needed."""
    num_tokens = 0
    for message in messages:
        # Every message has overhead tokens for role/name
        num_tokens += 4
        if isinstance(message, dict):
            for _, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(_TIKTOKEN_ENCODING.encode(value))
        elif isinstance(message, BaseMessage):
            content = message.content
            if isinstance(content, str):
                num_tokens += len(_TIKTOKEN_ENCODING.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        num_tokens += len(_TIKTOKEN_ENCODING.encode(block))
                    elif isinstance(block, dict) and "text" in block:
                        num_tokens += len(_TIKTOKEN_ENCODING.encode(block["text"]))
    num_tokens += 2  # every reply is primed with assistant
    return num_tokens

def dump_messages(messages: list[Message]) -> list[dict]:
    """Dump the messages to a list of dictionaries.

    Args:
        messages (list[Message]): The messages to dump.

    Returns:
        list[dict]: The dumped messages.
        messages: {role:[user, assistant,...],content: str}
    """
    return [message.model_dump() for message in messages]

def extract_text_content(content: str | list) -> str:
    """Extract plain text from an LLM content value.

    Handles both the simple string format and the structured block list returned
    by GPT-5 / Responses API models:
        [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}]

    Args:
        content: Raw content from a LangChain BaseMessage.

    Returns:
        Plain text string (empty string when nothing extractable is present).
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "reasoning":
                logger.debug(
                    "reasoning_block_received",
                    reasoning_id=block.get("id"),
                    has_summary=bool(block.get("summary")),
                )
    return "".join(parts)

def process_llm_response(response: BaseMessage) -> BaseMessage:
    """Normalise a raw LLM response so that ``response.content`` is always a plain string, regardless of the provider's content format.

    Args:
        response: The raw response from the LLM.

    Returns:
        The same BaseMessage instance with ``content`` set to a plain string.
    """
    if isinstance(response.content, list):
        response.content = extract_text_content(response.content)
        logger.debug(
            "processed_structured_content",
            content_block_count=len(response.content),
            extracted_length=len(response.content),
        )
    return response

def prepare_messages(messages: list[Message], system_prompt: str) -> list[Message]:
    """Prepare the messages for the LLM.

    Args:
        messages (list[Message]): The messages to prepare.
        system_prompt (str): The system prompt to use.

    Returns:
        list[Message]: The prepared messages.
    """

    try:
        trimmed_messages = _trim_messages(
            dump_messages(messages),
            strategy="last",
            token_counter=_count_tokens_tiktoken,
            max_tokens=settings.MAX_TOKENS,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
    except ValueError as e:
        # Handle unrecognized content blocks (e.g., reasoning blocks from GPT-5)
        if "Unrecognized content block type" in str(e):
            logger.warning(
                "token_counting_failed_skipping_trim",
                error=str(e),
                message_count=len(messages),
            )
            # Skip trimming and return all messages
            trimmed_messages = messages
        else:
            raise

    return [Message(role="system", content=system_prompt)] + trimmed_messages

def compute_call_signature(name: str, args: dict) -> str:
    """Stable short hash of a canonicalized (tool name, args) pair.

    Args:
        name: The tool name.
        args: The tool call arguments.

    Returns:
        str: A short hex digest identifying this exact call.
    """
    signature = json.dumps({"name": name, "args": args}, sort_keys=True, default=str)
    return hashlib.sha256(signature.encode()).hexdigest()[:16]



def find_duplicate_call(
    name: str, args: dict, history: list[ToolCallRecord], similarity_threshold: float
) -> Optional[ToolCallRecord]:
    """Find a prior call that exactly or near-exactly repeats this one.

    Only compares against history entries for the same tool name — args are
    compared as a canonicalized string via similarity ratio, so this is most
    meaningful for tools with a single string-ish field (e.g. a search query).

    Args:
        name: The tool name being called.
        args: The tool call arguments.
        history: Previously executed calls this turn/worker-run.
        similarity_threshold: Minimum ``difflib`` ratio (0-1) to count as a near-duplicate.

    Returns:
        Optional[ToolCallRecord]: The matching prior call, or ``None``.
    """
    signature = compute_call_signature(name, args)
    args_str = json.dumps(args, sort_keys=True, default=str)
    for entry in history:
        # called a different tool
        if entry.name != name:
            continue
        # name is same and signature is the same
        if entry.signature == signature:
            return entry
        # calculate similarity ratio for args
        # difflib only compare longest similarity of two string -> no semantic comparing
        ratio = difflib.SequenceMatcher(None, args_str, json.dumps(entry.args, sort_keys=True, default=str)).ratio()
        if ratio > similarity_threshold:
            return entry
    return None

def detect_cycle(history: list[ToolCallRecord], max_period: int = 3, min_repeats: int = 2) -> Optional[int]:
    """Detect a repeating call pattern at the tail of the history (e.g. A,B,A,B is period 2).

    Args:
        history: Previously executed calls this turn/worker-run, in order.
        max_period: Largest pattern length to check for.
        min_repeats: How many full repetitions of a pattern are required to flag a cycle.

    Returns:
        Optional[int]: The detected period, or ``None`` if no cycle is found.
    """

    signatures = [entry.signature for entry in history]
    n = len(signatures)
    # use sliding window to check last max_period
    for period in range (1, max_period + 1):
        window = period * min_repeats
        if n < window:
            continue
        tail = signatures[-window:]
        pattern = tail[:period]
        # period (A,B,C) period 3 A,B,C
        # repeats: determine the window size repeat = 2, A,B,A,B
        # check if all period == pattern in the window, AB == AB == AB
        if all(tail[i * period: (i + 1) * period] == pattern for i in range(min_repeats)):
            return period
    return None


def pending_interrupt_value(snapshot) -> Optional[object]:
    """Return the value of the first unanswered ``interrupt()`` on a thread, if any.

    ``snapshot.next`` is non-empty both when the graph is paused on an
    ``interrupt()`` (e.g. ``ask_human``) *and* when a run died mid-flight
    (client disconnect, crash), so it cannot be used to decide whether a
    ``Command(resume=...)`` has anything to consume. Only an interrupt attached
    to a pending task can.

    Args:
        snapshot: A LangGraph ``StateSnapshot`` from ``graph.aget_state``.

    Returns:
        The interrupt's payload, or ``None`` when no interrupt is pending.
    """
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def is_orphaned_run_for(snapshot, messages: list[Message]) -> bool:
    """Whether the thread holds an unfinished run for the turn ``messages`` is retrying.

    True when the checkpoint still has nodes to run, no interrupt is waiting on
    the user, and the last user message being sent is the one that started the
    unfinished run — i.e. the client is reconnecting after a dropped stream and
    the run should continue from its checkpoint rather than start over.

    Args:
        snapshot: A LangGraph ``StateSnapshot`` from ``graph.aget_state``.
        messages: The messages in the incoming request.

    Returns:
        bool: True if the run should be continued with ``None`` as graph input.
    """
    if not snapshot.next or pending_interrupt_value(snapshot) is not None:
        return False
    if not messages or messages[-1].role != "user":
        return False

    stored = (snapshot.values or {}).get("messages", [])
    last_human = next((m for m in reversed(stored) if m.type == "human"), None)
    if last_human is None:
        return False
    return extract_text_content(last_human.content) == messages[-1].content
