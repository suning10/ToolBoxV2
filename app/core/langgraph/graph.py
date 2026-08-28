import asyncio
from typing import (
    AsyncGenerator,
    Optional,
    cast,
)
from urllib.parse import quote_plus

import httpx
import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages, SystemMessage, HumanMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import (
    END,
    StateGraph, add_messages,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import (
    Command,
    CompiledStateGraph,
)
from langgraph.types import (
    RetryPolicy,
    StateSnapshot, Send,
)
from psycopg import (
    AsyncConnection,
    sql,
)
from psycopg.rows import (
    DictRow,
    dict_row,
)
from psycopg_pool import AsyncConnectionPool

from app.core.config import (
    Environment,
    settings,
)
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.metrics import llm_inference_duration_seconds
from app.core.observability import langfuse_callback_handler
from app.core.prompts import load_system_prompt, DECOMPOSITION_PROMPT
from app.schemas import (
    GraphState,
    Message,
)
from app.schemas.graph import ToolCallRecord, QueryPlan
from app.services.llm import llm_service
from app.services.memory import memory_service
from app.utils import (
    dump_messages,
    extract_text_content,
    prepare_messages,
    process_llm_response,
)
from app.utils.graph import find_duplicate_call, compute_call_signature, detect_cycle

PostgresConnPool = AsyncConnectionPool[AsyncConnection[DictRow]]

class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use the LLM service with tools bound
        self.llm_service = llm_service
        self.llm_service.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        # Workers run in parallel swarm branches and never get ask_human — pausing
        # one branch of a parallel swarm to ask the user is a known LangGraph
        # sharp edge (which branch would resume?), so it's out of scope for now.
        self._worker_tools: list[BaseTool] = [tool for tool in tools if tool.name != "ask_human"]
        self._connection_pool: Optional[PostgresConnPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        logger.info(
            "langgraph_agent_initialized",
            model=settings.DEFAULT_LLM_MODEL,
            environment=settings.ENVIRONMENT.value,
        )

    async def _get_connection_pool(self) -> Optional[PostgresConnPool]:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool or None when the pool fails to initialise in
            production (the app keeps running in a degraded mode).
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                connection_url = (
                    "postgresql://"
                    f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                )

                self._connection_pool = AsyncConnectionPool(
                    connection_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                        "row_factory": dict_row,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we might want to degrade gracefully
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                    return None
                raise e
        return self._connection_pool

    async def _chat(
            self,
            state: GraphState,
            config: RunnableConfig,
            allowed_tools: Optional[list[BaseTool]] = None,
            tool_call_limit: int = settings.MAX_TOOL_CALLS_PER_TURN,) -> Command:
        """Process the chat state and generate a response.

        Args:
            state (GraphState): The current state of the conversation.
            config (RunnableConfig): The runnable configuration for this invocation.

        Returns:
            Command: Command object with updated state and next node to execute.
        """
        # step 1: get model
        current_llm = self.llm_service.get_current_llm()
        model_name = (current_llm.model_name
                      if current_llm and hasattr(current_llm, "model_name")
                      else settings.DEFAULT_LLM_MODEL)
        # step 2: pull information from mem0 for user specific information
        username = config.get("metadata",{}).get("username")
        thread_id = config.get("configurable", {}).get("thread_id")
        SYSTEM_PROMPT = load_system_prompt(username = username, long_term_memory = state.long_term_memory)
        # step 3: prepare message
        messages = prepare_messages(state.messages, SYSTEM_PROMPT)
        # step 4: check reached turns allowed
        tool_limit_flag = state.tool_call_count >= settings.MAX_TOOL_LIMIT
        # step 5: fallback - ask LLM to response based on what currently have
        if tool_limit_flag:
            messages = messages + [
                Message(
                    role="user",
                    content=(
                        "Based on everything you've found so far, summarize your findings and give me "
                        "your best answer now. Do not call any more tools."
                    ),
                )
            ]
            logger.warning(
                "tool_call_limit_reached",
                session_id=thread_id,
                tool_call_count=state.tool_call_count,
                max_tool_calls=settings.MAX_TOOL_CALLS_PER_TURN,
            )
        # step 6: call LLM service
        try:
            # Use LLM service with automatic retries and circular fallback
            with llm_inference_duration_seconds.labels(model=model_name).time():
                if tool_limit_flag:
                    # model_name routes through the tool-less override path so the
                    # model has no tool schema and cannot emit further tool_calls
                    response_message = await self.llm_service.call(dump_messages(messages), model_name=model_name)
                elif allowed_tools is not None:
                    # avoid repetitive tool calls
                    response_message = await self.llm_service.call(dump_messages(messages), tools=allowed_tools)
                else:
                    response_message = await self.llm_service.call(dump_messages(messages))

            # Process response to handle structured content blocks
            response_message = process_llm_response(response_message)

            logger.info(
                "llm_response_generated",
                session_id=thread_id,
                model=model_name,
                environment=settings.ENVIRONMENT.value,
                tool_call_count=state.tool_call_count,
            )

            if not tool_limit_flag and isinstance(response_message, AIMessage) and response_message.tool_calls:
                goto = "tool_call"
            else:
                goto = END

            return Command(update={"messages": [response_message], "goto": goto}) #update: Update to apply to the graph's state.
        except Exception as e:
            logger.error(
                "llm_call_failed_all_models",
                session_id=thread_id,
                error=str(e),
                environment=settings.ENVIRONMENT.value,
            )
            raise Exception(f"failed to get llm response after trying all models: {str(e)}")

    async def _tool_call(self, state: GraphState) -> Command:
        """Process tool calls from the last message.

        Process tool calls from the last message.

        Detects exact-duplicate and near-duplicate (string-similarity on args)
        repeat calls against this turn's ``action_history``, reusing the
        cached result instead of re-invoking the tool. Also detects
        oscillating call patterns (e.g. A, B, A, B) across the accumulated
        history and appends a corrective note to this round's tool results
        when found.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Command: Command object with updated messages, action_history,
                and tool_call_count, routing back to chat.

        """
        tool_calls = state.messages[-1].tool_calls # # [{'name': 'get_weather', 'args': {'location': 'NYC'}, 'id': 'call_abc123'}]
        history = state.action_history
        async def _execute_tool(tool_call: dict) -> ToolMessage:
            name, args = tool_call["name"], tool_call["args"]
            # find duplicate
            duplicate = find_duplicate_call(name, args, history, settings.TOOL_CALL_SIMILARITY_THRESHOLD)

            if duplicate is not None:
                logger.warning("duplicate_tool_call_detected", tool_name=name, matched_signature=duplicate.signature)
                result = duplicate.result
                # inject prompt ask LLM to stop calling same tools
                content = (
                    "[Note: this call repeats one you already made — reusing the previous result "
                    f"instead of calling the tool again. Try a different approach.]\n\n{result}"
                )
            else:
                result = await self.tools_by_name[name].ainvoke(args)
                content = result

            record = ToolCallRecord(name=name, args=args, signature=compute_call_signature(name, args), result=result)
            return ToolMessage(content=content, name=name, tool_call_id=tool_call["id"]), record

        # execute all tools concurrently
        if len(tool_calls) == 1:
            output = await _execute_tool(tool_calls[0])

        else:
            # gather(tool1, tool2, tool3)
            # use * to unpack
            output = list(await asyncio.gather(*[_execute_tool(tool_call) for tool_call in tool_calls]))

        outputs = [result[0] for result in output]
        updated_history = history + [result[1] for result in output]
        # detect cycle
        cycle_period = detect_cycle(updated_history)
        if cycle_period is not None:
            logger.warning("tool_call_cycle_detected", period=cycle_period, history_length=len(updated_history))
            warning_prompt = (
                "\n\n[Note: you're repeating the same sequence of tool calls. "
                "Stop calling tools and answer with your best response now.]"
            )
            outputs = [
                ToolMessage(content=str(record.content) + warning_prompt, name= record.name, tool_call_id=record.tool_call_id)
                for record in outputs
            ]

        # apply the change to state update tool count in state, goto chat
        # GraphState: {messages, tool_call_count, longterm_memeory}
        return Command(update={"messages": output,
                               "tool_call_count": state.tool_call_count + 1,
                               "action_history": updated_history},
                       goto="chat")

    """
    Session Below is the agent swarm 
    ======================================================
    
    _plan: lead agent/classifier
    """
    async def _plan(self, state: GraphState, config: RunnableConfig) -> Command:
        """Classify the incoming query and route it to the single agent or a worker swarm.

        Args:
            state: The current state of the conversation.
            config: The runnable configuration for this invocation.

        Returns:
            Command: routes to "chat" for simple queries (the common case),
                or dynamically fans out to parallel "worker" branches — one
                per subtask — for genuinely complex, decomposable queries.
        """
        # 1. get current thread_id/session_id
        thread_id = config.get("configurable", {}).get("thread_id")
        # 2. extract user query
        user_query = extract_text_content(state.messages[-1].content) if state.messages else ""

        # 3. decompose a query by calling LLM - add extra latency, call use a smaller LLM Model
        try:
            plan: QueryPlan = await self.llm_service.call(
                [SystemMessage(content=DECOMPOSITION_PROMPT), HumanMessage(content=user_query)],
                model_name="gpt-5.4-nano",
                response_format=QueryPlan,
                reasoning={"effort": "low"},
            )
        except Exception as e:
            logger.exception("Exception during decomposition due to " + e, session_id = thread_id )
            return Command(goto="chat")

        # 4. publish subtasks if complex
        if plan.complexity == "simple" or not plan.subtasks:
            logger.info("query_routed_simple", session_id=thread_id)
            return Command(goto="chat")

        subtasks = plan.subtasks[: settings.MAX_SUBTASKS]
        logger.info("query_routed_complex", session_id=thread_id, subtask_count=len(subtasks))

        # use Send to fan out agents
        # each subagent is independent, do not share context, only send message + long term memeory
        return Command(
            update={"subtasks": subtasks},
            goto=[
                Send("worker",
                     {"messages": [{"role": "user", "content": subtask}], "long_term_memory": state.long_term_memory})
                for subtask in subtasks
            ])

    async def _worker(self, state: GraphState, config: RunnableConfig) -> Command:
        """Run one decomposed subtask to completion.

        Reuses ``_chat``/``_tool_call`` directly as a small local loop rather
        than the registered "chat"/"tool_call" graph nodes, so N of these can
        run concurrently as parallel ``Send`` branches without interfering
        with each other's state. Scoped to the worker toolset (no
        ``ask_human``) and a smaller per-worker tool-call budget.

        Args:
            state: This branch's scoped state — the subtask as its only
                message, via the ``Send`` payload from ``_plan``.
            config: The runnable configuration for this invocation.

        Returns:
            Command: appends this worker's answer to the shared
                ``subtask_results`` list (merged via its ``operator.add``
                reducer) and routes to "synthesize".
        """
        local_state = state
        while True:
            chat_command = await self._chat(
                local_state,
                config,
                allowed_tools= self._worker_tools,
                tool_call_limit=settings.MAX_SUBTASKS,
            )
            chat_update = cast(dict, chat_command.update)
            local_state.messages = cast(list, add_messages(local_state.messages, chat_update["messages"]))
            if chat_command.goto == END:
                break

            tool_command = await self._tool_call(local_state)
            tool_update = cast(dict, tool_command.update)
            local_state.messages = cast(list, add_messages(local_state.messages, tool_update["messages"]))
            local_state.tool_call_count = tool_update["tool_call_count"]
            local_state.action_history = tool_update["action_history"]

        result_text = extract_text_content(local_state.messages[-1].content) if local_state.messages else ""
        logger.info(
            "worker_completed",
            session_id=config.get("configurable", {}).get("thread_id"),
            tool_call_count=local_state.tool_call_count,
        )
        return Command(update={"subtask_results": [result_text]}, goto="synthesize")

    async def _synthesize(self, state: GraphState, config: RunnableConfig) -> Command:
        """Combine parallel worker findings into one final answer.

        Args:
            state: The shared state after all worker branches have
                converged, with ``subtask_results`` merged across branches.
            config: The runnable configuration for this invocation.

        Returns:
            Command: appends the synthesized answer and ends the turn.
        """

        username = config.get("metadata", {}).get("username")
        thread_id = config.get("configurable", {}).get("thread_id")
        original_query = extract_text_content(state.messages[-1].content) if state.messages else ""
        # consolidate sub agent result
        # result stored in graphstate
        findings = "\n\n".join(f"Finding {i + 1}: {result}" for i, result in enumerate(state.subtask_results))
        synthesis_request = (
            f"Original question: {original_query}\n\n"
            f"Sub-task findings:\n{findings}\n\n"
            "Synthesize these into one clear, direct answer to the original question."
        )
        # reconstruct the messages
        SYSTEM_PROMPT = load_system_prompt(username=username, long_term_memory=state.long_term_memory)
        prompt_messages = prepare_messages([Message(role="user", content=synthesis_request)], SYSTEM_PROMPT)
        # call LLM
        try:
            response_message = await self.llm_service.call(dump_messages(prompt_messages))
            response_message = process_llm_response(response_message)
        except Exception as e:
            logger.error("swarm_synthesis_failed", session_id=thread_id, error=str(e))
            raise Exception(f"failed to synthesize swarm results: {str(e)}")

        logger.info("swarm_synthesis_completed", session_id=thread_id, subtask_count=len(state.subtask_results))
        return Command(update={"messages": [response_message]}, goto=END)

    """
    ======================================================
    """


    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the LangGraph workflow.

        Returns:
            Optional[CompiledStateGraph]: The configured LangGraph instance or None if init fails
        """

        def should_retry(exc: Exception) -> bool:
            """Custom retry logic"""

            # Retry on rate limits
            if isinstance(exc, openai.RateLimitError):
                return True

            # Retry on 5xx server errors
            if isinstance(exc, httpx.HTTPStatusError):
                return exc.response.status_code >= 500

            # Retry on network issues
            if isinstance(exc, (ConnectionError, TimeoutError)):
                return True

            # Don't retry on 4xx client errors (bad input)
            if isinstance(exc, openai.BadRequestError):
                return False

            return False

        if self._graph is None:
            try:
                # create a new graph
                graph_builder = StateGraph(GraphState)
                # refer to agent swarm doc
                # Simple: plan -> chat -> tool call (if any) -> end
                # Complex: plan -> worker -> synthesize -> end
                # use leader agent to decompose query
                graph_builder.add_node("plan", self._plan,destinations=("chat", "worker"))
                graph_builder.add_node("chat", self._chat, destinations=("tool_call", END)) # destination apply to edgeless graph, Command
                graph_builder.add_node("tool_call", self._tool_call, destinations=("chat", ), retry_policy=RetryPolicy(max_attempts = 3, retry_on = should_retry))
                graph_builder.add_node("worker", self._worker, destinations=("synthesize",))
                graph_builder.add_node("synthesize", self._synthesize, destinations=(END,))
                graph_builder.set_entry_point("plan")
                graph_builder.set_finish_point("chat")

                # Get connection pool (may be None in production if DB unavailable)
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                else:
                    # In production, proceed without checkpointer if needed
                    checkpointer = None
                    if settings.ENVIRONMENT != Environment.PRODUCTION:
                        raise Exception("Connection pool initialization failed")

                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value}"
                )
                logger.info(
                    "graph_created",
                    graph_name=f"{settings.PROJECT_NAME} Agent",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we don't want to crash the app
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_graph")
                    return None
                raise e

        return self._graph

    async def _get_graph(self) -> CompiledStateGraph:
        """Return the compiled graph, creating it on first access.

        Raises:
            RuntimeError: When ``create_graph()`` swallowed an init failure
                (production-only path) and returned ``None``. Callers can
                rely on the return being non-``None``.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        if self._graph is None:
            raise RuntimeError("graph initialization failed")
        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> list[Message]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.
            username (Optional[str]): The display name of the user.

        Returns:
            list[Message]: The response from the LLM.
        """

        # step 1: get graph + RunnableConfig
        graph = await self._graph
        callbacks: list[BaseCallbackHandler] = [langfuse_callback_handler] if settings.LANGFUSE_TRACING_ENABLED else []
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
            "recursion_limit": settings.RECURSION_LIMIT,
        }

        try:
            # step 2: get long-term Memory(or cache) -- use await to wait for result to be returned
            state, relevant_memory = await asyncio.gather(
                graph.aget_state(config),
                memory_service.search(user_id, messages[-1].content),
            )
            # step 3: check if interrupt
            if state.next: # return empty if not interrupt
                logger.info("resuming_interrupted_graph", session_id=session_id, next_nodes=state.next)
                response = await graph.ainvoke(Command(resume=messages[-1].content), config=config)
            else:         # step 4: graph.invoke()
                relevant_memory = relevant_memory or "No relevant memory found."
                response = await graph.ainvoke(
                    input={
                        "messages": dump_messages(messages),
                        "long_term_memory": relevant_memory,
                        "tool_call_count": 0,
                        "subtasks": [],
                        "subtask_results": [],
                        "action_history": [],
                    },
                    config=config,
                )
            # step 5: check if graph was interrupted during this invocation

            state = await graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                logger.info("graph_interrupted", session_id=session_id, interrupt_value=str(interrupt_value))
                return [Message(role="assistant", content=str(interrupt_value))]

            openai_msgs = cast(list[dict], convert_to_openai_messages(response["messages"]))
            asyncio.create_task(memory_service.add(user_id, openai_msgs, config.get("metadata")))
            return self.__process_messages(response["messages"])
        except GraphInterrupt:
            state = await graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
            logger.info("graph_interrupted", session_id=session_id, interrupt_value=str(interrupt_value))
            return [Message(role="assistant", content=str(interrupt_value))]
        except Exception as e:
            logger.exception("get_response_failed", error=str(e), session_id=session_id)
            raise

    async def get_stream_response(
            self,
            messages: list[Message],
            session_id: str,
            user_id: Optional[str] = None,
            username: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.
            username (Optional[str]): The display name of the user.

        Yields:
            str: Tokens of the LLM response.
        """
        callbacks: list[BaseCallbackHandler] = [langfuse_callback_handler] if settings.LANGFUSE_TRACING_ENABLED else []
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }
        graph = await self._get_graph()

        try:
            state, relevant_memory = await asyncio.gather(
                graph.aget_state(config),
                memory_service.search(user_id, messages[-1].content),
            )

            if state.next:
                logger.info("resuming_interrupted_graph_stream", session_id=session_id, next_nodes=state.next)
                graph_input = Command(resume=messages[-1].content)

            else:
                relevant_memory = relevant_memory or "No relevant memory found."
                graph_input = {
                    "messages": dump_messages(messages),
                    "long_term_memory": relevant_memory,
                    "tool_call_count": 0,
                    "subtasks": [],
                    "subtask_results": [],
                    "action_history": [],
                }
                message_stream = cast(
                    AsyncGenerator[tuple[BaseMessage, dict], None],
                    graph.astream(graph_input, config, stream_mode="messages"),
                )
                async for token, metadata in message_stream:
                    # Only the final answer streams to the client — "plan"'s
                    # classification and "worker"'s per-subtask research happen
                    # server-side. metadata["langgraph_node"] is set by LangGraph
                    # to whichever node is currently executing, even when that
                    # node calls _chat/_tool_call internally (as "worker" does)
                    # rather than as their own registered graph nodes.
                    if metadata.get("langgraph_node") not in ("chat", "synthesize"):
                        continue
                    if not isinstance(token, (AIMessage, AIMessageChunk)):
                        continue

                    text = extract_text_content(token.content)
                    if text:
                        yield text
                # After streaming completes, check for interrupt or update memory
                state = await graph.aget_state(config)
                if state.next:
                    interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                    logger.info("graph_interrupted_stream", session_id=session_id, interrupt_value=str(interrupt_value))
                    yield str(interrupt_value)
                elif state.values and "messages" in state.values:
                    openai_msgs = cast(list[dict], convert_to_openai_messages(state.values["messages"]))
                    asyncio.create_task(memory_service.add(user_id, openai_msgs, config.get("metadata")))

        except GraphInterrupt:
            state = await graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
            logger.info("graph_interrupted_stream", session_id=session_id, interrupt_value=str(interrupt_value))
            yield str(interrupt_value)
        except Exception as stream_error:
            logger.exception("stream_processing_failed", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        graph = await self._get_graph()

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        state: StateSnapshot = await graph.aget_state(config=config)
        return self.__process_messages(state.values["messages"]) if state.values else []


    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        # keep just assistant and user messages
        return [
            Message(role=message["role"], content=str(message["content"]))
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()
            if conn_pool is None:
                raise RuntimeError("connection pool unavailable; cannot clear chat history")

            # Batch all DELETEs in a single pipeline round-trip
            async with conn_pool.connection() as conn:
                async with conn.pipeline():
                    for table in settings.CHECKPOINT_TABLES:
                        await conn.execute(
                            sql.SQL("DELETE FROM {} WHERE thread_id = %s").format(sql.Identifier(table)),
                            (session_id,),
                        )
                logger.info(
                    "checkpoint_tables_cleared_for_session",
                    tables=settings.CHECKPOINT_TABLES,
                    session_id=session_id,
                )

        except Exception as e:
            logger.error(
                "clear_chat_history_operation_failed",
                session_id=session_id,
                error=str(e),
            )
            raise
