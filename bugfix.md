# Bugfix log

Test-writing, reconnect/resume, and a sweep of the bugs that stopped the app from starting. Status per item: **Fixed**, **Open**, or **Not done**.

Full suite: `.venv/bin/python -m pytest` → **405 passed**. Every fix below was mutation-checked: re-introducing the bug in a scratch copy turns at least one test red.

## 1. Tests added (`test/`)

| File | Covers |
|---|---|
| `conftest.py` | Dummy `OPENAI_API_KEY` (the LLM registry builds clients at import) |
| `test_skills.py`, `test_skill_tools.py` | Skill registry, `load_skill`, `run_skill_script` (real subprocesses; only registered names runnable; no shell) |
| `test_sanitization.py`, `test_auth_utils.py` | Sanitizers, password rules, JWT create/verify incl. forged / expired / `alg: none` |
| `test_graph_utils.py`, `test_graph_resume.py` | Graph helpers; interrupt vs. orphaned-run detection on a real LangGraph graph |
| `test_settings.py` | Every `settings.X` referenced in `app/` is defined (source scan) |
| `test_cache.py`, `test_database_service.py`, `test_llm_service.py` | Cache backends, DB pool settings, tool binding |
| `test_session_naming.py`, `test_auth_api.py` | Title persistence + metrics; malformed token → 422 |
| `test_agent.py` | `get_response` entry paths, per-call tool budget, `close()` |
| `test_metrics.py`, `test_main.py` | Prometheus labels through nested routers; lifespan ordering/failure modes; routes |
| `test_run_stream.py` | Run buffer + resume on **both** backends (in-memory and Valkey Streams via `fakeredis`); leases, dead-producer detection, time limit, shutdown |
| `test_chat_stream_api.py` | HTTP level, with real mid-stream `http.disconnect`: drop → resume with no gaps/repeats, idempotent retry, 409s, access, headers |

Config: `pyproject.toml` has `pythonpath = ["."]`, `testpaths = ["test"]`. `test/testsplit.py` is a scratch file and is not collected.

## 2. Reconnect / resume: the `state.next` bug — **Fixed**

`state.next` is non-empty both for a real `ask_human` interrupt and for a run that died mid-flight, so after a dropped connection `Command(resume=...)` silently swallowed the user's message. Entry now branches three ways: pending `interrupt()` → `Command(resume=...)`; unfinished run whose last user message matches → `None` (continue from checkpoint); otherwise fresh turn. Helpers `pending_interrupt_value` / `is_orphaned_run_for` live in `app/utils/graph.py`.

Limits: continuing redoes only the in-flight node, so the user sees the answer restart and it may differ. A *different* message after a crash between an `AIMessage(tool_calls)` and its results leaves dangling tool calls (OpenAI will likely 400) — **Open**.

Token-level resume (section 5) removes the restart for ordinary disconnects; this node-level path now only matters after a hard crash of the server.

## 3. Bugs fixed

**Skills**
- `SkillDefinition` field was `script`, code used `scripts` (import crashed) → renamed. *(app/core/skills/__init__.py)*
- `logger.error("…", str(path))` raised `TypeError` instead of `ValueError` → keyword form.
- `SKILL_SCRIPT_*` settings — already added upstream in "first finish".
- Stale `# todo: add to pytest` removed; `_parse_skill_file` docstring corrected.

**Validation**
- `sanitize_email` / `verify_token` used `re.match(...$)`, which accepts a trailing `\n` → `re.fullmatch`.

**Startup / wiring**
- `app/main.py` was a stub with no `app`. Now has `lifespan` (cache init, run-stream init, mem0 warm-up, graph creation; cache/memory failures are non-fatal; shutdown stops in-flight runs, then closes agent, then cache), CORS, correlation-id, metrics, rate limiter, 422 handler, `/`, `/health`. CORS no longer combines a wildcard origin with credentials.
- `LangGraphAgent.close()` added so shutdown doesn't reach into a private attribute.
- `InMemoryCacheService.set` read `_default_ttl` (attribute is `default_ttl`) → `AttributeError` on every set.
- `database.py` read `settings.POOL_SIZE` / `POOL_MAX_OVERFLOW` → `POSTGRES_POOL_SIZE` / `POSTGRES_MAX_OVERFLOW`.
- `pgvector` was imported but neither declared nor locked → added to `pyproject.toml`, `uv lock` (only `pgvector` added).

**Graph / LLM**
- `get_response`: `await self._graph` → `await self._get_graph()` (was a `TypeError` on every `/chat`).
- `should_retry` used `openai` / `httpx` without importing them.
- `_chat` used non-existent `settings.MAX_TOOL_LIMIT` and ignored its `tool_call_limit` argument, so workers never got their smaller budget → compares against `tool_call_limit`.
- `RECURSION_LIMIT` setting was read but undefined → added (default 25).
- `LLM_TIMEOUT` / `LLM_RETRY_LIMIT` → `LLM_TOTAL_TIMEOUT` / `MAX_LLM_CALL_RETRIES`.
- `LLMService.bind_tools` called misspelled `bond_tools` and discarded the result → `self._llm = self._llm.bind_tools(tools)`.
- `_chat` called non-existent `get_current_llm()` → `get_llm()`.

**Other**
- `rag.py`: `RecursiveCharacterTextSplitter(cache_key=…)` → `chunk_size=` (blocked importing the whole app).
- `session_naming.py`: stray IDE imports (`envs.fastapi…`, `openai…reasoning`) removed; `result.title` → `generated_session_name.title`; `database_service.update(...)` (doesn't exist) → `update_session_name(session_id, title)`. Auto-generated titles were never being saved and every success was counted as an error.
- `auth.py`: `except ValueError as e/ve` handlers logged the wrong name → `NameError` → 500 instead of 422.
- `metrics.py`: `starlette-prometheus` 0.10 crashes on FastAPI 0.141's nested `_IncludedRouter` (`no attribute 'path'`) — every `/api/v1/*` request would 500. Replaced with `RouteTemplatePrometheusMiddleware`, which labels by the full route template after routing (bounded cardinality; 404s share one `unmatched` label; in-progress gauge is labelled by method only).

## 4. Open — needs a decision

**RAG (`app/services/rag.py`) — found while reading, not fixed**
- **Access control:** `search()` does `accessible_group_ids.append(group_id)` when the caller is *not* a member, contradicting its own docstring ("silently yields no results"). Should restrict to the group, not add it.
- `search_context` calls `self.search(int(user_id), query, settings.RAG_TOP_K)`; the 3rd positional parameter is `group_id`, so `RAG_TOP_K` (5) is passed as a group id — combined with the bug above, **every chat search includes group 5 for every user**. Should be `top_k=settings.RAG_TOP_K`.
- `list_accessible_documents` groups by `DocumentChunk.id`, giving one row per chunk (count always 1) and invalid SQL for the selected `Document` columns; should group by `Document.id`.
- RAG context never reaches the model: `GraphState.knowledge_base` is read in `_chat`, but nothing populates it and `system.md` has no `{knowledge_base}` placeholder. This is also why `graph.py` still imports `rag_service` unused — left in place as the placeholder for that wiring.

**Not done**
- `.env.development` has no `OPENAI_API_KEY` / Langfuse keys, so the app can't start locally without them (tests use a dummy).
- No `.gitignore`: `logs/` and `__pycache__/` show up untracked.
- `uv.lock` — run `uv sync` to pick up `pgvector` in your own environment.

## 5. Token-level resume on disconnect — **Fixed**

Generation is now decoupled from the HTTP connection (`app/services/run_stream.py`). The graph runs in a background task that appends every chunk to a per-run buffer; HTTP responses only *follow* the buffer, so a disconnect stops following, not generating. Because the run finishes, the answer is also checkpointed and `memory_service.add` runs (both were lost on disconnect before).

**Client contract**

| Request | Behaviour |
|---|---|
| `POST /chat/stream` | Starts a run. Every SSE event has an `id:`; the response header `X-Run-Id` names the run. Event `data` is unchanged (`content`, `done`, `request_id`). |
| `GET /chat/stream/{run_id}` + `Last-Event-ID: <last id received>` | Re-attach; receives exactly the events after that id (no gaps, no repeats). Works while running **and** for `STREAM_RUN_TTL_SECONDS` after it finishes, so a client that missed the end can still collect it. Omit the header to replay from the start. |
| `GET /chat/stream/active` | `{run_id}` if a response is being generated for the session (page reload), else 404. |
| `POST /chat/stream` with the **same** message while it is in flight | Idempotent: attaches to the existing run (no second generation on the thread). |
| `POST /chat/stream` with a **different** message while one is in flight | `409` with `{"detail": {"run_id": ...}}`. |
| `POST /chat` while a stream is in flight | `409` (previously would have run a second generation on the same thread). |
| Bad `Last-Event-ID` → `400`; unknown / expired / another session's run → `404` (indistinguishable, so runs can't be probed). |

Browser clients need `fetch` (not `EventSource`, which can't send `Authorization`); `X-Run-Id` is CORS-exposed. Send `Last-Event-ID` as the last `id:` actually received.

**Backends** (`VALKEY_HOST` selects, like the cache): `ValkeyRunBackend` uses one Redis Stream per run (`XADD` / `XREAD`, so "after id X" is native and any instance can serve a re-attach) with its own connection pool (blocking reads must not starve the cache's). `InMemoryRunBackend` is the fallback and is **single-process only** — with more than one worker/instance, configure Valkey. If Valkey is configured but down at startup the service degrades to the in-memory buffer instead of failing.

**Failure handling**
- One run per session: a lease (`SET NX PX`) renewed by the producer every `STREAM_LOCK_TTL_SECONDS / 3`. If the producer dies (crash, deploy), the lease lapses and followers receive a terminal "interrupted" event instead of hanging; re-sending the message then continues from the LangGraph checkpoint (section 2).
- Graph errors and the `STREAM_RUN_MAX_SECONDS` cap become a terminal event (same `content`/`done` convention as before). App shutdown cancels runs, each recording an "interrupted" event first.
- Keep-alive comments every `STREAM_HEARTBEAT_SECONDS`; `X-Accel-Buffering: no` so proxies don't hold tokens back.

**Settings:** `STREAM_RUN_TTL_SECONDS` (600), `STREAM_RUN_MAX_SECONDS` (600), `STREAM_LOCK_TTL_SECONDS` (30), `STREAM_HEARTBEAT_SECONDS` (15), rate limit key `RATE_LIMIT_CHAT_RESUME` (60/min).

**Known limits**
- The Valkey backend was tested against `fakeredis`, never a real Valkey/Redis server (none available here). Same commands, but run it once against the real thing.
- After a *hard* crash, the re-sent message starts a **new** `run_id` and regenerates the in-flight node; clients should reset the partial text when the run id changes.
- If the buffer lease expires while the producer is actually alive (e.g. a Valkey stall > `STREAM_LOCK_TTL_SECONDS`), a new POST could start a second concurrent run for the session.
- Re-POSTing the same message *after* the run finished starts a new turn (deliberately: a user may legitimately repeat "yes"); use `GET /chat/stream/{run_id}` with `Last-Event-ID` to collect a missed tail.
- Replayed events keep the original request's `request_id`.
- There is no "stop generating" endpoint: a user who closes the tab still lets the run finish (and spend tokens).
- Each token is one `XADD`; if that ever matters, coalesce chunks over a short window.
