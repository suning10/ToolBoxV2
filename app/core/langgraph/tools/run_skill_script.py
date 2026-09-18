"""Skill-script execution tool for LangGraph.

Some skills bundle a Python script for work an LLM shouldn't do in its head
(exact counting, deterministic data processing, etc. — see
``app/core/skills/text_stats/`` for an example). This tool is the *only* way
scripts get executed, and it deliberately cannot run arbitrary code: it looks
up ``script_name`` in that skill's pre-registered ``scripts`` dict (built by
``app/core/skills`` from what's actually on disk under the skill's
``scripts/`` directory) and rejects anything that isn't an exact match. The
LLM can choose *which* registered script to run and what arguments to pass
it, never an arbitrary path or shell command.
"""

import asyncio
import sys

from langchain_core.tools import tool

from app.core.config import settings
from app.core.logging import logger
from app.core.skills import SKILLS


@tool
async def run_skill_script(skill_name: str, script_name: str, script_args: list[str] | None = None) -> str:
    """Run one of a skill's bundled scripts and return its output.

    Only scripts already bundled with a skill can be run — check that
    skill's instructions (loaded via ``load_skill``) for the exact
    ``script_name`` and what arguments it expects before calling this.

    Args:
        skill_name: The skill that owns the script (e.g. "text_stats").
        script_name: The exact script filename, as listed in that skill's
            instructions (e.g. "text_stats.py").
        script_args: Command-line arguments to pass to the script, in order.

    Returns:
        str: The script's stdout on success. On failure, a message
            describing what went wrong (unknown skill/script, non-zero exit
            with stderr, or timeout) — never a raw traceback.
    """
    skill = SKILLS.get(skill_name)
    if skill is None:
        logger.warning("skill_script_unknown_skill", skill_name=skill_name)
        return f"skill '{skill_name}' not found. available skills: {', '.join(SKILLS.keys())}"

    script_path = skill.scripts.get(script_name)
    if script_path is None:
        logger.warning("skill_script_unknown_script", skill_name=skill_name, script_name=script_name)
        return (
            f"script '{script_name}' not found for skill '{skill_name}'. "
            f"available scripts: {', '.join(skill.scripts.keys()) or '(none)'}"
        )

    argv = [sys.executable, str(script_path), *(script_args or [])]
    logger.info(
        "skill_script_started", skill_name=skill_name, script_name=script_name, arg_count=len(script_args or [])
    )

    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=settings.SKILL_SCRIPT_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("skill_script_timed_out", skill_name=skill_name, script_name=script_name)
            return f"script '{script_name}' timed out after {settings.SKILL_SCRIPT_TIMEOUT_SECONDS}s"

        if process.returncode != 0:
            logger.warning(
                "skill_script_failed",
                skill_name=skill_name,
                script_name=script_name,
                return_code=process.returncode,
            )
            error_output = stderr.decode(errors="replace").strip() or "(no error output)"
            return f"script '{script_name}' failed (exit code {process.returncode}): {error_output}"

        output = stdout.decode(errors="replace").strip()
        if len(output) > settings.SKILL_SCRIPT_MAX_OUTPUT_CHARS:
            output = output[: settings.SKILL_SCRIPT_MAX_OUTPUT_CHARS] + "\n[output truncated]"

        logger.info("skill_script_completed", skill_name=skill_name, script_name=script_name)
        return output or "(script produced no output)"
    except Exception as e:
        logger.exception("skill_script_execution_error", skill_name=skill_name, script_name=script_name, error=str(e))
        return f"failed to run script '{script_name}': {str(e)}"
