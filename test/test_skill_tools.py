"""Tests for the ``load_skill`` and ``run_skill_script`` LangGraph tools."""

import asyncio
import sys

import pytest

from app.core.config import settings
from app.core.langgraph.tools.load_skill import load_skill
from app.core.langgraph.tools.run_skill_script import run_skill_script
from app.core.skills import SkillDefinition

# ``app.core.langgraph.tools`` re-exports the tool objects under the same names as their modules,
# so a dotted-string patch target would resolve to the tool, not the module. Go through sys.modules.
LOAD_SKILL_MODULE = sys.modules["app.core.langgraph.tools.load_skill"]
RUN_SKILL_SCRIPT_MODULE = sys.modules["app.core.langgraph.tools.run_skill_script"]

SCRIPTS = {
    "echo.py": "import sys\nprint(' '.join(sys.argv[1:]))\n",
    "fail.py": "import sys\nsys.stderr.write('boom\\n')\nsys.exit(3)\n",
    "fail_silently.py": "import sys\nsys.exit(2)\n",
    "hello.py": "print('hello from script')\n",
    "silent.py": "pass\n",
    "big.py": "print('x' * 1000)\n",
    "sleepy.py": "import time\ntime.sleep(30)\n",
}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Install a throwaway skill registry into both tool modules."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for filename, source in SCRIPTS.items():
        (scripts_dir / filename).write_text(source)

    demo = SkillDefinition(
        name="demo",
        description="A demo skill.",
        body="# Demo\n\nDo the thing.",
        scripts={filename: (scripts_dir / filename).resolve() for filename in SCRIPTS},
    )
    plain = SkillDefinition(name="plain", description="No scripts here.", body="Just words.")
    skills = {"demo": demo, "plain": plain}

    monkeypatch.setattr(LOAD_SKILL_MODULE, "SKILLS", skills)
    monkeypatch.setattr(RUN_SKILL_SCRIPT_MODULE, "SKILLS", skills)
    # raising=False: these limits are asserted to exist in ``test_settings_define_script_limits`` below, so
    # the behaviour tests stay independent of whether the settings class has been updated yet.
    monkeypatch.setattr(settings, "SKILL_SCRIPT_TIMEOUT_SECONDS", 10, raising=False)
    monkeypatch.setattr(settings, "SKILL_SCRIPT_MAX_OUTPUT_CHARS", 500, raising=False)
    return skills


def run_script(skill_name, script_name, script_args=None):
    payload = {"skill_name": skill_name, "script_name": script_name}
    if script_args is not None:
        payload["script_args"] = script_args
    return asyncio.run(run_skill_script.ainvoke(payload))


def test_settings_define_script_limits():
    assert isinstance(settings.SKILL_SCRIPT_TIMEOUT_SECONDS, (int, float))
    assert settings.SKILL_SCRIPT_TIMEOUT_SECONDS > 0
    assert isinstance(settings.SKILL_SCRIPT_MAX_OUTPUT_CHARS, int)
    assert settings.SKILL_SCRIPT_MAX_OUTPUT_CHARS > 0


class TestLoadSkill:
    def test_tool_metadata(self):
        assert load_skill.name == "load_skill"
        assert "skill_name" in load_skill.args

    def test_returns_body_for_skill_without_scripts(self, registry):
        assert load_skill.invoke({"skill_name": "plain"}) == "Just words."

    def test_appends_script_listing_for_skill_with_scripts(self, registry):
        result = load_skill.invoke({"skill_name": "demo"})

        assert result.startswith("# Demo\n\nDo the thing.")
        assert 'run_skill_script(skill_name="demo"' in result
        for script_name in SCRIPTS:
            assert script_name in result

    def test_unknown_skill_lists_available_names(self, registry):
        result = load_skill.invoke({"skill_name": "nope"})

        assert "'nope' not found" in result
        assert "demo" in result and "plain" in result

    def test_skill_name_must_match_exactly(self, registry):
        assert "not found" in load_skill.invoke({"skill_name": "Demo"})
        assert "not found" in load_skill.invoke({"skill_name": " demo"})
        assert "not found" in load_skill.invoke({"skill_name": ""})


class TestRunSkillScriptLookup:
    def test_tool_metadata(self):
        assert run_skill_script.name == "run_skill_script"
        assert {"skill_name", "script_name", "script_args"} <= set(run_skill_script.args)

    def test_unknown_skill(self, registry):
        result = run_script("nope", "echo.py")

        assert "skill 'nope' not found" in result
        assert "demo" in result

    def test_unknown_script_lists_available_scripts(self, registry):
        result = run_script("demo", "missing.py")

        assert "script 'missing.py' not found for skill 'demo'" in result
        assert "echo.py" in result

    def test_skill_without_scripts_reports_none_available(self, registry):
        result = run_script("plain", "echo.py")

        assert "not found" in result
        assert "(none)" in result

    @pytest.mark.parametrize(
        "script_name",
        [
            "../echo.py",
            "scripts/echo.py",
            "/bin/echo",
            "echo",
            "echo.py; echo pwned",
            "ECHO.PY",
            "",
        ],
    )
    def test_only_exact_registered_names_are_runnable(self, registry, script_name):
        result = run_script("demo", script_name)

        assert "not found for skill 'demo'" in result


class TestRunSkillScriptExecution:
    def test_returns_stripped_stdout(self, registry):
        assert run_script("demo", "echo.py", ["hello"]) == "hello"

    def test_arguments_are_passed_in_order(self, registry):
        assert run_script("demo", "echo.py", ["one", "two words", "three"]) == "one two words three"

    def test_arguments_are_optional(self, registry):
        assert run_script("demo", "hello.py") == "hello from script"
        assert run_script("demo", "hello.py", []) == "hello from script"

    def test_arguments_are_not_interpreted_by_a_shell(self, registry):
        args = ["; echo pwned", "$(echo pwned)", "`echo pwned`", "&& echo pwned", "*"]

        result = run_script("demo", "echo.py", args)

        assert result == "; echo pwned $(echo pwned) `echo pwned` && echo pwned *"

    def test_script_with_no_output(self, registry):
        assert run_script("demo", "silent.py") == "(script produced no output)"

    def test_non_zero_exit_reports_code_and_stderr(self, registry):
        result = run_script("demo", "fail.py")

        assert "failed" in result
        assert "exit code 3" in result
        assert "boom" in result
        assert "Traceback" not in result

    def test_non_zero_exit_without_stderr(self, registry):
        result = run_script("demo", "fail_silently.py")

        assert "exit code 2" in result
        assert "(no error output)" in result

    def test_long_output_is_truncated(self, registry, monkeypatch):
        monkeypatch.setattr(settings, "SKILL_SCRIPT_MAX_OUTPUT_CHARS", 100, raising=False)

        result = run_script("demo", "big.py")

        assert result == "x" * 100 + "\n[output truncated]"

    def test_output_at_exactly_the_limit_is_not_truncated(self, registry, monkeypatch):
        monkeypatch.setattr(settings, "SKILL_SCRIPT_MAX_OUTPUT_CHARS", 1000, raising=False)

        result = run_script("demo", "big.py")

        assert result == "x" * 1000

    def test_timeout_kills_the_script_and_reports_it(self, registry, monkeypatch):
        monkeypatch.setattr(settings, "SKILL_SCRIPT_TIMEOUT_SECONDS", 1, raising=False)

        result = asyncio.run(asyncio.wait_for(run_skill_script.ainvoke(
            {"skill_name": "demo", "script_name": "sleepy.py"}
        ), timeout=20))

        assert "timed out after 1s" in result

    def test_launch_failure_is_reported_not_raised(self, registry, monkeypatch):
        async def explode(*args, **kwargs):
            raise OSError("cannot spawn")

        monkeypatch.setattr(RUN_SKILL_SCRIPT_MODULE.asyncio, "create_subprocess_exec", explode)

        result = run_script("demo", "echo.py", ["hi"])

        assert result == "failed to run script 'echo.py': cannot spawn"
