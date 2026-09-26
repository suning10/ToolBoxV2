"""Every ``settings.X`` the app reads must be defined on ``Settings``.

A misspelled or never-added setting only fails when that line runs (often deep inside a request
or at import), so this scans the source instead of waiting for it to blow up.
"""

import pathlib
import re

from app.core.config import settings

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"
REFERENCE = re.compile(r"\bsettings\.([A-Z][A-Z0-9_]*)")


def test_every_referenced_setting_is_defined():
    missing = {}
    for path in APP_DIR.rglob("*.py"):
        for match in REFERENCE.finditer(path.read_text()):
            if not hasattr(settings, match.group(1)):
                missing.setdefault(match.group(1), set()).add(path.relative_to(APP_DIR.parent).as_posix())

    assert not missing, "settings referenced but not defined: " + "; ".join(
        f"{name} (in {', '.join(sorted(files))})" for name, files in sorted(missing.items())
    )


def test_limits_used_by_the_agent_have_sane_values():
    assert settings.RECURSION_LIMIT > 0
    assert settings.MAX_TOOL_CALLS_PER_WORKER <= settings.MAX_TOOL_CALLS_PER_TURN
    assert settings.SKILL_SCRIPT_TIMEOUT_SECONDS > 0
    assert settings.SKILL_SCRIPT_MAX_OUTPUT_CHARS > 0
