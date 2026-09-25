"""Skill registry: markdown-defined procedures the agent can load on demand.

A skill is distinct from a tool (see ``app/core/langgraph/tools/``): a tool is
a raw function call, while a skill is instructions on *when and how* to
sequence one or more tools — including its own bundled scripts — to
accomplish a class of task. Skills are not always-on — they are pulled into
context by the agent itself via the ``load_skill`` tool, keeping the base
system prompt lean.

Each skill is a directory alongside this module containing a ``SKILL.md``
with '---' delimited frontmatter, and optionally a ``scripts/`` subdirectory
of Python scripts the agent can run via the ``run_skill_script`` tool:

    app/core/skills/
      web_research/
        SKILL.md
      text_stats/
        SKILL.md
        scripts/
          text_stats.py

SKILL.md format:

    ---
    name: web_research
    description: One-line description used to decide when to load this skill.
    ---
    <markdown body with step-by-step guidance>

Scripts are discovered by presence, not declared in frontmatter — anything
under a skill's ``scripts/`` directory is runnable by name via
``run_skill_script(skill_name, script_name)``. The SKILL.md body is
responsible for telling the model which scripts exist and how to call them
(see ``text_stats/SKILL.md`` for an example); ``load_skill`` also appends a
generated list as a safety net so the model never has to guess a script name.
"""
from pathlib import Path

from pydantic import BaseModel

from app.core.logging import logger

_SKILLS_DIR = Path(__file__).parent
_FRONTMATTER_DELIMITER = "---"
_SKILL_FILENAME = "SKILL.md"


class SkillDefinition(BaseModel):
    """A single skill: metadata plus its full instruction body."""

    name: str
    description: str
    body: str
    scripts: dict[str, Path] = {}


def _parse_skill_file(path: Path) -> tuple[str,str,str]:
    """Parse a skill markdown file with '---' delimited frontmatter.
    refer to skill.md
    Args:
        path: Path to the skill markdown file.

    Returns:
        tuple[str, str, str]: The skill's ``(name, description, body)``.

    Raises:
        ValueError: If the file is missing frontmatter or required fields.
    """
    raw = path.read_text()
    parts = raw.split(_FRONTMATTER_DELIMITER, 2) # use maxsplit to split into 3
    if len(parts) < 3:
        raise ValueError(f"skill file missing '---' delimited frontmatter: {path.name}")

    frontmatter, body = parts[1], parts[2]
    fields: dict[str, str] = {}
    for line in frontmatter.strip().splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        fields[key.strip()] = value.strip()

    if "name" not in fields or "description" not in fields:
        logger.error("skill_load_failed", path=str(path))
        raise ValueError(f"skill file missing 'name' or 'description' in frontmatter: {path.name}")

    return fields["name"], fields["description"], body.strip()

def _discover_scripts(skill_dir: Path) -> dict[str, Path]:
    """Find every script bundled under a skill's ``scripts/`` subdirectory.

    Args:
        skill_dir: The skill's root directory.

    Returns:
        dict[str, Path]: Script filename (e.g. "text_stats.py") to its
            resolved path. Only these exact, pre-registered names can be
            executed by ``run_skill_script`` — never an arbitrary path.
    """
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir(): # check if /scripts exists
        return {}
    # resolve the abs path for *.py
    return {path.name: path.resolve() for path in sorted(scripts_dir.glob("*.py")) if path.is_file()}


def _load_skills() -> dict[str, SkillDefinition]:
    """Scan the skills directory and build the registry once at import time.

    Returns:
        dict[str, SkillDefinition]: Skills keyed by name.
    """
    registry: dict[str, SkillDefinition] = {}
    for skill_dir in sorted(p for p in _SKILLS_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")):
        skill_file = skill_dir / _SKILL_FILENAME
        if not skill_file.is_file():
            continue

        name, description, body = _parse_skill_file(skill_file)
        scripts = _discover_scripts(skill_dir)

        registry[name] = SkillDefinition(name=name, description=description, body=body, scripts=scripts)

    logger.info(
        "skills_loaded",
        skill_count=len(registry),
        skill_names=list(registry.keys()),
        skills_with_scripts=[s.name for s in registry.values() if s.scripts],
    )
    return registry



# Read and parse once at module load — no file I/O per request.
SKILLS: dict[str, SkillDefinition] = _load_skills()

def list_skills_summary() -> str:
    """Return a formatted skill listing for injection into the system prompt.

    Returns:
        str: One "- name: description" line per skill, or a placeholder
            when no skills are registered.
    """
    if not SKILLS:
        return "(no skills available)"
    return "\n".join(f"- {skill.name}: {skill.description}" for skill in SKILLS.values())
