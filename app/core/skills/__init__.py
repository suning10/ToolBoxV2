"""Skill registry: markdown-defined procedures the agent can load on demand.

A skill is distinct from a tool (see ``app/core/langgraph/tools/``): a tool is
a raw function call, while a skill is instructions on *when and how* to
sequence one or more tools to accomplish a class of task. Skills are not
always-on — they are pulled into context by the agent itself via the
``load_skill`` tool, keeping the base system prompt lean.

Skill files live alongside this module as ``*.md`` with '---' delimited
frontmatter, e.g.:

    ---
    name: web_research
    description: One-line description used to decide when to load this skill.
    ---
    <markdown body with step-by-step guidance>
"""

from pathlib import Path

from pydantic import BaseModel

from app.core.logging import logger

_SKILLS_DIR = Path(__file__).parent
_FRONTMATTER_DELIMITER = "---"


class SkillDefinition(BaseModel):
    """A single skill: metadata plus its full instruction body."""

    name: str
    description: str
    body: str


def _parse_skill_file(path: Path) -> SkillDefinition:
    """Parse a skill markdown file with '---' delimited frontmatter.

    Args:
        path: Path to the skill markdown file.

    Returns:
        SkillDefinition: The parsed skill.

    Raises:
        ValueError: If the file is missing frontmatter or required fields.
    """
    raw = path.read_text()
    parts = raw.split(_FRONTMATTER_DELIMITER, 2)
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
        raise ValueError(f"skill file missing 'name' or 'description' in frontmatter: {path.name}")

    return SkillDefinition(name=fields["name"], description=fields["description"], body=body.strip())


def _load_skills() -> dict[str, SkillDefinition]:
    """Scan the skills directory and build the registry once at import time.

    Returns:
        dict[str, SkillDefinition]: Skills keyed by name.
    """
    registry: dict[str, SkillDefinition] = {}
    for path in sorted(_SKILLS_DIR.glob("*.md")):
        skill = _parse_skill_file(path)
        registry[skill.name] = skill

    logger.info("skills_loaded", skill_count=len(registry), skill_names=list(registry.keys()))
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
