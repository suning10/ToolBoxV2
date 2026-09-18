"""Skill-loading tool for LangGraph.

Lets the agent pull a skill's markdown instructions into context on demand,
separating *what tools exist* (this package) from *how and when to use them
for a given class of task* (``app/core/skills/<name>/SKILL.md``).
"""

from langchain_core.tools import tool

from app.core.logging import logger
from app.core.skills import SKILLS

@tool
def load_skill(skill_name: str) -> str:
    """Load a skill's step-by-step instructions by name.

    Call this when the user's request matches one of the skills listed in
    your system prompt, before taking further action. The returned text
    contains guidance for handling that class of task, including which
    other tools to use and how. If the skill bundles scripts, run them via
    ``run_skill_script`` rather than reimplementing their logic yourself.

    Args:
        skill_name: The exact name of the skill to load.

    Returns:
        str: The skill's full instructions (plus a list of its runnable
            scripts, if any), or an error message listing the available
            skill names if no skill matches.
    """
    skill = SKILLS.get(skill_name)
    if skill is None:
        logger.warning("skill_not_found", requested=skill_name, available=list(SKILLS.keys()))
        return f"skill '{skill_name}' not found. available skills: {', '.join(SKILLS.keys())}"

    logger.info("skill_loaded", skill_name=skill_name, scripts=list(skill.scripts.keys()))
    if not skill.scripts:
        return skill.body

    script_list = ", ".join(skill.scripts.keys())
    return (
        f"{skill.body}\n\n"
        f"---\n"
        f"Available scripts for this skill (run via run_skill_script(skill_name=\"{skill.name}\", "
        f"script_name=...)): {script_list}"
    )
