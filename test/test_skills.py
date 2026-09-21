"""Tests for the skill registry in ``app/core/skills/__init__.py``."""

import pytest

from app.core import skills
from app.core.skills import (
    SKILLS,
    SkillDefinition,
    _discover_scripts,
    _load_skills,
    _parse_skill_file,
    list_skills_summary,
)


def write_skill(directory, name="demo", description="Does a demo.", body="# Demo\n\nStep one."):
    """Create ``<directory>/SKILL.md`` with valid frontmatter and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    return path


class TestParseSkillFile:
    def test_parses_name_description_and_body(self, tmp_path):
        path = write_skill(tmp_path / "demo")

        name, description, body = _parse_skill_file(path)

        assert name == "demo"
        assert description == "Does a demo."
        assert body == "# Demo\n\nStep one."

    def test_body_is_stripped(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: a\ndescription: b\n---\n\n\n  body  \n\n")

        assert _parse_skill_file(path)[2] == "body"

    def test_value_may_contain_colons(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: a\ndescription: Use when: the user asks e.g. 'why: because'\n---\nbody")

        _, description, _ = _parse_skill_file(path)

        assert description == "Use when: the user asks e.g. 'why: because'"

    def test_horizontal_rules_in_body_are_preserved(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: a\ndescription: b\n---\nfirst\n\n---\n\nsecond\n---\nthird")

        _, _, body = _parse_skill_file(path)

        assert body == "first\n\n---\n\nsecond\n---\nthird"

    def test_frontmatter_lines_without_colon_are_ignored(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: a\njust a stray line\ndescription: b\n---\nbody")

        name, description, _ = _parse_skill_file(path)

        assert (name, description) == ("a", "b")

    def test_extra_frontmatter_keys_are_tolerated(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: a\ndescription: b\nlicense: MIT\n---\nbody")

        assert _parse_skill_file(path)[:2] == ("a", "b")

    def test_file_without_frontmatter_raises(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("# Just markdown, no frontmatter")

        with pytest.raises(ValueError, match="frontmatter"):
            _parse_skill_file(path)

    def test_unterminated_frontmatter_raises(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: a\ndescription: b\n")

        with pytest.raises(ValueError, match="frontmatter"):
            _parse_skill_file(path)

    @pytest.mark.parametrize(
        "frontmatter",
        [
            "description: only a description",
            "name: only-a-name",
            "",
        ],
        ids=["missing-name", "missing-description", "empty-frontmatter"],
    )
    def test_missing_required_field_raises(self, tmp_path, frontmatter):
        path = tmp_path / "SKILL.md"
        path.write_text(f"---\n{frontmatter}\n---\nbody")

        with pytest.raises(ValueError, match="'name' or 'description'"):
            _parse_skill_file(path)

    def test_error_message_names_the_offending_file(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("no frontmatter")

        with pytest.raises(ValueError, match="SKILL.md"):
            _parse_skill_file(path)


class TestDiscoverScripts:
    def test_no_scripts_directory_returns_empty(self, tmp_path):
        assert _discover_scripts(tmp_path) == {}

    def test_empty_scripts_directory_returns_empty(self, tmp_path):
        (tmp_path / "scripts").mkdir()

        assert _discover_scripts(tmp_path) == {}

    def test_finds_python_scripts_by_filename_with_absolute_paths(self, tmp_path):
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "b_second.py").write_text("print('b')")
        (scripts_dir / "a_first.py").write_text("print('a')")

        found = _discover_scripts(tmp_path)

        assert list(found) == ["a_first.py", "b_second.py"]  # sorted, deterministic
        assert all(path.is_absolute() for path in found.values())
        assert found["a_first.py"] == (scripts_dir / "a_first.py").resolve()

    def test_ignores_non_python_files_and_nested_directories(self, tmp_path):
        scripts_dir = tmp_path / "scripts"
        (scripts_dir / "nested").mkdir(parents=True)
        (scripts_dir / "run.py").write_text("")
        (scripts_dir / "notes.txt").write_text("")
        (scripts_dir / "run.sh").write_text("")
        (scripts_dir / "nested" / "hidden.py").write_text("")

        assert list(_discover_scripts(tmp_path)) == ["run.py"]

    def test_directory_named_like_a_script_is_ignored(self, tmp_path):
        (tmp_path / "scripts" / "looks_like_script.py").mkdir(parents=True)

        assert _discover_scripts(tmp_path) == {}


class TestLoadSkills:
    @pytest.fixture
    def skills_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills, "_SKILLS_DIR", tmp_path)
        return tmp_path

    def test_loads_every_skill_directory(self, skills_dir):
        write_skill(skills_dir / "alpha", name="alpha")
        write_skill(skills_dir / "beta", name="beta")

        registry = _load_skills()

        assert set(registry) == {"alpha", "beta"}
        assert all(isinstance(skill, SkillDefinition) for skill in registry.values())

    def test_registry_is_keyed_by_frontmatter_name_not_directory_name(self, skills_dir):
        write_skill(skills_dir / "scr", name="skill-creator")

        registry = _load_skills()

        assert list(registry) == ["skill-creator"]
        assert registry["skill-creator"].name == "skill-creator"

    def test_attaches_discovered_scripts(self, skills_dir):
        write_skill(skills_dir / "with_scripts", name="with_scripts")
        (skills_dir / "with_scripts" / "scripts").mkdir()
        (skills_dir / "with_scripts" / "scripts" / "go.py").write_text("print('go')")
        write_skill(skills_dir / "without", name="without")

        registry = _load_skills()

        assert list(registry["with_scripts"].scripts) == ["go.py"]
        assert registry["without"].scripts == {}

    def test_skips_underscore_prefixed_directories(self, skills_dir):
        write_skill(skills_dir / "_private", name="private")
        write_skill(skills_dir / "__pycache__", name="cache")
        write_skill(skills_dir / "public", name="public")

        assert list(_load_skills()) == ["public"]

    def test_skips_directories_without_skill_md(self, skills_dir):
        (skills_dir / "no_skill_file").mkdir()
        write_skill(skills_dir / "real", name="real")

        assert list(_load_skills()) == ["real"]

    def test_skips_plain_files_next_to_skill_directories(self, skills_dir):
        (skills_dir / "README.md").write_text("not a skill")
        (skills_dir / "helper.py").write_text("")

        assert _load_skills() == {}

    def test_empty_directory_gives_empty_registry(self, skills_dir):
        assert _load_skills() == {}

    def test_malformed_skill_fails_loudly(self, skills_dir):
        (skills_dir / "broken").mkdir()
        (skills_dir / "broken" / "SKILL.md").write_text("no frontmatter here")

        with pytest.raises(ValueError, match="frontmatter"):
            _load_skills()


class TestListSkillsSummary:
    def test_placeholder_when_no_skills(self, monkeypatch):
        monkeypatch.setattr(skills, "SKILLS", {})

        assert list_skills_summary() == "(no skills available)"

    def test_one_line_per_skill_in_registry_order(self, monkeypatch):
        monkeypatch.setattr(
            skills,
            "SKILLS",
            {
                "alpha": SkillDefinition(name="alpha", description="First skill.", body="a"),
                "beta": SkillDefinition(name="beta", description="Second skill.", body="b"),
            },
        )

        assert list_skills_summary() == "- alpha: First skill.\n- beta: Second skill."


class TestSkillDefinition:
    def test_scripts_defaults_to_empty_and_is_not_shared_between_instances(self):
        first = SkillDefinition(name="a", description="d", body="b")
        second = SkillDefinition(name="b", description="d", body="b")

        assert first.scripts == {}
        assert first.scripts is not second.scripts


class TestBundledSkills:
    """Sanity checks on the skills that actually ship with the app."""

    def test_registry_is_not_empty(self):
        assert SKILLS

    @pytest.mark.parametrize("skill", list(SKILLS.values()), ids=list(SKILLS))
    def test_bundled_skill_is_well_formed(self, skill):
        assert skill.name.strip()
        assert skill.description.strip()
        assert "\n" not in skill.description  # rendered as a single "- name: description" line
        assert skill.body.strip()

    @pytest.mark.parametrize("skill", list(SKILLS.values()), ids=list(SKILLS))
    def test_bundled_skill_scripts_exist_on_disk(self, skill):
        for script_name, script_path in skill.scripts.items():
            assert script_path.is_file(), f"{skill.name}/{script_name} is registered but missing"
            assert script_path.name == script_name

    def test_summary_mentions_every_bundled_skill(self):
        summary = list_skills_summary()

        for skill in SKILLS.values():
            assert f"- {skill.name}: {skill.description}" in summary
