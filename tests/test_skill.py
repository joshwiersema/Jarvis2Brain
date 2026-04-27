"""Skill registry tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from brain.skill import Skill, SkillRegistry, discover_builtins, discover_user, skill


@pytest.fixture(autouse=True)
def _clean_registry():
    from brain.skill import _REGISTRY

    snapshot = dict(_REGISTRY.skills)
    yield
    _REGISTRY.skills.clear()
    _REGISTRY.skills.update(snapshot)


def test_skill_decorator_registers():
    @skill(name="echo", description="Echoes input")
    def echo(text: str) -> str:
        return text

    from brain.skill import get_registry

    s = get_registry().get("echo")
    assert s.name == "echo"
    assert s.fn("hi") == "hi"


def test_schema_inferred_from_signature():
    @skill(name="add")
    def add(a: int, b: int = 0) -> int:
        return a + b

    from brain.skill import get_registry

    s = get_registry().get("add")
    assert s.schema["properties"]["a"] == {"type": "integer"}
    assert s.schema["properties"]["b"] == {"type": "integer"}
    assert "a" in s.schema["required"]
    assert "b" not in s.schema["required"]


def test_optional_param_handled():
    @skill(name="opt")
    def opt(x: Optional[str] = None) -> str:
        return x or "default"

    from brain.skill import get_registry

    s = get_registry().get("opt")
    assert s.schema["properties"]["x"] == {"type": "string"}


def test_invoke_via_registry():
    @skill(name="doubler")
    def d(x: int) -> int:
        return x * 2

    from brain.skill import get_registry

    assert get_registry().invoke("doubler", x=21) == 42


def test_unknown_skill_raises():
    r = SkillRegistry()
    with pytest.raises(KeyError):
        r.invoke("nope")


def test_discover_builtins_imports_skills():
    n = discover_builtins()
    from brain.skill import get_registry

    names = {s.name for s in get_registry().list_skills()}
    assert n >= 4
    # Sanity check core skills are wired up.
    assert "memory_search" in names
    assert "notes_create" in names
    assert "graph_export" in names
    assert "brain_train" in names


def test_discover_user_loads_files(tmp_path: Path):
    skills_dir = tmp_path / ".brain" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "demo.py").write_text(
        "from brain.skill import skill\n"
        "@skill(name='demo_user', description='user demo')\n"
        "def demo() -> str:\n"
        "    return 'hello'\n",
        encoding="utf-8",
    )
    n = discover_user(tmp_path)
    assert n == 1
    from brain.skill import get_registry

    assert get_registry().get("demo_user").fn() == "hello"


def test_list_for_mcp_friendly_payload():
    @skill(name="x", description="x desc")
    def x() -> str:
        return ""

    from brain.skill import get_registry

    payload = [
        {"name": s.name, "desc": s.description, "schema": s.schema}
        for s in get_registry().list_skills()
    ]
    assert any(p["name"] == "x" for p in payload)
