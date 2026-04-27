"""Skill registry — every Brain capability is a `@skill` function.

A skill is a plain Python callable plus a JSON schema + a description. The
registry auto-discovers built-ins under `brain/skills/*.py` and user skills
under `<vault>/.brain/skills/*.py`. Hot reload is just a path scan from the
daemon's filesystem watcher.

The MCP server (`brain.mcp`) and the HTTP server both expose the registry —
the same function fires whether the caller is a CLI, an HTTP client, or
Jarvis 2 over MCP.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_PRIMITIVE_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class Skill:
    name: str
    description: str
    fn: Callable[..., Any]
    schema: dict
    trusted: bool = False
    source_path: Optional[Path] = None


@dataclass
class SkillRegistry:
    skills: dict[str, Skill] = field(default_factory=dict)

    def register(self, skill: Skill) -> None:
        self.skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self.skills:
            raise KeyError(f"unknown skill: {name}")
        return self.skills[name]

    def list_skills(self) -> list[Skill]:
        return sorted(self.skills.values(), key=lambda s: s.name)

    def invoke(self, name: str, **kwargs) -> Any:
        return self.get(name).fn(**kwargs)

    def clear_user(self) -> None:
        for k in list(self.skills.keys()):
            sk = self.skills[k]
            if sk.source_path and ".brain/skills" in str(sk.source_path).replace("\\", "/"):
                del self.skills[k]


_REGISTRY = SkillRegistry()


def get_registry() -> SkillRegistry:
    return _REGISTRY


def _python_type_to_jsonschema(t: Any) -> dict:
    origin = typing.get_origin(t)
    args = typing.get_args(t)
    if origin is None:
        if t in _PRIMITIVE_TYPES:
            return {"type": _PRIMITIVE_TYPES[t]}
        if t is type(None):
            return {"type": "null"}
        return {"type": "string"}
    if origin in (list, tuple):
        item_t = args[0] if args else str
        return {"type": "array", "items": _python_type_to_jsonschema(item_t)}
    if origin is dict:
        return {"type": "object"}
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_jsonschema(non_none[0])
        return {"oneOf": [_python_type_to_jsonschema(a) for a in non_none]}
    return {"type": "string"}


def _build_schema(fn: Callable, override: Optional[dict] = None) -> dict:
    if override is not None:
        return override
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    props: dict[str, dict] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        ann = hints.get(pname, str)
        props[pname] = _python_type_to_jsonschema(ann)
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": props, "required": required}


def skill(
    name: Optional[str] = None,
    description: Optional[str] = None,
    schema: Optional[dict] = None,
    trusted: bool = False,
):
    """Register a function as a callable skill in the global registry."""

    def decorator(fn: Callable) -> Callable:
        skill_name = name or fn.__name__
        doc_lines = (inspect.getdoc(fn) or "").splitlines()
        first_line = doc_lines[0] if doc_lines else ""
        skill_desc = description or first_line or skill_name
        s = Skill(
            name=skill_name,
            description=skill_desc,
            fn=fn,
            schema=_build_schema(fn, override=schema),
            trusted=trusted,
            source_path=Path(inspect.getfile(fn)) if hasattr(fn, "__code__") else None,
        )
        _REGISTRY.register(s)
        return fn

    return decorator


def discover_builtins() -> int:
    """Import all modules under `brain.skills` to register their @skill fns."""
    skills_pkg = importlib.import_module("brain.skills")
    pkg_dir = Path(skills_pkg.__file__).parent
    n = 0
    for py in sorted(pkg_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        mod_name = f"brain.skills.{py.stem}"
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)
        n += 1
    return n


def discover_user(vault_path: Path) -> int:
    """Import user-supplied skill modules from `<vault>/.brain/skills/*.py`."""
    skills_dir = vault_path / ".brain" / "skills"
    if not skills_dir.exists():
        return 0
    _REGISTRY.clear_user()
    n = 0
    for py in sorted(skills_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"_brain_user_skill_{py.stem}", py)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
            n += 1
        except Exception:
            continue
    return n
