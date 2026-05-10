"""Method discovery: entry points + ad-hoc dotted-path imports.

Two ways to register a method:

1. **Entry point** in your own ``pyproject.toml``::

       [project.entry-points."framework_eval.methods"]
       my-method = "my_pkg.my_module:MyMethod"

   ``framework-eval list-methods`` will pick it up after ``pip install``.

2. **Ad-hoc dotted path** on the command line::

       framework-eval run --method my_pkg.my_module:MyMethod

   No installation required.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import entry_points

from framework_eval.plugins.protocol import QAClient

ENTRY_POINT_GROUP = "framework_eval.methods"


@dataclass(frozen=True)
class MethodEntry:
    name: str
    target: str             # "pkg.module:ClassName"
    source: str             # "entry_point" | "dotted_path"


def discover_entry_points() -> list[MethodEntry]:
    """Return every method registered via the ``framework_eval.methods``
    entry-point group across the active Python environment."""
    out: list[MethodEntry] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        target = ep.value if hasattr(ep, "value") else f"{ep.module}:{ep.attr}"
        out.append(MethodEntry(name=ep.name, target=target, source="entry_point"))
    return sorted(out, key=lambda m: m.name)


def parse_dotted_path(target: str) -> tuple[str, str]:
    """Split ``"pkg.module:Class"`` into module + attribute."""
    if ":" not in target:
        raise ValueError(
            f"Method target {target!r} must use 'pkg.module:Class' form."
        )
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ValueError(
            f"Method target {target!r} is malformed; "
            "expected 'pkg.module:Class' with both halves non-empty."
        )
    return module_name, attr


def load_method(target: str) -> type[QAClient]:
    """Import ``target`` and return the class.

    Accepts either an entry-point name (matched case-sensitively against
    the discovered set) or a ``pkg.module:Class`` dotted path.
    """
    # Entry-point name?
    for entry in discover_entry_points():
        if entry.name == target:
            target = entry.target
            break

    module_name, attr = parse_dotted_path(target)
    module = importlib.import_module(module_name)
    if not hasattr(module, attr):
        raise AttributeError(
            f"Module {module_name!r} has no attribute {attr!r} "
            f"(loading method target {target!r})."
        )
    cls = getattr(module, attr)
    if not isinstance(cls, type):
        raise TypeError(
            f"Method target {target!r} resolves to {type(cls).__name__}, "
            "expected a class."
        )
    return cls


def iter_all_methods() -> Iterator[MethodEntry]:
    """Yield every discoverable method (entry-points only)."""
    yield from discover_entry_points()
