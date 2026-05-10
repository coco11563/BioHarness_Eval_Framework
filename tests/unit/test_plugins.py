"""Tests for the plugin protocol + discovery."""

from __future__ import annotations

import sys
import types
from typing import cast

import pytest

from framework_eval.eval.types import Item, Prediction
from framework_eval.plugins import (
    QAClient,
    discover_entry_points,
    load_method,
    parse_dotted_path,
)


# ---------- protocol --------------------------------------------------


class DummyClient:
    name = "dummy"

    async def generate(self, item: Item) -> Prediction:  # type: ignore[override]
        return Prediction(item_id=item.id, answer="yes")

    async def aclose(self) -> None: ...


def test_dummy_client_satisfies_protocol() -> None:
    assert isinstance(cast(QAClient, DummyClient()), QAClient)


def test_protocol_check_rejects_missing_method() -> None:
    class Broken:
        name = "broken"
    assert not isinstance(cast(QAClient, Broken()), QAClient)


# ---------- parse_dotted_path -----------------------------------------


def test_parse_dotted_path_ok() -> None:
    mod, attr = parse_dotted_path("a.b.c:Class")
    assert mod == "a.b.c"
    assert attr == "Class"


def test_parse_dotted_path_missing_colon() -> None:
    with pytest.raises(ValueError, match="pkg.module:Class"):
        parse_dotted_path("a.b.c.Class")


def test_parse_dotted_path_empty_half() -> None:
    with pytest.raises(ValueError, match="malformed"):
        parse_dotted_path(":Class")
    with pytest.raises(ValueError, match="malformed"):
        parse_dotted_path("a.b:")


# ---------- load_method via dotted path -------------------------------


def _install_temp_module(name: str, body: dict[str, object]) -> None:
    module = types.ModuleType(name)
    for k, v in body.items():
        setattr(module, k, v)
    sys.modules[name] = module


def test_load_method_imports_class(tmp_path) -> None:  # noqa: ANN001
    _install_temp_module("framework_eval_test_dummy_mod", {"DummyClient": DummyClient})
    cls = load_method("framework_eval_test_dummy_mod:DummyClient")
    assert cls is DummyClient


def test_load_method_unknown_attribute() -> None:
    _install_temp_module("framework_eval_test_empty", {})
    with pytest.raises(AttributeError, match="no attribute"):
        load_method("framework_eval_test_empty:Missing")


def test_load_method_target_is_function_not_class() -> None:
    def make_client() -> DummyClient:
        return DummyClient()

    _install_temp_module(
        "framework_eval_test_func", {"make_client": make_client}
    )
    with pytest.raises(TypeError, match="expected a class"):
        load_method("framework_eval_test_func:make_client")


# ---------- entry-point discovery -------------------------------------


def test_discover_entry_points_returns_sorted_list() -> None:
    """Even with zero registrations the call must work and be sorted."""
    entries = discover_entry_points()
    assert entries == sorted(entries, key=lambda e: e.name)


def test_load_method_via_entry_point_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """If an entry point matches the target name, resolve through it."""
    from framework_eval.plugins import discovery

    fake_entry = discovery.MethodEntry(
        name="my-method",
        target="framework_eval_test_ep:DummyClient",
        source="entry_point",
    )
    _install_temp_module("framework_eval_test_ep", {"DummyClient": DummyClient})
    monkeypatch.setattr(discovery, "discover_entry_points", lambda: [fake_entry])

    cls = load_method("my-method")
    assert cls is DummyClient
