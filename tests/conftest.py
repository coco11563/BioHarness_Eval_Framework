"""Pytest configuration.

The runner test module needs pytest-asyncio. CI installs it via
``pip install -e ".[dev]"``; if a local environment is missing it,
the async tests are skipped rather than failing the whole suite.
"""

from __future__ import annotations

collect_ignore: list[str] = []

try:
    import pytest_asyncio  # noqa: F401
except ImportError:  # pragma: no cover - dev-env optional
    collect_ignore.append("unit/test_runner.py")
