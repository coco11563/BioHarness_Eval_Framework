"""Hugging Face Hub loader for Shaow/BioHarness_Eval.

This module wraps ``huggingface_hub.snapshot_download`` so that we can pin
the dataset to the exact revision recorded in MANIFEST.toml. Downstream code
treats the result as a normal local directory and forwards to the JSONL
loader. The ``datasets`` library is intentionally avoided to keep the
dependency surface small.

Revision resolution order:
    1. The ``revision`` argument, when explicitly provided.
    2. The pinned ``dataset.hf_revision`` in the auto-discovered MANIFEST.toml.
    3. Caller opt-in via ``allow_main=True`` only; otherwise an unpinned
       call raises ``ValueError``. We never silently fall back to ``main``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from framework_eval.eval.types import Item, QuestionType
from framework_eval.loader.jsonl import load_config

DEFAULT_REPO_ID = "Shaow/BioHarness_Eval"

# Provenance side-table: maps a resolved snapshot path (str) -> (revision,
# source). We cannot stash this on the Path object itself because pathlib's
# ``PosixPath`` defines ``__slots__`` and therefore has no ``__dict__`` on
# Python 3.12+, so attribute assignment raises ``AttributeError``.
_RESOLVED_REVISIONS: dict[str, tuple[str, str]] = {}


def resolved_revision(path: Path) -> tuple[str, str] | None:
    """Return the ``(revision, source)`` recorded for a snapshot path, if any."""
    return _RESOLVED_REVISIONS.get(str(path))


def _candidate_manifest_paths() -> list[Path]:
    """Common locations a caller's MANIFEST.toml might live."""
    here = Path(__file__).resolve()
    pkg_root = here.parents[3]                       # …/framework_eval install root
    src_root = here.parents[2]                       # …/src/framework_eval/loader
    return [
        Path.cwd() / "MANIFEST.toml",
        pkg_root / "MANIFEST.toml",
        src_root.parent.parent / "MANIFEST.toml",
    ]


def _autodiscover_manifest() -> Path | None:
    for p in _candidate_manifest_paths():
        if p.exists():
            return p
    return None


def _read_manifest_revision(manifest_path: Path) -> str | None:
    if not manifest_path.exists():
        return None
    data = tomllib.loads(manifest_path.read_text())
    rev = data.get("dataset", {}).get("hf_revision")
    return rev if isinstance(rev, str) and rev else None


def snapshot_dataset(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: Path | None = None,
    manifest_path: Path | None = None,
    token: str | None = None,
    allow_main: bool = False,
) -> Path:
    """Download (or reuse) the HF dataset snapshot and return its local path.

    Revision resolution: explicit > manifest > error (unless ``allow_main``).
    """
    from huggingface_hub import snapshot_download

    resolved = revision
    source = "explicit"
    if resolved is None:
        manifest = manifest_path or _autodiscover_manifest()
        if manifest is not None:
            resolved = _read_manifest_revision(manifest)
            source = f"manifest:{manifest}"
    if resolved is None:
        if not allow_main:
            raise ValueError(
                "No HF revision pin: pass `revision=`, supply a "
                "`manifest_path=` containing `dataset.hf_revision`, place "
                "MANIFEST.toml on the discovery path, or set `allow_main=True` "
                "to opt into the unpinned default branch."
            )
        resolved = "main"
        source = "fallback:main"

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=token,
    )
    path = Path(local_path)
    # Stash provenance for callers that want to log or assert. ``Path`` has no
    # ``__dict__`` on py3.12+, so keep it in a side-table keyed by path.
    _RESOLVED_REVISIONS[str(path)] = (resolved, source)
    return path


def load_from_hub(
    config: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: Path | None = None,
    manifest_path: Path | None = None,
    token: str | None = None,
    allow_main: bool = False,
    dedup_by_id: bool = True,
    question_types: Iterable[QuestionType] | None = None,
) -> list[Item]:
    """Load a single config directly from the Hub. See ``snapshot_dataset``."""
    root = snapshot_dataset(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        token=token,
        allow_main=allow_main,
    )
    return load_config(
        root,
        config,
        dedup_by_id=dedup_by_id,
        question_types=tuple(question_types) if question_types is not None else None,
    )
