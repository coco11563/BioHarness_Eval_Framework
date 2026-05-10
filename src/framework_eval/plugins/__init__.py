"""Plugin discovery for external evaluation methods."""

from framework_eval.plugins.discovery import (
    ENTRY_POINT_GROUP,
    MethodEntry,
    discover_entry_points,
    iter_all_methods,
    load_method,
    parse_dotted_path,
)
from framework_eval.plugins.protocol import QAClient

__all__ = [
    "ENTRY_POINT_GROUP",
    "MethodEntry",
    "QAClient",
    "discover_entry_points",
    "iter_all_methods",
    "load_method",
    "parse_dotted_path",
]
