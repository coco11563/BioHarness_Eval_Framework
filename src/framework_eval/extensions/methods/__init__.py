"""QAClient plugin methods shipped with the extensions layer."""

from framework_eval.extensions.methods.oai_chat import OAIChat
from framework_eval.extensions.methods.rag_qdrant import RagQdrant

__all__ = ["OAIChat", "RagQdrant"]
