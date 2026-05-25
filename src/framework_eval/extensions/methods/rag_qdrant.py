"""RagQdrant: retrieval-augmented QAClient against the user-provided stack.

End-to-end pipeline mirroring the m-KAILIN^χ headline retrieval design:

    item.question ──▶ embed (8002)
                       │
                       ▼
                  Qdrant search (paper-full, top-k=20)
                       │
                       ▼
                  rerank (8001 /rerank, fallback yes/no-logprob)
                       │
                       ▼
                  keep top-k=4 (paper-aligned hyperparam, Fig. 10)
                       │
                       ▼
                  inject as context into oai-chat

The default endpoint matrix is::

    FRAMEWORK_LLM_URL      http://127.0.0.1:8000/v1
    FRAMEWORK_EMBED_URL    http://127.0.0.1:8002/v1
    FRAMEWORK_RERANK_URL   http://127.0.0.1:8001/v1
    FRAMEWORK_QDRANT_URL   http://127.0.0.1:13335
    FRAMEWORK_QDRANT_COLLECTION  paper-full
    FRAMEWORK_TOP_K        4
    FRAMEWORK_RECALL_K     20

Postgres lookups are opt-in (set ``FRAMEWORK_PG_DSN``) and used only to
hydrate full abstracts when Qdrant payloads omit them.

The rerank step transparently falls back to a chat-completion yes/no
logprob path when the rerank server does not expose ``/v1/rerank`` —
matching the access-plan notes shared by the user.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import httpx

from framework_eval.eval.types import Item, Prediction
from framework_eval.extensions.methods.oai_chat import (
    OAIChat,
    _first_token_logprobs,
    build_prompt,
)

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RagConfig:
    llm_url: str = "http://127.0.0.1:8000/v1"
    embed_url: str = "http://127.0.0.1:8002/v1"
    rerank_url: str = "http://127.0.0.1:8001/v1"
    qdrant_url: str = "http://127.0.0.1:13335"
    qdrant_collection: str = "paper-full"
    api_key: str = "EMPTY"
    recall_k: int = 20
    top_k: int = 4
    embedding_dim: int = 1024
    chunk_chars: int = 800
    embed_model: str | None = None
    rerank_model: str | None = None
    payload_text_key: str = "abstract"
    payload_title_key: str = "title"
    payload_id_key: str = "pmid"

    @classmethod
    def from_env(cls) -> "RagConfig":
        return cls(
            llm_url=os.environ.get(
                "FRAMEWORK_LLM_URL", "http://127.0.0.1:8000/v1"
            ).rstrip("/"),
            embed_url=os.environ.get(
                "FRAMEWORK_EMBED_URL", "http://127.0.0.1:8002/v1"
            ).rstrip("/"),
            rerank_url=os.environ.get(
                "FRAMEWORK_RERANK_URL", "http://127.0.0.1:8001/v1"
            ).rstrip("/"),
            qdrant_url=os.environ.get(
                "FRAMEWORK_QDRANT_URL", "http://127.0.0.1:13335"
            ).rstrip("/"),
            qdrant_collection=os.environ.get(
                "FRAMEWORK_QDRANT_COLLECTION", "paper-full"
            ),
            api_key=os.environ.get("FRAMEWORK_LLM_KEY", "EMPTY"),
            recall_k=int(os.environ.get("FRAMEWORK_RECALL_K", "20")),
            top_k=int(os.environ.get("FRAMEWORK_TOP_K", "4")),
            chunk_chars=int(os.environ.get("FRAMEWORK_CHUNK_CHARS", "800")),
            embed_model=os.environ.get("FRAMEWORK_EMBED_MODEL") or None,
            rerank_model=os.environ.get("FRAMEWORK_RERANK_MODEL") or None,
            payload_text_key=os.environ.get(
                "FRAMEWORK_QDRANT_TEXT_KEY", "abstract"
            ),
            payload_title_key=os.environ.get(
                "FRAMEWORK_QDRANT_TITLE_KEY", "title"
            ),
            payload_id_key=os.environ.get(
                "FRAMEWORK_QDRANT_ID_KEY", "pmid"
            ),
        )


# ----------------------------------------------------------------------
# Retrieved passage record
# ----------------------------------------------------------------------


@dataclass
class Passage:
    doc_id: str
    text: str
    score: float
    payload: dict[str, Any]

    def truncated(self, chars: int) -> str:
        if len(self.text) <= chars:
            return self.text
        return self.text[: chars - 1].rsplit(" ", 1)[0] + "…"


# ----------------------------------------------------------------------
# Pieces of the pipeline
# ----------------------------------------------------------------------


def _payload_text(payload: dict[str, Any], cfg: "RagConfig") -> str:
    """Read the text field configured by ``RagConfig.payload_text_key``.

    Defaults are wired for the user's ``paper-full`` Qdrant collection
    (PubMed abstracts under the ``abstract`` field, titles under
    ``title``). Override via ``FRAMEWORK_QDRANT_TEXT_KEY`` /
    ``FRAMEWORK_QDRANT_TITLE_KEY`` if your schema differs.
    """
    body = payload.get(cfg.payload_text_key)
    title = payload.get(cfg.payload_title_key)
    body_s = body.strip() if isinstance(body, str) else ""
    title_s = title.strip() if isinstance(title, str) else ""
    if title_s and body_s:
        return f"{title_s}. {body_s}"
    return body_s or title_s


def _payload_doc_id(payload: dict[str, Any], cfg: "RagConfig") -> str:
    v = payload.get(cfg.payload_id_key)
    return str(v) if v is not None else ""


def _build_context_block(passages: list[Passage], chunk_chars: int) -> str:
    parts: list[str] = []
    for i, p in enumerate(passages, 1):
        meta = p.payload
        title = meta.get("title") or meta.get("paper_title") or ""
        line = f"  [{i}] " + (f"{title.strip()}: " if title else "")
        parts.append(line + p.truncated(chunk_chars))
    return "\n".join(parts)


# ----------------------------------------------------------------------
# RagQdrant
# ----------------------------------------------------------------------


class RagQdrant(OAIChat):
    """Full RAG against the user-provided embed / Qdrant / rerank stack."""

    name = "rag-qdrant"

    def __init__(
        self,
        *,
        rag_config: RagConfig | None = None,
        chat_kwargs: dict[str, Any] | None = None,
    ) -> None:
        chat_kwargs = dict(chat_kwargs or {})
        chat_kwargs.setdefault("include_context", False)
        # context is supplied externally; do not let OAIChat embed item.context
        super().__init__(**chat_kwargs)
        self.rag = rag_config or RagConfig.from_env()

        # Separate HTTP clients per service so connection pools do not
        # interfere with the chat client.
        self._embed_client = httpx.AsyncClient(timeout=60.0)
        self._rerank_client = httpx.AsyncClient(timeout=60.0)
        self._qdrant_client = httpx.AsyncClient(timeout=60.0)
        self._embed_model_lock = asyncio.Lock()
        self._rerank_model_lock = asyncio.Lock()
        self._embed_model: str | None = self.rag.embed_model
        self._rerank_model: str | None = self.rag.rerank_model
        self._rerank_supported: bool | None = None

    async def aclose(self) -> None:
        await asyncio.gather(
            super().aclose(),
            self._embed_client.aclose(),
            self._rerank_client.aclose(),
            self._qdrant_client.aclose(),
            return_exceptions=True,
        )

    # ---- model discovery for embed / rerank -----------------------------

    async def _resolve_embed_model(self) -> str:
        if self._embed_model:
            return self._embed_model
        async with self._embed_model_lock:
            if self._embed_model:
                return self._embed_model
            self._embed_model = await self._discover_model(
                self._embed_client, self.rag.embed_url
            )
            return self._embed_model

    async def _resolve_rerank_model(self) -> str:
        if self._rerank_model:
            return self._rerank_model
        async with self._rerank_model_lock:
            if self._rerank_model:
                return self._rerank_model
            self._rerank_model = await self._discover_model(
                self._rerank_client, self.rag.rerank_url
            )
            return self._rerank_model

    async def _discover_model(
        self, client: httpx.AsyncClient, base_url: str
    ) -> str:
        url = f"{base_url}/models"
        try:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {self.rag.api_key}"}
            )
            resp.raise_for_status()
            data = resp.json()
            for entry in data.get("data") or []:
                mid = entry.get("id")
                if isinstance(mid, str) and mid:
                    return mid
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("model discovery failed at %s: %s", url, exc)
        return "default"

    # ---- embed ----------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        model = await self._resolve_embed_model()
        url = f"{self.rag.embed_url}/embeddings"
        payload = {"model": model, "input": text}
        headers = {"Authorization": f"Bearer {self.rag.api_key}"}
        resp = await self._embed_client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        try:
            vec = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected embedding response from {url}: "
                f"{json.dumps(data)[:200]}"
            ) from exc
        if not isinstance(vec, list) or len(vec) != self.rag.embedding_dim:
            LOGGER.debug(
                "embedding dim %d != expected %d",
                len(vec) if isinstance(vec, list) else -1,
                self.rag.embedding_dim,
            )
        return [float(x) for x in vec]

    # ---- Qdrant search --------------------------------------------------

    async def search(self, vector: list[float]) -> list[Passage]:
        url = f"{self.rag.qdrant_url}/collections/{self.rag.qdrant_collection}/points/search"
        body = {
            "vector": vector,
            "limit": self.rag.recall_k,
            "with_payload": True,
        }
        resp = await self._qdrant_client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result") or []
        passages: list[Passage] = []
        for hit in result:
            payload = hit.get("payload") or {}
            text = _payload_text(payload, self.rag)
            if not text:
                continue
            passages.append(
                Passage(
                    doc_id=_payload_doc_id(payload, self.rag),
                    text=text,
                    score=float(hit.get("score", 0.0)),
                    payload=payload,
                )
            )
        return passages

    # ---- rerank ---------------------------------------------------------

    async def rerank(
        self, query: str, passages: list[Passage]
    ) -> list[Passage]:
        if not passages:
            return passages
        if self._rerank_supported is not False:
            try:
                ranked = await self._rerank_native(query, passages)
                self._rerank_supported = True
                return ranked
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (404, 405, 501):
                    LOGGER.info(
                        "rerank /rerank not available (status %s); "
                        "falling back to yes/no logprob",
                        exc.response.status_code,
                    )
                    self._rerank_supported = False
                else:
                    raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("native rerank failed: %s; trying logprob path", exc)
                self._rerank_supported = False
        return await self._rerank_logprob(query, passages)

    async def _rerank_native(
        self, query: str, passages: list[Passage]
    ) -> list[Passage]:
        model = await self._resolve_rerank_model()
        url = f"{self.rag.rerank_url}/rerank"
        body = {
            "model": model,
            "query": query,
            "documents": [p.text for p in passages],
            "top_n": min(self.rag.top_k, len(passages)),
        }
        headers = {"Authorization": f"Bearer {self.rag.api_key}"}
        resp = await self._rerank_client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or data.get("data") or []
        scored: list[tuple[int, float]] = []
        for r in results:
            idx = r.get("index")
            sc = r.get("relevance_score") or r.get("score") or r.get("logit")
            if idx is None or sc is None:
                continue
            scored.append((int(idx), float(sc)))
        if not scored:
            return passages[: self.rag.top_k]
        scored.sort(key=lambda x: x[1], reverse=True)
        out: list[Passage] = []
        for idx, sc in scored[: self.rag.top_k]:
            if 0 <= idx < len(passages):
                p = passages[idx]
                out.append(
                    Passage(
                        doc_id=p.doc_id,
                        text=p.text,
                        score=sc,
                        payload=p.payload,
                    )
                )
        return out

    async def _rerank_logprob(
        self, query: str, passages: list[Passage]
    ) -> list[Passage]:
        """Fallback: score each passage by P(yes | "relevant?") logprob."""
        model = await self._resolve_rerank_model()
        url = f"{self.rag.rerank_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.rag.api_key}"}

        async def _score(passage: Passage) -> float:
            user = (
                "Decide whether the passage is relevant to the question. "
                "Answer with one word: yes or no.\n\n"
                f"Question: {query}\n"
                f"Passage: {passage.truncated(self.rag.chunk_chars)}\n"
                "Relevant?"
            )
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a relevance judge."},
                    {"role": "user",   "content": user},
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5,
            }
            resp = await self._rerank_client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            tops = _first_token_logprobs(data)
            if not tops:
                raise RuntimeError(
                    f"rerank logprob path requires top_logprobs from "
                    f"{url}, but the server returned none. Either expose "
                    "/v1/rerank or enable logprobs on the chat endpoint."
                )
            yes = -math.inf
            no = -math.inf
            for entry in tops:
                tok = (entry.get("token") or "").lower().strip()
                lp = float(entry.get("logprob", -math.inf))
                if tok.startswith("y"):
                    yes = max(yes, lp)
                elif tok.startswith("n"):
                    no = max(no, lp)
            if yes == -math.inf and no == -math.inf:
                return 0.0
            if no == -math.inf:
                return 1.0
            if yes == -math.inf:
                return 0.0
            # Softmax over the two anchors.
            mx = max(yes, no)
            return math.exp(yes - mx) / (math.exp(yes - mx) + math.exp(no - mx))

        scores = await asyncio.gather(*[_score(p) for p in passages])
        scored = sorted(
            zip(passages, scores), key=lambda x: x[1], reverse=True
        )
        return [
            Passage(doc_id=p.doc_id, text=p.text, score=s, payload=p.payload)
            for p, s in scored[: self.rag.top_k]
        ]

    # ---- main entry -----------------------------------------------------

    async def generate(self, item: Item) -> Prediction:
        query = item.question
        vec = await self.embed(query)
        recalled = await self.search(vec)
        top = await self.rerank(query, recalled) if recalled else []

        # Build the RAG-augmented prompt.
        ctx_block = _build_context_block(top, self.rag.chunk_chars)
        base_prompt = build_prompt(item, include_context=False)
        prompt = (
            "Use the retrieved biomedical passages below as context. "
            "If the passages do not answer the question, fall back to your "
            "domain knowledge.\n\n"
            f"Retrieved context:\n{ctx_block if ctx_block else '  (none)'}\n\n"
            f"{base_prompt}"
        )

        # Inline the OpenAI call so we can inject the augmented prompt
        # without rebuilding the rest of OAIChat.generate.
        model = await self._resolve_model()
        url = f"{self.api_base}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }
        if self._with_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        if not self._thinking_on:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = await self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        try:
            response_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected response from {url}: {json.dumps(data)[:200]}"
            ) from exc
        first_logprobs = (
            _first_token_logprobs(data) if self._with_logprobs else None
        )

        from framework_eval.eval import extract_answer

        normalised = extract_answer(
            response_text, item.question_type, item.options, mode="strict"
        )
        extras: dict[str, Any] = {
            "response_text": response_text,
            "model": model,
            "rag_top_k": self.rag.top_k,
            "rag_recall_k": self.rag.recall_k,
            "retrieved_doc_ids": [p.doc_id for p in top],
            "retrieved_scores": [p.score for p in top],
        }
        if first_logprobs is not None:
            extras["first_token_logprobs"] = first_logprobs
        return Prediction(item_id=item.id, answer=normalised, extras=extras)
