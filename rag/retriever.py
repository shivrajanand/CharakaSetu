"""
rag/retriever.py

Defines a swappable retriever interface plus the current prototype
implementation, JinaColBERT-v2 (jinaai/jina-colbert-v2) using late-interaction
MaxSim scoring.

Design goal (per project spec): the Streamlit app and pipeline only ever talk
to `BaseRetriever`. Swapping in BM25 / EmbeddingGemma / a reranker later means
writing a new class here -- nothing else changes.
"""

from __future__ import annotations

import abc
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

QUERY_MAX_TOKENS = 32  # JinaColBERT-v2 query width -- do not change without cause.


class RetrieverError(Exception):
    """Raised when a retriever fails to load or fails during search."""


@dataclass
class RetrievedItem:
    verse_id: str
    score: float
    rank: int


class BaseRetriever(abc.ABC):
    """Interface every retriever (JinaColBERT, BM25, EmbeddingGemma, ...) implements."""

    name: str = "base"

    @abc.abstractmethod
    def index(self, doc_ids: List[str], doc_texts: List[str]) -> None:
        """Build (or load a cached) index over the given corpus texts."""
        raise NotImplementedError

    @abc.abstractmethod
    def search(self, query: str, top_k: int) -> List[RetrievedItem]:
        """Return the top_k most relevant doc_ids for the query, best first."""
        raise NotImplementedError


class JinaColBERTRetriever(BaseRetriever):
    """
    Late-interaction (ColBERT-style) retriever backed by jinaai/jina-colbert-v2.

    jina-colbert-v2's multi-vector API (encode_query / encode_document) is
    only exposed through `sentence_transformers.MultiVectorEncoder`
    (sentence-transformers>=6.0.0). A plain `transformers.AutoModel` does NOT
    have these methods, so that is not used here.

    Scoring: MaxSim -- for each query token embedding, take the max cosine/dot
    similarity across all document token embeddings, then sum over query
    tokens. This is the standard ColBERT scoring function.

    Notes
    -----
    * Model/query/doc tensors coming back from the model may be BF16 and/or
      live on CUDA. Anything that needs NumPy is converted with
      `.detach().cpu().float().numpy()` first -- BF16 has no direct NumPy
      dtype and CUDA tensors can't be converted directly. (If the encoder
      already returns a plain NumPy array, it's used as-is.)
    * Document embeddings are computed once in `index()` and cached in memory
      for the lifetime of the process; Streamlit-level caching (see
      rag/pipeline.py) avoids recomputing them across reruns/queries.
    """

    name = "JinaColBERT-v2"
    MODEL_NAME = "jinaai/jina-colbert-v2"

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_doc_length: int = 300,
    ):
        """
        Parameters
        ----------
        device : "cuda" / "cpu" / None (auto-detect).
        batch_size : documents encoded per forward pass. Bumped up
            automatically on CUDA (see _encode_documents) since GPUs have
            slack to batch more aggressively.
        max_doc_length : caps the encoder's max sequence length (in tokens).
            jina-colbert-v2 supports up to 8192 tokens, but our documents
            (translation + heading_path) are short. Without this cap the
            encoder may pad/attend over a much longer sequence than needed,
            which is the single biggest CPU/GPU speed killer for this model.
            Raise this only if your corpus has genuinely long documents.
        """
        self._device = device
        self._batch_size = batch_size
        self._max_doc_length = max_doc_length
        self._model = None
        self._doc_ids: List[str] = []
        self._doc_embeddings: List[np.ndarray] = []  # one (n_tokens, dim) array per doc

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _ensure_model_loaded(self):
        if self._model is not None:
            return

        import os

        try:
            import torch
        except ImportError as e:
            raise RetrieverError(
                "PyTorch is required for JinaColBERTRetriever but is not installed."
            ) from e

        try:
            from sentence_transformers import MultiVectorEncoder
        except ImportError as e:
            raise RetrieverError(
                "sentence-transformers>=6.0.0 is required for JinaColBERTRetriever "
                "(it provides MultiVectorEncoder, the supported way to get "
                "jina-colbert-v2's multi-vector encode_query/encode_document API). "
                "Plain `transformers.AutoModel` does not expose these methods. "
                "Install/upgrade with: pip install -U 'sentence-transformers>=6.0.0'"
            ) from e

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        if device == "cpu":
            # Many environments (containers, CI, some cloud VMs) leave torch
            # pinned to 1 thread by default, which makes CPU inference on a
            # model this size unusably slow. Use all available cores.
            try:
                torch.set_num_threads(max(1, os.cpu_count() or 1))
            except Exception:  # noqa: BLE001 - best-effort only
                pass

        try:
            model = MultiVectorEncoder(
                self.MODEL_NAME, trust_remote_code=True, device=device
            )
        except Exception as e:  # noqa: BLE001 - surface as a friendly RetrieverError
            raise RetrieverError(
                f"Failed to load model '{self.MODEL_NAME}'. "
                "Check your internet connection / model cache. "
                f"Underlying error: {e}"
            ) from e

        # Cap sequence length -- see max_doc_length docstring above. This is
        # the single most impactful speed fix for this model on both CPU
        # and GPU: without it, encoding can implicitly run against a much
        # longer context window than these short verse/translation texts need.
        if hasattr(model, "max_seq_length"):
            try:
                model.max_seq_length = self._max_doc_length
            except Exception:  # noqa: BLE001 - best-effort only
                pass

        # Half precision roughly doubles GPU throughput for this kind of
        # encoder with negligible quality impact. Skip on CPU -- fp16/bf16
        # matmul is usually *slower* than fp32 on CPU.
        if device == "cuda":
            try:
                underlying = getattr(model, "model", None) or getattr(
                    model, "_model", None
                )
                if underlying is not None and hasattr(underlying, "half"):
                    underlying.half()
                elif hasattr(model, "half"):
                    model.half()
            except Exception:  # noqa: BLE001 - best-effort only
                pass

        self._model = model
        self._torch = torch

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #
    def _to_numpy(self, tensor) -> np.ndarray:
        """Safely convert a (possibly BF16, possibly CUDA) tensor to NumPy.

        MultiVectorEncoder may hand back either a NumPy array already or a
        torch tensor (BF16/CUDA) depending on version/config, so both are
        handled here.
        """
        if isinstance(tensor, np.ndarray):
            return tensor.astype(np.float32, copy=False)
        return tensor.detach().cpu().float().numpy()

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into (n_tokens<=32, dim) float32 NumPy array.

        jina-colbert-v2 truncates queries to QUERY_MAX_TOKENS (32) internally,
        so no explicit max-length argument is passed here.
        """
        self._ensure_model_loaded()
        if not hasattr(self._model, "encode_query"):
            raise RetrieverError(
                "Loaded model does not expose encode_query; check the "
                "jina-colbert-v2 model card for the current multi-vector API."
            )
        with self._torch.inference_mode():
            emb = self._model.encode_query(query)
        return self._to_numpy(emb)

    def _encode_documents(self, texts: List[str]) -> List[np.ndarray]:
        self._ensure_model_loaded()
        if not hasattr(self._model, "encode_document"):
            raise RetrieverError(
                "Loaded model does not expose encode_document; check the "
                "jina-colbert-v2 model card for the current multi-vector API."
            )
        # GPUs have slack to batch more aggressively than the configured
        # default (which is sized conservatively for CPU memory/latency).
        effective_batch = self._batch_size
        if self._device == "cuda":
            effective_batch = max(self._batch_size, 32)

        all_embs: List[np.ndarray] = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), effective_batch):
                batch = texts[i : i + effective_batch]
                # encode_document takes a list and returns one variable-length
                # (n_tokens, dim) array per document -- do not call it once
                # per string, it expects the whole batch.
                batch_embs = self._model.encode_document(batch)
                for emb in batch_embs:
                    all_embs.append(self._to_numpy(emb))
        return all_embs

    # ------------------------------------------------------------------ #
    # BaseRetriever interface
    # ------------------------------------------------------------------ #
    def index(self, doc_ids: List[str], doc_texts: List[str]) -> None:
        if len(doc_ids) != len(doc_texts):
            raise RetrieverError("doc_ids and doc_texts must be the same length.")
        self._doc_ids = list(doc_ids)
        self._doc_embeddings = self._encode_documents(doc_texts)

    def load_precomputed(self, doc_ids: List[str], doc_embeddings: List[np.ndarray]) -> None:
        """Attach previously-computed + cached document embeddings directly."""
        if len(doc_ids) != len(doc_embeddings):
            raise RetrieverError("doc_ids and doc_embeddings must be the same length.")
        self._doc_ids = list(doc_ids)
        self._doc_embeddings = list(doc_embeddings)

    def index_with_cache(
        self, doc_ids: List[str], doc_texts: List[str], cache_path: Optional[Path] = None
    ) -> None:
        """
        Like index(), but if `cache_path` exists and matches doc_ids exactly,
        load embeddings from disk instead of re-encoding the whole corpus.
        Writes the cache after a fresh encode so the next process start is fast.
        """
        if len(doc_ids) != len(doc_texts):
            raise RetrieverError("doc_ids and doc_texts must be the same length.")

        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("doc_ids") == list(doc_ids):
                    self._doc_ids = cached["doc_ids"]
                    self._doc_embeddings = cached["doc_embeddings"]
                    return
            except (pickle.UnpicklingError, EOFError, KeyError, OSError):
                pass  # fall through and re-encode

        self.index(doc_ids, doc_texts)

        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {"doc_ids": self._doc_ids, "doc_embeddings": self._doc_embeddings}, f
                    )
            except OSError:
                pass  # caching is best-effort; retrieval still works without it

    @property
    def doc_embeddings(self) -> List[np.ndarray]:
        return self._doc_embeddings

    @property
    def doc_ids(self) -> List[str]:
        return self._doc_ids

    @staticmethod
    def _maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """
        MaxSim: sum over query tokens of the max similarity to any doc token.
        query_emb: (Q, D), doc_emb: (T, D) -- both assumed L2-normalized-ish,
        we still normalize defensively so BM25-swap-in doesn't have to care.
        """
        q = query_emb / (np.linalg.norm(query_emb, axis=-1, keepdims=True) + 1e-8)
        d = doc_emb / (np.linalg.norm(doc_emb, axis=-1, keepdims=True) + 1e-8)
        sim = q @ d.T  # (Q, T)
        return float(sim.max(axis=1).sum())

    def search(self, query: str, top_k: int) -> List[RetrievedItem]:
        if not query or not query.strip():
            raise RetrieverError("Query is empty.")
        if not self._doc_ids:
            raise RetrieverError("Retriever has no indexed documents. Call index() first.")

        query_emb = self._encode_query(query)

        scores = np.array(
            [self._maxsim_score(query_emb, d) for d in self._doc_embeddings],
            dtype=np.float32,
        )
        order = np.argsort(-scores)[: min(top_k, len(scores))]

        return [
            RetrievedItem(verse_id=self._doc_ids[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(order)
        ]


def get_retriever(name: str = "jina-colbert-v2", **kwargs) -> BaseRetriever:
    """
    Factory so the pipeline/UI can request a retriever by name without
    importing a specific class. Add new retrievers here as they're built.
    """
    registry = {
        "jina-colbert-v2": JinaColBERTRetriever,
    }
    key = name.lower()
    if key not in registry:
        raise RetrieverError(
            f"Unknown retriever '{name}'. Available: {list(registry.keys())}"
        )
    return registry[key](**kwargs)