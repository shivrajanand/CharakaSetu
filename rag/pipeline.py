"""
rag/pipeline.py

Glue layer:

    User query
      -> retriever.search()
      -> ranked verse_ids
      -> corpus metadata lookup
      -> formatted results (utils/formatting.py)

Streamlit-facing code (app.py) should only ever call functions in this
module -- it should not import rag.corpus or rag.retriever directly. That
keeps the retriever swap-out contained to this file + retriever.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from rag.corpus import CorpusError, CorpusLoadResult, load_corpus
from rag.retriever import BaseRetriever, RetrieverError, get_retriever


class PipelineError(Exception):
    """User-facing pipeline error (wraps corpus/retriever errors)."""


@dataclass
class RetrievalResult:
    rank: int
    score: float
    verse_id: str
    chapter_title: str
    sthana: str
    heading_path: str
    devanagari: str
    iast: str
    translation: str
    verse_numbers_list: str
    source_url: str


class RagPipeline:
    """
    Holds a loaded corpus + an indexed retriever. Meant to be constructed
    once per process (see app.py's `@st.cache_resource` usage) and reused
    across queries.
    """

    def __init__(self, retriever_name: str = "jina-colbert-v2"):
        self.retriever_name = retriever_name
        self.corpus: Optional[CorpusLoadResult] = None
        self.retriever: Optional[BaseRetriever] = None
        self._id_to_row = {}

    def load(self, corpus_dir: Optional[str] = None) -> None:
        """Load the corpus and build the retriever's index. Call once."""
        try:
            self.corpus = load_corpus(corpus_dir)
        except CorpusError as e:
            raise PipelineError(str(e)) from e

        df = self.corpus.df
        self._id_to_row = {row["verse_id"]: row for _, row in df.iterrows()}

        try:
            self.retriever = get_retriever(self.retriever_name)
            doc_ids = df["verse_id"].astype(str).tolist()
            doc_texts = df["search_text"].tolist()

            if hasattr(self.retriever, "index_with_cache"):
                cache_path = Path(self.corpus.source_path).parent / ".embedding_cache" / (
                    f"{self.retriever_name}.pkl"
                )
                self.retriever.index_with_cache(doc_ids, doc_texts, cache_path=cache_path)
            else:
                self.retriever.index(doc_ids, doc_texts)
        except RetrieverError as e:
            raise PipelineError(str(e)) from e

    @property
    def is_ready(self) -> bool:
        return self.corpus is not None and self.retriever is not None

    @property
    def doc_count(self) -> int:
        """Number of indexed documents, or 0 if the corpus isn't loaded yet."""
        if self.corpus is None:
            return 0
        return len(self.corpus.df)

    def query(self, text: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self.is_ready:
            raise PipelineError("Pipeline is not loaded. Call load() first.")
        if not text or not text.strip():
            raise PipelineError("Please enter a patient symptom description before searching.")

        try:
            hits = self.retriever.search(text.strip(), top_k=top_k)
        except RetrieverError as e:
            raise PipelineError(f"Retrieval failed: {e}") from e

        results: List[RetrievalResult] = []
        for hit in hits:
            row = self._id_to_row.get(hit.verse_id)
            if row is None:
                # Shouldn't happen, but don't let one bad id crash the UI.
                continue
            results.append(
                RetrievalResult(
                    rank=hit.rank,
                    score=hit.score,
                    verse_id=str(row.get("verse_id", "")),
                    chapter_title=str(row.get("chapter_title", "")),
                    sthana=str(row.get("sthana", "")),
                    heading_path=str(row.get("heading_path", "")),
                    devanagari=str(row.get("devanagari", "")),
                    iast=str(row.get("iast", "")),
                    translation=str(row.get("translation", "")),
                    verse_numbers_list=str(row.get("verse_numbers_list", "")),
                    source_url=str(row.get("source_url", "")),
                )
            )
        if not results:
            raise PipelineError("No results could be retrieved for this query.")
        return results