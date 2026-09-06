"""
scripts/precompute_embeddings.py

Builds the JinaColBERT-v2 document embedding cache for the existing
rag-corpus *ahead of time*, so `streamlit run app.py` doesn't have to
encode the whole corpus during someone's first interactive session.

Why this matters
-----------------
Encoding a few hundred/thousand verses with a transformer encoder can take
anywhere from seconds (GPU) to several minutes (CPU-only). Doing that
inside the Streamlit request path makes the app feel like it's "hanging."
Run this once -- ideally on whatever machine is fastest available to you
(a GPU box, a beefier CPU machine, etc.) -- and the resulting cache file
can be copied alongside the corpus to wherever the app actually runs.
The app (rag/pipeline.py) automatically picks up this cache the next time
it starts, keyed on the exact set of verse_ids, and skips re-encoding.

Usage
-----
    python scripts/precompute_embeddings.py
    python scripts/precompute_embeddings.py --corpus-dir data/rag-corpus
    python scripts/precompute_embeddings.py --device cuda --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python scripts/precompute_embeddings.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.corpus import CorpusError, load_corpus  # noqa: E402
from rag.retriever import RetrieverError, get_retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Path to the corpus directory (default: data/rag-corpus).",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Force a device. Default: auto-detect (cuda if available).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Documents encoded per forward pass (bumped up automatically on CUDA).",
    )
    parser.add_argument(
        "--max-doc-length",
        type=int,
        default=300,
        help="Max encoder sequence length for documents (tokens). Raise only "
        "if your corpus has genuinely long entries.",
    )
    args = parser.parse_args()

    print("Loading corpus...")
    try:
        corpus = load_corpus(args.corpus_dir)
    except CorpusError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    df = corpus.df
    print(f"Loaded {len(df)} documents from {corpus.source_path}")

    doc_ids = df["verse_id"].astype(str).tolist()
    doc_texts = df["search_text"].tolist()

    cache_path = Path(corpus.source_path).parent / ".embedding_cache" / "jina-colbert-v2.pkl"
    print(f"Cache will be written to: {cache_path}")

    print("Loading JinaColBERT-v2 (this downloads the model on first run)...")
    try:
        retriever = get_retriever(
            "jina-colbert-v2",
            device=args.device,
            batch_size=args.batch_size,
            max_doc_length=args.max_doc_length,
        )
    except RetrieverError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Encoding {len(doc_texts)} documents...")
    start = time.time()
    try:
        retriever.index_with_cache(doc_ids, doc_texts, cache_path=cache_path)
    except RetrieverError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.time() - start

    print(f"Done in {elapsed:0.1f}s. Cache written to: {cache_path}")
    print(
        "Copy the data/rag-corpus/.embedding_cache/ directory alongside your "
        "corpus wherever the Streamlit app runs to skip re-encoding there."
    )


if __name__ == "__main__":
    main()