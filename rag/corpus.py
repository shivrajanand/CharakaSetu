"""
rag/corpus.py

Loads and validates the existing `rag-corpus` (Charaka Samhita corpus).

This module does NOT create a new corpus. It only reads whatever already
exists on disk under `data/rag-corpus/` (or a path supplied by the caller)
and validates that it has the fields the rest of the pipeline expects.

Supported on-disk formats (first match wins):
    - a single file:      data/rag-corpus/corpus.parquet
    - a single file:      data/rag-corpus/corpus.jsonl
    - a single file:      data/rag-corpus/corpus.json
    - a single file:      data/rag-corpus/corpus.csv
    - a directory of .json / .jsonl shards inside data/rag-corpus/

If you already have `rag_utils.py` with a corpus loader, wire it in at the
marked spot below instead of using the generic loader -- that keeps this
prototype from duplicating existing logic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

REQUIRED_FIELDS = [
    "verse_id",
    "chapter_title",
    "sthana",
    "heading_path",
    "devanagari",
    "iast",
    "translation",
    "verse_numbers_list",
    "source_url",
]

# Fields concatenated together to build the text that gets indexed/searched.
SEARCH_TEXT_FIELDS = ["translation", "heading_path"]

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "rag-corpus"


class CorpusError(Exception):
    """Raised when the corpus is missing or malformed."""


@dataclass
class CorpusLoadResult:
    df: pd.DataFrame
    source_path: str


def _candidate_files(corpus_dir: Path) -> List[Path]:
    return [
        corpus_dir / "corpus.parquet",
        corpus_dir / "corpus.jsonl",
        corpus_dir / "corpus.json",
        corpus_dir / "corpus.csv",
    ]


def _load_shards(corpus_dir: Path) -> pd.DataFrame:
    shard_files = sorted(corpus_dir.glob("*.jsonl")) + sorted(corpus_dir.glob("*.json"))
    if not shard_files:
        raise CorpusError(
            f"No corpus file or shards found in '{corpus_dir}'. "
            "Expected one of corpus.parquet / corpus.jsonl / corpus.json / "
            "corpus.csv, or a directory of .json/.jsonl shards."
        )
    records = []
    for shard in shard_files:
        with open(shard, "r", encoding="utf-8") as f:
            if shard.suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                else:
                    records.append(data)
    return pd.DataFrame.from_records(records)


def _load_single_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame.from_records(rows)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame.from_records(data if isinstance(data, list) else [data])
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise CorpusError(f"Unsupported corpus file type: {path}")


def validate_fields(df: pd.DataFrame) -> None:
    """Raise CorpusError if any required field is missing."""
    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        raise CorpusError(
            "Corpus is missing required field(s): "
            f"{', '.join(missing)}. Found columns: {list(df.columns)}"
        )
    if df.empty:
        raise CorpusError("Corpus loaded but contains zero rows.")


def _build_search_text(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `search_text` column = translation + heading_path (per spec)."""

    def _row_text(row) -> str:
        parts = []
        for field in SEARCH_TEXT_FIELDS:
            val = row.get(field, "")
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        return " | ".join(parts)

    df = df.copy()
    df["search_text"] = df.apply(_row_text, axis=1)
    return df


def load_corpus(corpus_dir: Optional[os.PathLike] = None) -> CorpusLoadResult:
    """
    Load the existing rag-corpus from disk.

    Parameters
    ----------
    corpus_dir : path to the corpus directory. Defaults to data/rag-corpus/.

    Returns
    -------
    CorpusLoadResult with a validated dataframe (plus a `search_text` column)
    and the path it was loaded from.

    Raises
    ------
    CorpusError if nothing usable is found or required fields are missing.
    """
    corpus_dir = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR

    if not corpus_dir.exists():
        raise CorpusError(
            f"Corpus directory not found: '{corpus_dir}'. "
            "Place the existing rag-corpus there (see README)."
        )

    # --- If you already have rag_utils.py with a loader, call it here instead ---
    # from rag_utils import load_rag_corpus
    # df = load_rag_corpus(corpus_dir)
    # -----------------------------------------------------------------------------

    df = None
    source_path = ""
    for candidate in _candidate_files(corpus_dir):
        if candidate.exists():
            df = _load_single_file(candidate)
            source_path = str(candidate)
            break

    if df is None:
        df = _load_shards(corpus_dir)
        source_path = str(corpus_dir)

    validate_fields(df)
    df = _build_search_text(df)

    return CorpusLoadResult(df=df, source_path=source_path)
