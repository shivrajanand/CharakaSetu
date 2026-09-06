"""
utils/samples.py

Loads curated example patient-complaint queries (data/sample_queries.json)
so the Streamlit UI can offer a "try a sample case" picker. Purely a
convenience for demoing the app -- has no effect on retrieval quality and
is not part of the rag/ pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_queries.json"


def load_sample_queries(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return the list of sample query records, or [] if the file is missing."""
    if not SAMPLES_PATH.exists():
        return []
    try:
        with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return data[:limit] if limit else data


def sample_label(sample: Dict[str, Any], max_len: int = 70) -> str:
    """Short label for a dropdown option, e.g. 'Jwara – Vataja Jwara'."""
    source = sample.get("source", {})
    disease = source.get("disease", "").strip()
    subtype = source.get("subtype", "").strip()
    if disease and subtype:
        return f"{disease} — {subtype}"
    query = sample.get("query", "")
    return (query[:max_len] + "…") if len(query) > max_len else query