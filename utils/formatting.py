"""
utils/formatting.py

Small presentation helpers kept separate from rag/ so display concerns never
leak into retrieval logic.
"""

from __future__ import annotations

from typing import List

from rag.pipeline import RetrievalResult


def result_card_title(result: "RetrievalResult") -> str:
    """Title line for an expandable result card, e.g. '#1  Jwara Chikitsa'."""
    title = result.chapter_title.strip() or "Untitled section"
    return f"#{result.rank}  {title}"


def format_verse_numbers(raw: str) -> str:
    """
    verse_numbers_list may already be a clean string ('12.34') or a
    stringified list ("['12.34', '12.35']"). Normalize either into a
    comma-separated, bracket-free display string.
    """
    if not raw:
        return "-"
    cleaned = raw.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
        cleaned = cleaned.replace("'", "").replace('"', "")
    return cleaned.strip() or "-"


def truncate(text: str, max_chars: int = 220) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."
