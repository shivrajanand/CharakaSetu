"""
utils/progress.py

Runs a slow function call (model load + retrieval) in a background thread
while showing the user an *honest* live status: elapsed time and the
current wall-clock time, both ticking in real time. No fake percentage or
guessed ETA -- a prior version of this tried to estimate completion time
and just froze near 100% once the guess was wrong, which was worse than
showing nothing.

After the call finishes, the caller gets the elapsed duration and the
completion timestamp back so it can show a short "Completed in Xs at
HH:MM:SS" summary.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Generic, TypeVar

import streamlit as st

T = TypeVar("T")


@dataclass
class TimedResult(Generic[T]):
    value: T
    elapsed_seconds: float
    completed_at: str  # HH:MM:SS


def run_with_live_timer(
    fn: Callable[..., T],
    *args: Any,
    label: str = "Working...",
    **kwargs: Any,
) -> TimedResult[T]:
    """
    Run fn(*args, **kwargs) in a background thread while rendering a live
    "Elapsed: Xs | Now: HH:MM:SS" status line. Any exception raised inside
    fn propagates to the caller (via future.result()) once the thread ends.
    """
    status_slot = st.empty()
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        while not future.done():
            elapsed = time.time() - start
            now_str = datetime.now().strftime("%H:%M:%S")
            status_slot.markdown(
                f"🌿 {label}  \n"
                f"*Elapsed: **{elapsed:0.1f}s** &nbsp;·&nbsp; "
                f"Current time: **{now_str}***"
            )
            time.sleep(0.2)

        elapsed_total = time.time() - start
        completed_at = datetime.now().strftime("%H:%M:%S")
        status_slot.empty()
        # Propagates any exception raised inside fn.
        value = future.result()

    return TimedResult(value=value, elapsed_seconds=elapsed_total, completed_at=completed_at)