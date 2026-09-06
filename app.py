"""
Ayurvedic RAG Prototype -- Streamlit UI

Retrieval only. No LLM answer generation, no diagnosis, no treatment advice.

Flow:
    English patient complaint
      -> JinaColBERT-v2 retriever (rag/pipeline.py)
      -> ranked passages from the existing rag-corpus (Charaka Samhita)
      -> displayed as expandable result cards
"""

from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from rag.pipeline import PipelineError, RagPipeline, RetrievalResult
from utils.formatting import format_verse_numbers, result_card_title, truncate
from utils.progress import run_with_live_timer
from utils.samples import load_sample_queries, sample_label

st.set_page_config(
    page_title="CharakaSetu",
    page_icon="🌿",
    layout="wide",
)

RETRIEVER_LABEL = "JinaColBERT-v2"
SOURCE_LABEL = "Charaka Samhita"
CREDIT_NAME = "Sanganaka, IIT Kharagpur"
CREDIT_URL = "https://sanganaka-iitkgp.github.io/"

EXAMPLE_QUERY = (
    "Patient presents with fatigue, restlessness, discoloration of the skin, "
    "and bad taste in the mouth."
)

QUERY_INPUT_KEY = "symptom_input"


# --------------------------------------------------------------------------- #
# Theming -- warm, Ayurvedic-inspired accents layered on top of Streamlit's
# own theme variables, so it adapts correctly to light AND dark mode instead
# of forcing one palette. Only the hero banner / button use fixed brand
# colors (they're readable against either background by design); everything
# else reads var(--background-color) / var(--secondary-background-color) /
# var(--text-color), which Streamlit updates automatically when the user
# switches themes.
# --------------------------------------------------------------------------- #
def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@400;500;600&family=Noto+Sans+Devanagari:wght@500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        /* Subtle themed wash behind the whole app -- derived from the
           active theme's own colors, so it works in light and dark mode. */
        .stApp {
            background: linear-gradient(160deg, var(--background-color) 0%, var(--secondary-background-color) 100%);
        }

        /* Hero banner -- fixed brand gradient, readable in either theme */
        .ayur-hero {
            background: linear-gradient(120deg, #4F6F45 0%, #7C9A63 55%, #C89B3C 100%);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        }
        .ayur-hero h1 {
            font-family: 'Playfair Display', serif;
            color: #FFF8E7;
            font-size: 2.1rem;
            margin: 0 0 6px 0;
        }
        .ayur-hero p {
            color: #FBF3E1;
            font-size: 1rem;
            margin: 0;
            opacity: 0.95;
        }
        .ayur-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.18);
            color: #FFF8E7;
            border: 1px solid rgba(255,255,255,0.4);
            border-radius: 999px;
            padding: 3px 12px;
            font-size: 0.8rem;
            margin-top: 10px;
            margin-right: 8px;
        }

        /* Info banner -- theme-aware */
        .ayur-note {
            background: var(--secondary-background-color);
            color: var(--text-color);
            border-left: 4px solid #8FAE6E;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.92rem;
            margin-bottom: 18px;
        }

        /* Text area -- theme-aware background/text, fixed gold accent border */
        .stTextArea textarea {
            border: 1.5px solid #C89B3C !important;
            border-radius: 12px !important;
            background-color: var(--secondary-background-color) !important;
            color: var(--text-color) !important;
            font-size: 1rem !important;
        }

        /* Primary button -- fixed brand gradient */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(120deg, #A9642B 0%, #C89B3C 100%);
            border: none;
            border-radius: 999px;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            color: #FFF8E7;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        div.stButton > button[kind="primary"]:hover {
            filter: brightness(1.08);
        }

        /* Expander (result cards) -- theme-aware */
        div[data-testid="stExpander"] {
            background: var(--secondary-background-color);
            border: 1px solid rgba(200, 155, 60, 0.35);
            border-radius: 14px;
            margin-bottom: 12px;
        }
        div[data-testid="stExpander"] summary {
            font-family: 'Playfair Display', serif;
            font-size: 1.05rem;
            color: var(--text-color);
        }

        .devanagari-text {
            font-family: 'Noto Sans Devanagari', sans-serif;
            font-size: 1.15rem;
            color: var(--text-color);
            background: var(--background-color);
            border-radius: 8px;
            padding: 10px 14px;
            border-left: 3px solid #C89B3C;
        }
        .iast-text {
            font-style: italic;
            color: var(--text-color);
            background: var(--background-color);
            border-radius: 8px;
            padding: 8px 14px;
        }

        /* Sidebar -- theme-aware */
        section[data-testid="stSidebar"] {
            background: var(--secondary-background-color);
        }

        /* Live status line (elapsed timer / completion summary) */
        .ayur-status {
            color: var(--text-color);
            opacity: 0.85;
            font-size: 0.9rem;
            margin: 4px 0 14px 0;
        }

        /* Footer credit -- theme-aware */
        .ayur-footer {
            text-align: center;
            margin-top: 36px;
            padding: 14px 0 6px 0;
            border-top: 1px solid rgba(200, 155, 60, 0.35);
            color: var(--text-color);
            opacity: 0.75;
            font-size: 0.85rem;
        }
        .ayur-footer a {
            color: #C89B3C;
            font-weight: 600;
            text-decoration: none;
        }
        .ayur-footer a:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
        st.markdown(
            f"""
            <div class="ayur-hero">
                <h1>🌿 CharakaSetu</h1>
                <p>AI-powered retrieval of classical Ayurvedic knowledge from the
                {SOURCE_LABEL} &mdash; describe a patient's symptoms in English and
                discover relevant passages, verbatim.</p>
                <span class="ayur-badge">Retrieval method: {RETRIEVER_LABEL}</span>
                <span class="ayur-badge">Source: {SOURCE_LABEL}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )



def render_footer() -> None:
    st.markdown(
        f"""
        <div class="ayur-footer">
            Made by <a href="{CREDIT_URL}" target="_blank">{CREDIT_NAME}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Cached resource: load corpus + build retriever index once per process.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_pipeline() -> RagPipeline:
    pipeline = RagPipeline(retriever_name="jina-colbert-v2")
    pipeline.load()
    return pipeline


def render_result_card(result: RetrievalResult) -> None:
    with st.expander(result_card_title(result), expanded=(result.rank <= 3)):
        st.markdown(f"**Sthana:** {result.sthana or '-'}")

        if result.heading_path:
            st.markdown(f"**Section:** {result.heading_path}")

        st.markdown("**Sanskrit**")
        st.markdown(
            f'<div class="devanagari-text">{result.devanagari or "-"}</div>',
            unsafe_allow_html=True,
        )

        if result.iast:
            st.markdown("**IAST (transliteration)**")
            st.markdown(
                f'<div class="iast-text">{result.iast}</div>', unsafe_allow_html=True
            )

        st.markdown("**English Translation**")
        st.markdown(result.translation or "-")

        footer_col1, footer_col2 = st.columns(2)
        with footer_col1:
            st.caption(f"Verse: {format_verse_numbers(result.verse_numbers_list)}")
        with footer_col2:
            if result.source_url:
                st.caption(f"Source: [{truncate(result.source_url, 60)}]({result.source_url})")
            else:
                st.caption("Source: -")


def results_to_json(results: list[RetrievalResult], query: str) -> str:
    """
    Serialize retrieved results to a pretty-printed JSON string for
    copy/download. Deliberately excludes the internal relevance score --
    the UI doesn't surface it elsewhere, so the export stays consistent.
    """
    payload = {
        "query": query,
        "retriever": RETRIEVER_LABEL,
        "source": SOURCE_LABEL,
        "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result_count": len(results),
        "results": [
            {
                "rank": r.rank,
                "verse_id": r.verse_id,
                "chapter_title": r.chapter_title,
                "sthana": r.sthana,
                "heading_path": r.heading_path,
                "devanagari": r.devanagari,
                "iast": r.iast,
                "translation": r.translation,
                "verse_numbers": format_verse_numbers(r.verse_numbers_list),
                "source_url": r.source_url,
            }
            for r in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_export_section(results: list[RetrievalResult], query: str) -> None:
    json_str = results_to_json(results, query)

    with st.expander("📋 Copy or download these results as JSON", expanded=False):
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name="ayurvedic_rag_results.json",
            mime="application/json",
            use_container_width=False,
        )
        st.caption("Or use the copy icon in the top-right corner of the box below:")
        st.code(json_str, language="json")


def render_sample_picker() -> None:
    samples = load_sample_queries()
    if not samples:
        return

    options = ["— Select a sample case —"] + [sample_label(s) for s in samples]

    def _apply_sample() -> None:
        idx = st.session_state.get("_sample_choice_idx", 0)
        if idx > 0:
            st.session_state[QUERY_INPUT_KEY] = samples[idx - 1]["query"]

    st.selectbox(
        "📚 Try a sample case",
        options=range(len(options)),
        format_func=lambda i: options[i],
        index=0,
        key="_sample_choice_idx",
        on_change=_apply_sample,
    )


def _load_and_query(query: str, top_k: int):
    """Runs entirely inside the background thread used by run_with_live_timer.

    Combining "make sure the pipeline is loaded" and "run the query" into
    one call means the user sees a single, honest elapsed timer covering
    the whole request instead of two back-to-back progress widgets.
    """
    pipeline = get_pipeline()
    return pipeline.query(query, top_k=top_k)


def main() -> None:
    inject_theme()
    render_hero()

    st.markdown(
        '<div class="ayur-note">ℹ️ This tool retrieves relevant passages only. '
        "It does not generate medical claims, diagnoses, or treatment "
        "recommendations.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("🍃 Settings")
        top_k = st.slider("Top-K results", min_value=1, max_value=50, value=10, step=1)
        st.divider()
        st.markdown(
            "**Pipeline**\n\n"
            "Query → JinaColBERT-v2 → MaxSim → ranked verse IDs → "
            "corpus lookup → results"
        )
        st.caption(
            "Pilot eval (not final): Recall@10 53.70% · MRR 0.511 · nDCG@10 0.435"
        )
        st.divider()
        st.caption(
            "⏱️ First run encodes the whole corpus, which can take a while "
            "on CPU. Consider running `scripts/precompute_embeddings.py` "
            "ahead of time (see README) to skip that wait here."
        )

    render_sample_picker()

    if QUERY_INPUT_KEY not in st.session_state:
        st.session_state[QUERY_INPUT_KEY] = ""

    query = st.text_area(
        "Patient Symptoms / Clinical Complaint",
        height=140,
        placeholder=EXAMPLE_QUERY,
        key=QUERY_INPUT_KEY,
    )

    search_clicked = st.button("🔎 Retrieve Relevant Knowledge", type="primary")

    if not search_clicked:
        render_footer()
        return

    if not query or not query.strip():
        st.warning("Please enter a patient symptom description before searching.")
        render_footer()
        return

    try:
        timed = run_with_live_timer(
            _load_and_query,
            query,
            top_k,
            label="Retrieving relevant knowledge from the Charaka Samhita...",
        )
    except PipelineError as e:
        st.error(str(e))
        render_footer()
        return
    except Exception as e:  # noqa: BLE001 - last-resort guard for the UI
        st.error(f"Unexpected error during retrieval: {e}")
        render_footer()
        return

    results = timed.value
    st.markdown(
        f'<div class="ayur-status">✅ Completed in <b>{timed.elapsed_seconds:0.1f}s</b> '
        f"&nbsp;·&nbsp; finished at <b>{timed.completed_at}</b></div>",
        unsafe_allow_html=True,
    )

    st.subheader(f"🌱 Top {len(results)} results")

    render_export_section(results, query)

    for result in results:
        render_result_card(result)

    render_footer()


if __name__ == "__main__":
    main()