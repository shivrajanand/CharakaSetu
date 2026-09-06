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
    page_title="Ayurvedic RAG Prototype",
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
# Theming -- warm, Ayurvedic-inspired palette (turmeric / leaf green / cream)
# --------------------------------------------------------------------------- #
def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@400;500;600&family=Noto+Sans+Devanagari:wght@500&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 10% 0%, #FBF3E1 0%, #F6ECD9 40%, #F1E6D0 100%);
        }

        /* Hero banner */
        .ayur-hero {
            background: linear-gradient(120deg, #5B7B4F 0%, #7C9A63 55%, #C89B3C 100%);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(91, 66, 26, 0.18);
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

        /* Info banner */
        .ayur-note {
            background: #EFF3E3;
            border-left: 4px solid #6E8F52;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.92rem;
            color: #3B4A2E;
            margin-bottom: 18px;
        }
        
        /* Retrieval pipeline */
.pipeline-card {
    background: #FFFDF7;
    border: 1px solid #DCCFA3;
    border-radius: 14px;
    padding: 16px 14px;
    margin: 10px 0 18px 0;
    box-shadow: 0 3px 10px rgba(91, 66, 26, 0.07);
}

.pipeline-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #5B4526;
    margin-bottom: 16px;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 9px 8px;
    background: #F6F1E3;
    border-radius: 10px;
}

.pipeline-number {
    min-width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #6E8F52;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 600;
}

.pipeline-step-title {
    color: #4A3410;
    font-size: 0.84rem;
    font-weight: 600;
}

.pipeline-step-desc {
    color: #7A6740;
    font-size: 0.70rem;
    margin-top: 2px;
    line-height: 1.3;
}

.pipeline-arrow {
    text-align: center;
    color: #A9642B;
    font-size: 1rem;
    line-height: 1;
    padding: 3px 0;
}

        /* Text area */
        .stTextArea textarea {
            border: 1.5px solid #C9B37F !important;
            border-radius: 12px !important;
            background-color: #FFFDF7 !important;
            font-size: 1rem !important;
        }

        /* Primary button */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(120deg, #A9642B 0%, #C89B3C 100%);
            border: none;
            border-radius: 999px;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            color: #FFF8E7;
            box-shadow: 0 4px 12px rgba(169, 100, 43, 0.35);
        }
        div.stButton > button[kind="primary"]:hover {
            filter: brightness(1.05);
        }

        /* Expander (result cards) */
        div[data-testid="stExpander"] {
            background: #FFFDF7;
            border: 1px solid #E4D6AE;
            border-radius: 14px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(91, 66, 26, 0.06);
        }
        div[data-testid="stExpander"] summary {
            font-family: 'Playfair Display', serif;
            font-size: 1.05rem;
            color: #5B4526;
        }

        .devanagari-text {
            font-family: 'Noto Sans Devanagari', sans-serif;
            font-size: 1.15rem;
            color: #4A3410;
            background: #FBF3E1;
            border-radius: 8px;
            padding: 10px 14px;
            border-left: 3px solid #C89B3C;
        }
        .iast-text {
            font-style: italic;
            color: #6E5A34;
            background: #F6EFDD;
            border-radius: 8px;
            padding: 8px 14px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #EFEADA 0%, #E7E0C9 100%);
        }

        /* Live status line (completion summary) */
        .ayur-status {
            color: #6E5A34;
            opacity: 0.9;
            font-size: 0.9rem;
            margin: 4px 0 14px 0;
        }

        /* Footer credit */
        .ayur-footer {
            text-align: center;
            margin-top: 36px;
            padding: 14px 0 6px 0;
            border-top: 1px solid #DCCFA3;
            color: #7A6740;
            font-size: 0.85rem;
        }
        .ayur-footer a {
            color: #7A4A1A;
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
            <p>Grounded in the {SOURCE_LABEL} &mdash; describe a patient's symptoms in
            English and retrieve the closest classical passages, verbatim.</p>
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


# st.dialog (native modal popups) was added in Streamlit 1.37. Feature-detect
# so this still degrades gracefully on older versions instead of crashing.
_HAS_DIALOG = hasattr(st, "dialog")

if _HAS_DIALOG:

    @st.dialog("📋 Results as JSON")
    def _show_json_popup(json_str: str) -> None:
        st.code(json_str, language="json")
        st.caption("Use the copy icon in the top-right corner of the box above.")
        st.download_button(
            "⬇️ Download JSON",
            data=json_str,
            file_name="ayurvedic_rag_results.json",
            mime="application/json",
            use_container_width=True,
        )

else:

    def _show_json_popup(json_str: str) -> None:  # pragma: no cover - old Streamlit fallback
        with st.expander("📋 Results as JSON", expanded=True):
            st.code(json_str, language="json")
            st.caption("Use the copy icon in the top-right corner of the box above.")


def render_export_bar(results: list[RetrievalResult], query: str) -> None:
    """Compact download / copy-as-JSON controls -- no raw JSON on the page itself."""
    json_str = results_to_json(results, query)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download as JSON",
            data=json_str,
            file_name="ayurvedic_rag_results.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        if st.button("📋 Copy as JSON", use_container_width=True):
            _show_json_popup(json_str)


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
    the whole request instead of two back-to-back widgets.
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
            """
            <div class="pipeline-card">
                <div class="pipeline-title">🔄 Retrieval Pipeline</div>
                <div class="pipeline-step">
                    <div class="pipeline-number">1</div>
                    <div>
                        <div class="pipeline-step-title">Patient Query</div>
                        <div class="pipeline-step-desc">
                            English symptoms / clinical complaint
                        </div>
                    </div>
                </div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">
                    <div class="pipeline-number">2</div>
                    <div>
                        <div class="pipeline-step-title">JinaColBERT-v2</div>
                        <div class="pipeline-step-desc">
                            Semantic retrieval
                        </div>
                    </div>
                </div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">
                    <div class="pipeline-number">3</div>
                    <div>
                        <div class="pipeline-step-title">MaxSim</div>
                        <div class="pipeline-step-desc">
                            Passage relevance scoring
                        </div>
                    </div>
                </div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">
                    <div class="pipeline-number">4</div>
                    <div>
                        <div class="pipeline-step-title">Ranked Verse IDs</div>
                        <div class="pipeline-step-desc">
                            Results ordered by relevance
                        </div>
                    </div>
                </div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">
                    <div class="pipeline-number">5</div>
                    <div>
                        <div class="pipeline-step-title">Corpus Lookup</div>
                        <div class="pipeline-step-desc">
                            Retrieve passages from Charaka Samhita
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
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

    # Run a new search only when the search button itself was just clicked.
    # Any OTHER widget (e.g. the "Copy as JSON" button) also triggers a full
    # script rerun, during which `search_clicked` is False again -- without
    # persisting results in session_state, that rerun would wipe this whole
    # section before the copy popup ever got a chance to open.
    if search_clicked:
        if not query or not query.strip():
            st.warning("Please enter a patient symptom description before searching.")
            st.session_state.pop("last_results", None)
        else:
            try:
                timed = run_with_live_timer(
                    _load_and_query,
                    query,
                    top_k,
                    label="Retrieving relevant knowledge from the Charaka Samhita...",
                )
            except PipelineError as e:
                st.session_state.pop("last_results", None)
                st.session_state["last_error"] = str(e)
            except Exception as e:  # noqa: BLE001 - last-resort guard for the UI
                st.session_state.pop("last_results", None)
                st.session_state["last_error"] = f"Unexpected error during retrieval: {e}"
            else:
                st.session_state.pop("last_error", None)
                st.session_state["last_results"] = timed.value
                st.session_state["last_query"] = query
                st.session_state["last_elapsed"] = timed.elapsed_seconds
                st.session_state["last_completed_at"] = timed.completed_at

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    results = st.session_state.get("last_results")
    if results:
        st.markdown(
            f'<div class="ayur-status">✅ Completed in '
            f'<b>{st.session_state["last_elapsed"]:0.1f}s</b> '
            f'&nbsp;·&nbsp; finished at <b>{st.session_state["last_completed_at"]}</b></div>',
            unsafe_allow_html=True,
        )

        st.subheader(f"🌱 Top {len(results)} results")

        render_export_bar(results, st.session_state.get("last_query", query))

        for result in results:
            render_result_card(result)

    render_footer()


if __name__ == "__main__":
    main()