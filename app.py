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
    page_icon="data/logo.png",
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
# Theming -- editorial / archival aesthetic: restrained ink-and-gold palette,
# hairline borders instead of shadows or glows, a serif/sans pairing (Lora +
# Inter) suited to a scholarly research tool rather than a consumer app.
# Colors read from Streamlit's own theme variables so light/dark mode both
# work correctly; only the hero band keeps a fixed brand color deliberately.
# --------------------------------------------------------------------------- #
def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600;700&family=Noto+Serif+Devanagari:wght@500;600&display=swap');

        :root {
            --ayur-ink: #22301D;
            --ayur-ink-soft: #37472F;
            --ayur-gold: #A98346;
            --ayur-hairline: rgba(140, 120, 90, 0.28);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, rgba(0, 0, 0, 0.02), transparent 220px),
                var(--background-color);
        }

        /* ---------- Hero ---------- */
        .ayur-hero {
            background: var(--ayur-ink);
            border-radius: 6px;
            padding: 38px 42px 28px 42px;
            margin-bottom: 26px;
            border-bottom: 3px solid var(--ayur-gold);
        }
        .ayur-kicker {
            font-size: 0.7rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--ayur-gold);
            font-weight: 600;
            margin-bottom: 10px;
        }
        .ayur-hero h1 {
            font-family: 'Lora', serif;
            font-weight: 700;
            color: #F6F1E6;
            font-size: 2.05rem;
            margin: 0 0 10px 0;
            letter-spacing: 0.2px;
        }
        .ayur-hero p {
            color: rgba(246, 241, 230, 0.78);
            font-size: 0.97rem;
            margin: 0 0 18px 0;
            max-width: 620px;
            line-height: 1.55;
        }
        .ayur-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 22px;
            padding-top: 14px;
            border-top: 1px solid rgba(246, 241, 230, 0.16);
        }
        .ayur-meta-item {
            font-size: 0.7rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: rgba(246, 241, 230, 0.55);
        }
        .ayur-meta-item b {
            color: #F6F1E6;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        /* ---------- Info note ---------- */
        .ayur-note {
            background: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid var(--ayur-hairline);
            border-left: 3px solid var(--ayur-gold);
            border-radius: 4px;
            padding: 12px 16px;
            font-size: 0.88rem;
            margin-bottom: 22px;
            opacity: 0.94;
        }
        .ayur-note b {
            color: var(--ayur-gold);
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: var(--secondary-background-color);
            border-right: 1px solid var(--ayur-hairline);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-family: 'Inter', sans-serif;
            font-size: 0.76rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-color);
            opacity: 0.65;
            font-weight: 700;
        }

        /* Retrieval pipeline -- vertical timeline, not colored badges */
        .pipeline-title {
            font-size: 0.76rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-color);
            opacity: 0.65;
            font-weight: 700;
            margin-bottom: 16px;
        }
        .pipeline-step {
            display: flex;
            gap: 12px;
            padding: 0 0 18px 0;
            position: relative;
        }
        .pipeline-step:not(:last-child)::before {
            content: "";
            position: absolute;
            left: 11px;
            top: 25px;
            bottom: -3px;
            width: 1px;
            background: var(--ayur-hairline);
        }
        .pipeline-number {
            min-width: 23px;
            height: 23px;
            border-radius: 2px;
            border: 1px solid var(--ayur-gold);
            color: var(--ayur-gold);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.66rem;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            background: var(--background-color);
            flex-shrink: 0;
        }
        .pipeline-step-title {
            color: var(--text-color);
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 1px;
        }
        .pipeline-step-desc {
            color: var(--text-color);
            opacity: 0.58;
            font-size: 0.72rem;
            margin-top: 2px;
            line-height: 1.4;
        }

        /* ---------- Text area ---------- */
        .stTextArea textarea {
            border: 1px solid var(--ayur-hairline) !important;
            border-radius: 4px !important;
            background-color: var(--secondary-background-color) !important;
            color: var(--text-color) !important;
            font-size: 0.96rem !important;
            transition: border-color 0.12s ease;
        }
        .stTextArea textarea:focus {
            border-color: var(--ayur-gold) !important;
            box-shadow: none !important;
        }

        /* ---------- Buttons ---------- */
        div.stButton > button[kind="primary"] {
            background: var(--ayur-ink);
            border: 1px solid var(--ayur-ink);
            border-radius: 4px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.02em;
            color: #F6F1E6;
            box-shadow: none;
            transition: background 0.12s ease, border-color 0.12s ease;
        }
        div.stButton > button[kind="primary"]:hover {
            background: var(--ayur-ink-soft);
            border-color: var(--ayur-gold);
        }

        div.stButton > button:not([kind="primary"]),
        div.stDownloadButton > button {
            background: transparent;
            color: var(--text-color);
            border: 1px solid var(--ayur-hairline);
            border-radius: 4px;
            font-weight: 500;
            font-size: 0.85rem;
            box-shadow: none;
            transition: border-color 0.12s ease, background 0.12s ease;
        }
        div.stButton > button:not([kind="primary"]):hover,
        div.stDownloadButton > button:hover {
            border-color: var(--ayur-gold);
            background: var(--secondary-background-color);
        }

        /* ---------- Result cards ---------- */
        div[data-testid="stExpander"] {
            background: var(--secondary-background-color);
            border: 1px solid var(--ayur-hairline);
            border-radius: 6px;
            margin-bottom: 10px;
            box-shadow: none;
        }
        div[data-testid="stExpander"] summary {
            font-family: 'Lora', serif;
            font-weight: 600;
            font-size: 1rem;
            color: var(--text-color);
        }

        .devanagari-text {
            font-family: 'Noto Serif Devanagari', serif;
            font-size: 1.15rem;
            color: var(--text-color);
            border-left: 2px solid var(--ayur-gold);
            padding: 4px 0 4px 16px;
            line-height: 1.8;
            background: transparent;
        }
        .iast-text {
            font-style: italic;
            color: var(--text-color);
            opacity: 0.72;
            padding: 2px 0 2px 16px;
            background: transparent;
            font-size: 0.92rem;
        }

        /* ---------- Result card: slim metadata list + content blocks ---------- */
        .ayur-meta-table {
            margin-bottom: 14px;
        }
        .ayur-meta-row {
            display: flex;
            align-items: baseline;
            gap: 16px;
            padding: 6px 0;
            border-bottom: 1px solid var(--ayur-hairline);
        }
        .ayur-meta-row:last-child {
            border-bottom: none;
        }
        .ayur-meta-row .label {
            flex: 0 0 128px;
            font-size: 0.66rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-color);
            opacity: 0.5;
        }
        .ayur-meta-row .value {
            flex: 1;
            min-width: 0;
            font-size: 0.85rem;
            color: var(--text-color);
            line-height: 1.4;
            word-break: break-word;
        }
        .ayur-meta-row .value a {
            color: var(--ayur-gold);
            text-decoration: none;
        }
        .ayur-meta-row .value a:hover {
            text-decoration: underline;
        }
        .ayur-block-label {
            font-size: 0.7rem;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: var(--text-color);
            opacity: 0.55;
            font-weight: 600;
            margin: 12px 0 6px 0;
        }
        .ayur-translation {
            color: var(--text-color);
            font-size: 0.94rem;
            line-height: 1.6;
        }

        /* ---------- Status + footer ---------- */
        .ayur-status {
            color: var(--text-color);
            opacity: 0.62;
            font-size: 0.82rem;
            margin: 2px 0 16px 0;
            letter-spacing: 0.01em;
        }

        .ayur-footer {
            text-align: center;
            margin-top: 40px;
            padding: 16px 0 4px 0;
            border-top: 1px solid var(--ayur-hairline);
            color: var(--text-color);
            opacity: 0.55;
            font-size: 0.78rem;
        }
        .ayur-footer a {
            color: var(--ayur-gold);
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
            <div class="ayur-kicker">Classical Text Retrieval</div>
            <h1>CharakaSetu</h1>
            <p>A retrieval interface over the {SOURCE_LABEL}. Describe a patient's
            symptoms in English to surface the closest classical passages,
            verbatim and unaltered.</p>
            <div class="ayur-meta">
                <div class="ayur-meta-item">Retriever&nbsp; <b>{RETRIEVER_LABEL}</b></div>
                <div class="ayur-meta-item">Source&nbsp; <b>{SOURCE_LABEL}</b></div>
            </div>
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
        source_value = (
            f'<a href="{result.source_url}" target="_blank">'
            f'{truncate(result.source_url, 55)}</a>'
            if result.source_url
            else "-"
        )

        # Slim metadata list -- one row per short/identifying corpus field,
        # labeled with the corpus's own column names (or the closest plain-
        # English equivalent) so nothing is ambiguous or renamed unclearly.
        st.markdown(
            f"""
            <div class="ayur-meta-table">
                <div class="ayur-meta-row">
                    <span class="label">Verse ID</span>
                    <span class="value">{result.verse_id or "-"}</span>
                </div>
                <div class="ayur-meta-row">
                    <span class="label">Chapter Title</span>
                    <span class="value">{result.chapter_title or "-"}</span>
                </div>
                <div class="ayur-meta-row">
                    <span class="label">Sthana</span>
                    <span class="value">{result.sthana or "-"}</span>
                </div>
                <div class="ayur-meta-row">
                    <span class="label">Heading Path</span>
                    <span class="value">{result.heading_path or "-"}</span>
                </div>
                <div class="ayur-meta-row">
                    <span class="label">Verse Numbers</span>
                    <span class="value">{format_verse_numbers(result.verse_numbers_list)}</span>
                </div>
                <div class="ayur-meta-row">
                    <span class="label">Source URL</span>
                    <span class="value">{source_value}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Long-form content fields keep their own labeled block rather than
        # a table row, since Sanskrit/IAST/translation text needs room to
        # breathe and doesn't fit a single line.
        st.markdown('<div class="ayur-block-label">Devanagari</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="devanagari-text">{result.devanagari or "-"}</div>',
            unsafe_allow_html=True,
        )

        if result.iast:
            st.markdown('<div class="ayur-block-label">IAST</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="iast-text">{result.iast}</div>', unsafe_allow_html=True
            )

        st.markdown('<div class="ayur-block-label">Translation</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ayur-translation">{result.translation or "-"}</div>',
            unsafe_allow_html=True,
        )


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

    @st.dialog("Results as JSON")
    def _show_json_popup(json_str: str) -> None:
        st.code(json_str, language="json")
        st.caption("Use the copy icon in the top-right corner of the box above.")
        st.download_button(
            "Download JSON",
            data=json_str,
            file_name="ayurvedic_rag_results.json",
            mime="application/json",
            width=True,
        )

else:

    def _show_json_popup(json_str: str) -> None:  # pragma: no cover - old Streamlit fallback
        with st.expander("Results as JSON", expanded=True):
            st.code(json_str, language="json")
            st.caption("Use the copy icon in the top-right corner of the box above.")


def render_export_bar(results: list[RetrievalResult], query: str) -> None:
    """Compact download / copy-as-JSON controls -- no raw JSON on the page itself."""
    json_str = results_to_json(results, query)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download as JSON",
            data=json_str,
            file_name="ayurvedic_rag_results.json",
            mime="application/json",
            width='stretch',
        )
    with col2:
        if st.button("Copy as JSON", width='stretch'):
            _show_json_popup(json_str)


def render_sample_picker() -> None:
    samples = load_sample_queries()
    if not samples:
        return

    options = ["Select a sample case..."] + [sample_label(s) for s in samples]

    def _apply_sample() -> None:
        idx = st.session_state.get("_sample_choice_idx", 0)
        if idx > 0:
            st.session_state[QUERY_INPUT_KEY] = samples[idx - 1]["query"]

    st.selectbox(
        "Try a sample case",
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
        '<div class="ayur-note"><b>Note</b> &mdash; This tool retrieves relevant '
        "passages only. It does not generate medical claims, diagnoses, or "
        "treatment recommendations.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image("data/logo.png", width='stretch')
        st.header("Settings")
        top_k = st.slider("Top-K results", min_value=1, max_value=50, value=10, step=1)
        st.divider()
        st.markdown(
            """
            <div class="pipeline-title">Retrieval Pipeline</div>
            <div class="pipeline-step">
                <div class="pipeline-number">01</div>
                <div>
                    <div class="pipeline-step-title">Patient Query</div>
                    <div class="pipeline-step-desc">
                        English symptoms / clinical complaint
                    </div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-number">02</div>
                <div>
                    <div class="pipeline-step-title">JinaColBERT-v2</div>
                    <div class="pipeline-step-desc">
                        Semantic retrieval
                    </div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-number">03</div>
                <div>
                    <div class="pipeline-step-title">MaxSim</div>
                    <div class="pipeline-step-desc">
                        Passage relevance scoring
                    </div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-number">04</div>
                <div>
                    <div class="pipeline-step-title">Ranked Verse IDs</div>
                    <div class="pipeline-step-desc">
                        Results ordered by relevance
                    </div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-number">05</div>
                <div>
                    <div class="pipeline-step-title">Corpus Lookup</div>
                    <div class="pipeline-step-desc">
                        Retrieve passages from Charaka Samhita
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

    search_clicked = st.button("Retrieve Relevant Knowledge", type="primary")

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
            f'<div class="ayur-status">Completed in '
            f'<b>{st.session_state["last_elapsed"]:0.1f}s</b> '
            f'&nbsp;·&nbsp; finished at <b>{st.session_state["last_completed_at"]}</b></div>',
            unsafe_allow_html=True,
        )

        st.subheader(f"Top {len(results)} Results")

        render_export_bar(results, st.session_state.get("last_query", query))

        for result in results:
            render_result_card(result)

    render_footer()


if __name__ == "__main__":
    main()