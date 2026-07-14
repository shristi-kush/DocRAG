from __future__ import annotations

import html

import streamlit as st
from werkzeug.utils import secure_filename

from src.config import DATA_RAW_DIR
from src.llm import init_llm
from src.rag import DocumentNotLoadedError, process_document, process_prompt

st.set_page_config(
    page_title="DocRAG",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 20%, rgba(196, 181, 253, 0.35) 0%, transparent 55%),
        radial-gradient(ellipse 70% 45% at 80% 70%, rgba(167, 139, 250, 0.3) 0%, transparent 50%),
        linear-gradient(160deg, #5b21b6 0%, #7c3aed 40%, #6d28d9 70%, #4c1d95 100%);
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"],
#MainMenu,
footer {
    visibility: hidden;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 760px;
}

.brand-row {
    color: #fff;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
    margin-bottom: 1.75rem;
}

.brand-row span {
    opacity: 0.85;
    font-weight: 400;
}

.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.18);
    color: #fff;
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

.hero-title {
    color: #fff !important;
    font-size: 2.35rem;
    font-weight: 700;
    line-height: 1.2;
    text-align: center;
    margin: 0 0 0.75rem 0;
}

.hero-sub {
    color: rgba(255, 255, 255, 0.88);
    text-align: center;
    font-size: 1.05rem;
    line-height: 1.5;
    max-width: 520px;
    margin: 0 auto 1.75rem auto;
}

div[data-testid="stVerticalBlock"] > div:has(.card-marker) {
    background: #ffffff;
    border-radius: 20px;
    padding: 1.35rem 1.4rem 1.1rem 1.4rem;
    box-shadow: 0 20px 50px rgba(46, 16, 101, 0.28);
    margin-bottom: 1.25rem;
}

.card-marker { display: none; }

.answer-shell {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 12px 30px rgba(46, 16, 101, 0.2);
    color: #1f1635;
    margin-bottom: 1rem;
    white-space: pre-wrap;
}

.answer-shell h4 {
    margin: 0 0 0.5rem 0;
    color: #5b21b6;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.footer-note {
    text-align: center;
    color: rgba(255, 255, 255, 0.7);
    font-size: 0.8rem;
    margin-top: 1.5rem;
}

div[data-testid="stTextArea"] textarea {
    background: #f3f4f6 !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 140px !important;
    color: #111827 !important;
}

div[data-testid="stFileUploader"] section {
    background: #fafafa;
    border: 1.5px dashed #c4b5fd !important;
    border-radius: 12px;
    padding: 0.75rem;
}

div[data-testid="stFileUploader"] label {
    color: #6b7280 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.6rem !important;
    box-shadow: 0 8px 18px rgba(109, 40, 217, 0.35);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
"""


@st.cache_resource
def bootstrap_llm():
    return init_llm()


def ensure_state() -> None:
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "last_filename" not in st.session_state:
        st.session_state.last_filename = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "ingest_note" not in st.session_state:
        st.session_state.ingest_note = None


def save_and_ingest(uploaded_file) -> str:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(uploaded_file.name)
    dest = DATA_RAW_DIR / filename
    dest.write_bytes(uploaded_file.getvalue())
    process_document(dest)
    return filename


def main() -> None:
    bootstrap_llm()
    ensure_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="brand-row">DocRAG <span>by Shristi</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="text-align:center">'
        '<span class="hero-badge">RAG Chat: Fast &amp; Private</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 class="hero-title">Ask anything about your documents</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">Upload a PDF, then ask in natural language. '
        "Answers stay grounded in your document.</p>",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<span class="card-marker"></span>', unsafe_allow_html=True)
        question = st.text_area(
            "Question",
            placeholder="Ask a question about your document...",
            label_visibility="collapsed",
            height=140,
            key="question_input",
        )

        uploaded = st.file_uploader(
            "Upload or drop files here (pdf)",
            type=["pdf"],
            label_visibility="visible",
            key="pdf_uploader",
        )

        _, right = st.columns([3, 1])
        with right:
            ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    if ask_clicked:
        st.session_state.last_error = None
        st.session_state.ingest_note = None

        try:
            if uploaded is not None:
                filename = save_and_ingest(uploaded)
                st.session_state.last_filename = filename
                st.session_state.ingest_note = f"Indexed {filename}"

            if not question or not question.strip():
                st.session_state.last_error = "Enter a question before asking."
                st.session_state.last_answer = None
            else:
                with st.spinner("Generating answer..."):
                    answer = process_prompt(question)
                st.session_state.last_answer = answer
        except DocumentNotLoadedError as exc:
            st.session_state.last_error = str(exc)
            st.session_state.last_answer = None
        except Exception as exc:  # noqa: BLE001
            st.session_state.last_error = str(exc)
            st.session_state.last_answer = None

    if st.session_state.ingest_note:
        st.success(st.session_state.ingest_note)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if st.session_state.last_answer:
        safe = html.escape(st.session_state.last_answer)
        st.markdown(
            f'<div class="answer-shell"><h4>Answer</h4><p>{safe}</p></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.last_filename:
        st.caption(f"Active document: {st.session_state.last_filename}")

    st.markdown(
        '<p class="footer-note">Answers depend on uploaded content and may be incomplete.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
