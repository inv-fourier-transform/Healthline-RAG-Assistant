"""
Healthline RAG Assistant - Streamlit UI Interface
A RAG-based healthcare chatbot interface for querying Healthline articles.
"""

import re
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Path Setup
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.indexer import build_index
from core.qa import answer_query

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Configuration
# ──────────────────────────────────────────────────────────────────────────────

ALLOWED_URL_PREFIXES = (
    "https://www.healthline.com",
    "www.healthline.com",
    "healthline.com",
)

MAX_URL_SLOTS = 10

THEME_CSS = """
<style>
:root {
  --bg-primary: #0e1117;
  --bg-panel-1: #111827;
  --bg-panel-2: #0f172a;
  --bg-response: #1f2937;
  --border-1: #1f2937;
  --border-2: #243041;
  --text-primary: #e5e7eb;
  --text-secondary: #cbd5e1;
  --accent-blue: #3b82f6;
  --accent-blue-hover: #2563eb;
  --accent-green: #10b981;
  --accent-green-hover: #059669;
}

html, body, [class^="css"] {
  font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  color: var(--text-primary);
}

[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 60% -20%, #121621 10%, var(--bg-primary) 60%);
}

#app-title {
  font-size: 2.2rem;
  font-weight: 800;
  color: #f8fafc;
  text-align: center;
  margin: 0 0 1rem 0;
  letter-spacing: 0.3px;
}

#app-title .icon { margin-right: 0.6rem; }

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) {
  background: var(--bg-panel-1);
  border: 1px solid var(--border-1);
  border-radius: 14px;
  padding: 18px 16px;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) {
  background: var(--bg-panel-2);
  border: 1px solid var(--border-2);
  border-radius: 14px;
  padding: 18px 16px;
}

.section-label {
  font-size: 0.9rem;
  color: var(--text-secondary);
  margin-bottom: 0.5rem;
  letter-spacing: 0.2px;
  font-weight: 600;
}

input[type="text"], textarea {
  background: #0b1220 !important;
  color: var(--text-primary) !important;
  border: 1px solid #223049 !important;
  border-radius: 10px !important;
}

input[type="text"]::placeholder, textarea::placeholder { color: #94a3b8 !important; }

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) div.stButton > button {
  background: var(--accent-blue);
  color: #ffffff;
  border-radius: 10px;
  border: 1px solid var(--accent-blue-hover);
  padding: 0.6rem 1rem;
  font-weight: 700;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) div.stButton > button:hover {
  background: var(--accent-blue-hover);
}

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) div.stButton > button {
  background: var(--accent-green);
  color: #ffffff;
  border-radius: 10px;
  border: 1px solid var(--accent-green-hover);
  padding: 0.6rem 1rem;
  font-weight: 700;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) div.stButton > button:hover {
  background: var(--accent-green-hover);
}

button[disabled] {
  background: #2f3542 !important;
  color: #9aa3af !important;
  border-color: #3a4150 !important;
}

.response-area {
  background: var(--bg-response);
  border: 1px solid #334155;
  border-radius: 14px;
  padding: 16px;
  margin-top: 14px;
  color: var(--text-primary);
}

.response-title {
  font-weight: 800;
  margin-bottom: 8px;
  color: #dbeafe;
}

.disclaimer {
  font-size: 0.8rem;
  color: #9ca3af;
  text-align: center;
  margin-top: 26px;
}
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialize all session state variables with default values."""
    defaults = {
        "urls_ok": False,
        "validated_urls": [],
        "index_ready": False,
        "last_query": "",
        "show_response": False,
        "last_answer": "",
        "last_sources": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Initialize URL input slots
    for i in range(MAX_URL_SLOTS):
        key = f"url_{i}"
        if key not in st.session_state:
            st.session_state[key] = ""


# ──────────────────────────────────────────────────────────────────────────────
# URL Validation Helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_valid_healthline_prefix(url: str) -> bool:
    """Check if URL starts with an allowed Healthline prefix."""
    if not isinstance(url, str):
        return False
    return any(url.strip().startswith(prefix) for prefix in ALLOWED_URL_PREFIXES)


def canonicalize_healthline_url(url: str) -> str | None:
    """Normalize Healthline URL to canonical https://www.healthline.com format."""
    if not isinstance(url, str) or not is_valid_healthline_prefix(url):
        return None

    url = url.strip()

    # Normalize to https://www.healthline.com prefix
    if url.startswith("https://www.healthline.com"):
        fixed = url
    elif url.startswith("www.healthline.com"):
        fixed = "https://" + url
    elif url.startswith("healthline.com"):
        fixed = "https://www." + url
    else:
        return None

    # Parse and rebuild clean URL
    parsed = urlparse(fixed)
    path = parsed.path or "/"
    path = re.sub(r"/{2,}", "/", path).strip()

    if not path.startswith("/"):
        path = "/" + path
    path = path.lower()

    if path != "/" and path.endswith("/"):
        path = path[:-1]

    return urlunparse(("https", "www.healthline.com", path, "", "", ""))


# ──────────────────────────────────────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────────────────────────────────────

def render_header() -> None:
    """Render the application title and styling."""
    st.set_page_config(page_title="Healthline RAG Assistant", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div id="app-title"><span class="icon">🩺</span>Healthline RAG Assistant</div>',
        unsafe_allow_html=True,
    )


def render_url_input_panel() -> None:
    """Render the left panel for URL input and indexing."""
    st.markdown('<div class="section-label">Enter up to 10 Healthline URLs (one per line)</div>',
                unsafe_allow_html=True)

    # URL input fields
    for i in range(MAX_URL_SLOTS):
        st.text_input(
            label="URL",
            key=f"url_{i}",
            placeholder="Paste a Healthline URL (e.g., https://www.healthline.com/health/...)",
            label_visibility="collapsed",
        )

    # Validate and submit button
    if st.button("Validate & Submit URLs"):
        process_url_submission()


def process_url_submission() -> None:
    """Handle URL validation, deduplication, and indexing."""
    # Collect non-empty URLs
    raw_urls = [
        st.session_state.get(f"url_{i}", "").strip()
        for i in range(MAX_URL_SLOTS)
        if st.session_state.get(f"url_{i}", "").strip()
    ]

    if not raw_urls:
        reset_index_state()
        st.error("No URLs uploaded by the user.")
        return

    # Validate and canonicalize
    invalid_urls = [u for u in raw_urls if not is_valid_healthline_prefix(u)]
    canonical_urls = []
    seen_urls = set()
    duplicates = []

    for url in raw_urls:
        canonical = canonicalize_healthline_url(url)
        if canonical is None:
            if url not in invalid_urls:
                invalid_urls.append(url)
        elif canonical in seen_urls:
            duplicates.append(url)
        else:
            seen_urls.add(canonical)
            canonical_urls.append(canonical)

    # Handle validation errors
    if invalid_urls:
        reset_index_state()
        st.error(
            "Invalid URL format detected. A valid URL must start with one of: "
            "https://www.healthline.com, www.healthline.com, or healthline.com."
        )
        for bad_url in invalid_urls:
            st.warning(f"Rejected: {bad_url}")
        return

    if duplicates:
        reset_index_state()
        st.error("Duplicate URLs detected across different formats. Please remove duplicates.")
        for dup_url in duplicates:
            st.warning(f"Duplicate: {dup_url}")
        return

    # Build index
    st.session_state.validated_urls = canonical_urls[:MAX_URL_SLOTS]

    with st.spinner("Indexing vectorstore (clearing old embeddings and creating new ones)..."):
        result = build_index(st.session_state.validated_urls)

    if result.get("status") == "ok":
        st.session_state.urls_ok = True
        st.session_state.index_ready = True
        st.success(f"Indexed {result.get('chunks_indexed', 0)} chunks.")
    else:
        reset_index_state()
        st.error("Failed to index content.")
        for error in result.get("errors", []):
            st.warning(error)


def reset_index_state() -> None:
    """Reset all index-related session state flags."""
    st.session_state.urls_ok = False
    st.session_state.index_ready = False
    st.session_state.show_response = False


def render_query_panel() -> None:
    """Render the right panel for query input and response display."""
    query = st.text_area("Query", placeholder="Enter your query", height=220)

    # Determine if query can be submitted
    can_submit = (
            st.session_state.get("urls_ok")
            and st.session_state.get("index_ready")
            and query.strip()
    )

    if st.button("Submit query", disabled=not can_submit):
        process_query(query.strip())

    # Display response if available
    if st.session_state.get("show_response"):
        render_response()


def process_query(query: str) -> None:
    """Execute query and store results in session state."""
    st.session_state.last_query = query

    with st.spinner("Retrieving and generating response..."):
        response = answer_query(st.session_state.last_query)

    st.session_state.last_answer = response.get("answer", "")
    st.session_state.last_sources = response.get("sources", [])
    st.session_state.show_response = True


def render_response() -> None:
    """Render the answer and sources in the response area."""
    st.markdown(
        '<div class="response-area"><div class="response-title">Response</div>',
        unsafe_allow_html=True,
    )

    # Answer text
    if st.session_state.last_answer:
        st.markdown(st.session_state.last_answer)
    else:
        st.markdown("_No answer generated from the provided sources._")

    # Sources list
    if st.session_state.last_sources:
        st.markdown('<div class="response-title">Sources</div>', unsafe_allow_html=True)
        for source in st.session_state.last_sources:
            st.markdown(f"- {source}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_footer() -> None:
    """Render the copyright disclaimer."""
    st.markdown(
        '<div class="disclaimer">"All rights to the content in the provided URLs belong solely to Healthline Media LLC."</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main application entry point."""
    render_header()
    init_session_state()

    # Two-column layout
    left_column, right_column = st.columns([1.2, 2.0])

    with left_column:
        render_url_input_panel()

    with right_column:
        render_query_panel()

    render_footer()


if __name__ == "__main__":
    main()
