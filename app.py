import streamlit as st
from pathlib import Path

from rag.knowledge_base_manager import KnowledgeBaseManager
from rag.retriever import Retriever
from rag.generator import AnswerGenerator


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Banking Guidelines Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

INSUFFICIENT_INFO = (
    "I could not find sufficient information to answer this question "
    "in the selected document(s)."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

DEFAULT_STATE = {
    "knowledge_base_ready": False,
    "messages": [],
    "retriever": None,
    "generator": None,
    "pending_question": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --navy: #101B2D;
        --navy-soft: #17263D;
        --blue: #4776E6;
        --cyan: #63C7D8;
        --ink: #182437;
        --muted: #78869A;
        --surface: #FFFFFF;
        --line: #E6EBF2;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 82% 0%, rgba(71,118,230,0.08), transparent 28%),
            #F5F7FB;
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101B2D 0%, #17263D 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #EAF0FA;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.07);
        border: 1px dashed rgba(255,255,255,0.28);
        border-radius: 14px;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        color: #B8C5D8 !important;
    }

    [data-testid="stSidebar"] .stButton button {
        background: #4776E6;
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background: #5A86F0;
        color: white;
        border: 0;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.035em;
        color: #17263D;
    }

    .hero {
        background: linear-gradient(115deg, #101B2D 0%, #1D3557 65%, #294C78 100%);
        border-radius: 22px;
        padding: 30px 34px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 14px 35px rgba(16,27,45,0.13);
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 50%;
        right: -65px;
        top: -115px;
        box-shadow: 0 0 0 28px rgba(255,255,255,0.025),
                    0 0 0 58px rgba(255,255,255,0.02);
    }

    .hero-kicker {
        color: #9EDCE5;
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        margin-bottom: 10px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 3vw, 2.55rem);
        line-height: 1.16;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 10px 0;
    }

    .hero-copy {
        color: #D5DFED;
        max-width: 720px;
        font-size: 0.98rem;
        margin: 0;
        line-height: 1.65;
    }

    .section-label {
        color: #77869A;
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin: 8px 0 12px 0;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E7ECF3;
        border-radius: 15px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(26,45,75,0.035);
        min-height: 92px;
    }

    .metric-label {
        color: #7C899B;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #17263D;
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-note {
        color: #91A0B2;
        font-size: 0.72rem;
        margin-top: 5px;
    }

    .welcome-card {
        background: #FFFFFF;
        border: 1px solid #E7ECF3;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 8px 26px rgba(26,45,75,0.045);
        margin: 18px 0;
    }

    .welcome-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.35rem;
        color: #17263D;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .welcome-copy {
        color: #7C899B;
        line-height: 1.6;
        margin-bottom: 18px;
    }

    .source-intro {
        color: #7C899B;
        font-size: 0.82rem;
        margin: 4px 0 10px 0;
    }

    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #E7ECF3;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 5px 18px rgba(26,45,75,0.035);
        margin-bottom: 14px;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #EEF3FF;
        border-color: #DCE6FF;
    }

    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E7ECF3;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        font-weight: 600;
        color: #26364D;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 8px 0 22px 0;
        border-bottom: 1px solid rgba(255,255,255,0.13);
        margin-bottom: 20px;
    }

    .sidebar-icon {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 13px;
        background: linear-gradient(135deg, #5B8AF0, #62C6D7);
        font-size: 1.35rem;
    }

    .sidebar-brand-name {
        font-family: 'Manrope', sans-serif;
        font-size: 0.98rem;
        font-weight: 800;
        color: white;
        line-height: 1.2;
    }

    .sidebar-brand-caption {
        color: #9EADC2;
        font-size: 0.72rem;
        margin-top: 3px;
    }

    .sidebar-heading {
        color: #A9B9CF;
        text-transform: uppercase;
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        margin: 17px 0 9px 0;
    }

    .document-item {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 10px;
        padding: 9px 10px;
        margin: 6px 0;
        color: #E6EDF7;
        font-size: 0.78rem;
        overflow-wrap: anywhere;
    }

    .status-pill {
        display: inline-block;
        border-radius: 30px;
        padding: 5px 10px;
        font-size: 0.72rem;
        font-weight: 700;
        background: rgba(75,190,145,0.14);
        color: #19845D;
        border: 1px solid rgba(75,190,145,0.22);
    }

    .footer {
        color: #97A3B3;
        font-size: 0.75rem;
        text-align: center;
        padding: 22px 0 5px 0;
    }

    div[data-testid="stChatInput"] {
        border-radius: 16px;
    }

    .stButton button {
        border-radius: 11px;
        font-weight: 600;
    }

    @media (max-width: 768px) {
        .block-container { padding-top: 1rem; }
        .hero { padding: 23px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def render_source_evidence(sources, key_prefix):
    """Render retrieved passages in compact expandable evidence panels."""
    if not sources:
        return

    st.markdown("**Sources & retrieved evidence**")
    st.markdown(
        '<div class="source-intro">Open a source to inspect the exact passage '
        'provided to the AI. Page numbers refer to the PDF page.</div>',
        unsafe_allow_html=True,
    )

    for index, source in enumerate(sources, start=1):
        document_name = source.get("document_name", "Unknown document")
        page_number = source.get("page_number", "—")
        with st.expander(
            f"📄  {document_name}  ·  Page {page_number}",
            expanded=False,
        ):
            st.caption(f"Retrieved passage {index}")
            st.write(source.get("text", "No passage text available."))


def reset_ai_session():
    """Clear in-memory AI objects after the document set changes."""
    st.session_state.knowledge_base_ready = False
    st.session_state.retriever = None
    st.session_state.generator = None
    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR: DOCUMENT WORKSPACE
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-icon">🏦</div>
            <div>
                <div class="sidebar-brand-name">Banking Intelligence</div>
                <div class="sidebar-brand-caption">Document AI workspace</div>
            </div>
        </div>
        <div class="sidebar-heading">Document workspace</div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Add your banking PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF documents at once.",
    )

    if st.button("＋  Add documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one PDF first.")
        else:
            saved_count = 0
            for uploaded_file in uploaded_files:
                file_path = UPLOAD_FOLDER / Path(uploaded_file.name).name
                file_path.write_bytes(uploaded_file.getvalue())
                saved_count += 1

            reset_ai_session()
            st.success(f"{saved_count} document(s) saved.")
            st.rerun()

    pdf_files = sorted(UPLOAD_FOLDER.glob("*.pdf"))

    st.markdown(
        f'<div class="sidebar-heading">Library · {len(pdf_files)} PDF(s)</div>',
        unsafe_allow_html=True,
    )

    if pdf_files:
        for pdf_file in pdf_files:
            st.markdown(
                f'<div class="document-item">📄 &nbsp;{pdf_file.name}</div>',
                unsafe_allow_html=True,
            )

        # Delete one uploaded PDF at a time. A confirmation checkbox
        # helps prevent accidental removal from the document library.
        st.markdown('<div class="sidebar-heading">Manage documents</div>', unsafe_allow_html=True)

        delete_options = [pdf.name for pdf in pdf_files]
        selected_delete_file = st.selectbox(
            "Select a PDF to remove",
            delete_options,
            key="delete_document_selection",
        )
        confirm_delete = st.checkbox(
            "I understand this will remove the PDF from the uploads folder.",
            key="confirm_document_delete",
        )

        if st.button(
            "🗑️  Delete selected PDF",
            use_container_width=True,
            disabled=not confirm_delete,
            type="secondary",
        ):
            target_path = UPLOAD_FOLDER / selected_delete_file

            try:
                # Resolve the path and ensure it remains inside uploads.
                safe_target = target_path.resolve()
                safe_upload_folder = UPLOAD_FOLDER.resolve()

                if safe_target.parent != safe_upload_folder:
                    st.error("Invalid document path. No file was deleted.")
                elif not safe_target.is_file() or safe_target.suffix.lower() != ".pdf":
                    st.error("The selected PDF could not be found.")
                else:
                    safe_target.unlink()
                    reset_ai_session()
                    st.session_state.pop("delete_document_selection", None)
                    st.session_state.pop("confirm_document_delete", None)
                    st.success(f"Removed {selected_delete_file} from the document library.")
                    st.info("Your saved index may still contain the old document until you click Build / refresh index.")
                    st.rerun()

            except OSError as error:
                st.error(f"Could not delete the selected PDF: {error}")

    else:
        st.caption("Your uploaded documents will appear here.")

    st.markdown('<div class="sidebar-heading">Knowledge base</div>', unsafe_allow_html=True)

    if st.button("↻  Build / refresh index", use_container_width=True):
        if not pdf_files:
            st.error("Upload at least one PDF first.")
        else:
            try:
                with st.spinner("Preparing your document knowledge base..."):
                    manager = KnowledgeBaseManager()
                    vector_store, embedding_model, rebuilt = manager.get_or_build()

                    retriever = Retriever(
                        embedding_model=embedding_model,
                        vector_store=vector_store,
                    )
                    generator = AnswerGenerator()

                    st.session_state.retriever = retriever
                    st.session_state.generator = generator
                    st.session_state.knowledge_base_ready = True
                    st.session_state.messages = []

                if rebuilt:
                    st.success(f"Index refreshed · {vector_store.index.ntotal:,} chunks")
                else:
                    st.success(f"Knowledge base loaded · {vector_store.index.ntotal:,} chunks")
                st.rerun()

            except Exception as error:
                reset_ai_session()
                st.error(f"Could not prepare the knowledge base: {error}")

    if st.session_state.knowledge_base_ready:
        st.markdown(
            '<div class="status-pill">●  Knowledge base ready</div>',
            unsafe_allow_html=True,
        )
        total_chunks = st.session_state.retriever.vector_store.index.ntotal
        st.caption(f"{total_chunks:,} indexed text chunks")
    else:
        st.caption("Upload PDFs, then build or refresh the index.")

    st.divider()
    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        <div style="color:#91A1B7;font-size:0.7rem;line-height:1.5;margin-top:18px;">
            AI-powered document retrieval<br>
            Python · FAISS · Sentence Transformers · Gemini
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">AI-powered document intelligence</div>
        <div class="hero-title">Banking Guidelines Assistant</div>
        <p class="hero-copy">
            Ask questions across banking regulations, policy documents and financial reports.
            Get evidence-grounded answers with document names and page-level references.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# WORKSPACE METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">PDF documents</div>
            <div class="metric-value">{len(pdf_files):,}</div>
            <div class="metric-note">Available in your library</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    chunk_count = (
        st.session_state.retriever.vector_store.index.ntotal
        if st.session_state.knowledge_base_ready
        else 0
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Indexed passages</div>
            <div class="metric-value">{chunk_count:,}</div>
            <div class="metric-note">Searchable text chunks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    status_text = "Ready" if st.session_state.knowledge_base_ready else "Not indexed"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Assistant status</div>
            <div class="metric-value" style="font-size:1.3rem;">{status_text}</div>
            <div class="metric-note">Evidence-grounded question answering</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# --------------------------------------------------
# KNOWLEDGE BASE GUARD
# --------------------------------------------------

if not st.session_state.knowledge_base_ready:
    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-title">Your document workspace is ready</div>
            <div class="welcome-copy">
                Add one or more PDF documents in the sidebar, then build the knowledge base.
                The assistant will extract the text, split it into searchable passages and
                prepare it for question answering.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Start by uploading your PDFs in the left sidebar and selecting **Build / refresh index**.")

else:
    # --------------------------------------------------
    # SEARCH SCOPE
    # --------------------------------------------------

    document_options = ["All documents"] + [pdf.name for pdf in pdf_files]

    filter_col, hint_col = st.columns([1.2, 2.8])
    with filter_col:
        selected_document = st.selectbox(
            "Search scope",
            document_options,
            help="Choose whether to search every indexed PDF or just one document.",
        )

    document_filter = None if selected_document == "All documents" else selected_document

    with hint_col:
        st.markdown(
            """
            <div style="padding-top:29px;color:#7C899B;font-size:0.83rem;">
                Searching across all documents can help answer comparison questions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # WELCOME / SUGGESTED QUESTIONS
    # --------------------------------------------------

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-title">What would you like to understand?</div>
                <div class="welcome-copy">
                    Ask a question in natural language. The assistant searches your selected
                    documents and uses retrieved passages to prepare its answer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        suggestions = [
            "When do the RBI Directions come into force?",
            "Compare the implementation dates of RBI and Jordan.",
            "What is the global speculative-grade default rate forecast?",
            "What is the minimum CAR specified by RBI?",
        ]

        suggestion_cols = st.columns(2)
        for index, suggestion in enumerate(suggestions):
            with suggestion_cols[index % 2]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{index}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    # --------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_source_evidence(
                    message.get("sources", []),
                    key_prefix=f"history_{message_index}",
                )

    # --------------------------------------------------
    # CHAT INPUT AND ANSWER GENERATION
    # --------------------------------------------------

    typed_question = st.chat_input(
        "Ask about a regulation, capital requirement, date or financial report..."
    )

    question = typed_question or st.session_state.pending_question
    if question:
        st.session_state.pending_question = None

        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents and preparing an evidence-based answer..."):
                    retrieved_chunks = st.session_state.retriever.retrieve(
                        question=question,
                        top_k=5,
                        document_name=document_filter,
                    )

                    answer = st.session_state.generator.generate_answer(
                        question=question,
                        retrieved_chunks=retrieved_chunks,
                    )

                st.markdown(answer)
                render_source_evidence(retrieved_chunks, key_prefix="latest")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_chunks,
                })

            except Exception as error:
                error_message = f"An error occurred while processing your question: {error}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Banking Guidelines Assistant · Built with Python, FAISS, Sentence Transformers and Gemini
        <br>Answers are generated from the selected documents. Always verify important regulatory decisions against the source.
    </div>
    """,
    unsafe_allow_html=True,
)
