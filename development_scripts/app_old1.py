
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
    layout="wide"
)

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "generator" not in st.session_state:
    st.session_state.generator = None


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🏦 Banking Guidelines Assistant")

st.markdown(
    """
    Ask questions about banking regulations, guidelines
    and financial documents using AI-powered retrieval.

    Answers are generated using your uploaded documents
    with document and page references.
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR: DOCUMENT MANAGEMENT
# --------------------------------------------------

st.sidebar.header("📚 Document Library")

uploaded_files = st.sidebar.file_uploader(
    "Upload banking PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button(
    "Add Uploaded Documents",
    use_container_width=True
):

    if not uploaded_files:

        st.sidebar.warning(
            "Please select at least one PDF."
        )

    else:

        saved_count = 0

        for uploaded_file in uploaded_files:

            file_path = (
                UPLOAD_FOLDER /
                Path(uploaded_file.name).name
            )

            file_path.write_bytes(
                uploaded_file.getvalue()
            )

            saved_count += 1

        st.session_state.knowledge_base_ready = False
        st.session_state.retriever = None
        st.session_state.generator = None
        st.session_state.messages = []

        st.sidebar.success(
            f"{saved_count} PDF document(s) saved successfully!"
        )


# --------------------------------------------------
# DISPLAY AVAILABLE DOCUMENTS
# --------------------------------------------------

pdf_files = sorted(
    UPLOAD_FOLDER.glob("*.pdf")
)

st.sidebar.subheader("Available Documents")

if pdf_files:

    for pdf_file in pdf_files:

        st.sidebar.write(
            f"📄 {pdf_file.name}"
        )

else:

    st.sidebar.info(
        "No PDF documents uploaded yet."
    )


# --------------------------------------------------
# BUILD / LOAD KNOWLEDGE BASE
# --------------------------------------------------

st.sidebar.divider()

if st.sidebar.button(
    "Build / Refresh Knowledge Base",
    use_container_width=True
):

    if not pdf_files:

        st.sidebar.error(
            "Upload at least one PDF first."
        )

    else:

        try:

            with st.spinner(
                "Checking documents and preparing knowledge base..."
            ):

                manager = KnowledgeBaseManager()

                (
                    vector_store,
                    embedding_model,
                    rebuilt
                ) = manager.get_or_build()

                retriever = Retriever(
                    embedding_model=embedding_model,
                    vector_store=vector_store
                )

                generator = AnswerGenerator()

                st.session_state.retriever = retriever
                st.session_state.generator = generator
                st.session_state.knowledge_base_ready = True
                st.session_state.messages = []

            if rebuilt:

                st.sidebar.success(
                    f"Knowledge base rebuilt! "
                    f"{vector_store.index.ntotal} chunks indexed."
                )

            else:

                st.sidebar.success(
                    f"Existing knowledge base loaded! "
                    f"{vector_store.index.ntotal} chunks available."
                )

        except Exception as error:

            st.session_state.knowledge_base_ready = False
            st.session_state.retriever = None
            st.session_state.generator = None

            st.sidebar.error(
                f"Knowledge base preparation failed: {error}"
            )


# --------------------------------------------------
# KNOWLEDGE BASE STATUS
# --------------------------------------------------

if st.session_state.knowledge_base_ready:

    st.success(
        "Knowledge base is ready. You can ask questions!"
    )

else:

    st.info(
        "Upload your PDFs and click "
        "'Build / Refresh Knowledge Base' "
        "before asking questions."
    )


# --------------------------------------------------
# DOCUMENT FILTER
# --------------------------------------------------

if st.session_state.knowledge_base_ready:

    document_options = [
        "All documents"
    ] + [
        pdf.name for pdf in pdf_files
    ]

    selected_document = st.selectbox(
        "Search within:",
        document_options
    )

    document_filter = (
        None
        if selected_document == "All documents"
        else selected_document
    )


    # --------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )

            if message["role"] == "assistant":

                sources = message.get(
                    "sources", []
                )

                if sources:

                    st.markdown(
                        "### Retrieved Evidence"
                    )

                    st.caption(
                        "These passages were retrieved from your "
                        "documents and provided to the AI. "
                        "Not every passage is necessarily cited "
                        "in the final answer."
                    )

                    for index, source in enumerate(
                        sources,
                        start=1
                    ):

                        with st.expander(
                            f"Evidence {index}: "
                            f"{source['document_name']} "
                            f"(Page {source['page_number']})"
                        ):

                            st.write(
                                source["text"]
                            )


    # --------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------

    question = st.chat_input(
        "Ask a question about your banking documents..."
    )

    if question:

        # Save and display the user's question.

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)


        # --------------------------------------------------
        # RETRIEVAL AND ANSWER GENERATION
        # --------------------------------------------------

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching documents and generating answer..."
                ):

                    # Step 1: Retrieve relevant document chunks.

                    retrieved_chunks = (
                        st.session_state.retriever.retrieve(
                            question=question,
                            top_k=5,
                            document_name=document_filter
                        )
                    )

                    # Step 2: Generate an answer using Gemini.

                    answer = (
                        st.session_state.generator.generate_answer(
                            question=question,
                            retrieved_chunks=retrieved_chunks
                        )
                    )


                # Step 3: Display the generated answer.
                
                st.markdown(answer)


                # --------------------------------------------------
                # DISPLAY RETRIEVED EVIDENCE
                # --------------------------------------------------

                if retrieved_chunks:

                    st.markdown(
                        "### Retrieved Evidence"
                    )

                    st.caption(
                        "These are the passages retrieved from "
                        "your documents and supplied to the AI. "
                        "They are candidate evidence and are not "
                        "necessarily all cited in the answer."
                    )

                    for index, source in enumerate(
                        retrieved_chunks,
                        start=1
                    ):

                        with st.expander(
                            f"Evidence {index}: "
                            f"{source['document_name']} "
                            f"(Page {source['page_number']})"
                        ):

                            st.write(
                                source["text"]
                            )


                # Step 4: Save the conversation.

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_chunks
                })


            except Exception as error:

                error_message = (
                    f"An error occurred: {error}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Banking Guidelines Assistant | "
    "Powered by Python, FAISS, Sentence Transformers and Gemini"
)
