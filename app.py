from pathlib import Path
import streamlit as st

from src.rag_pipeline import (
    RAGPipeline,
    DATA_RAW,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
)

st.set_page_config(page_title="Lecture RAG Bot", page_icon="📘", layout="wide")

# ---------- Utility Functions ----------

def save_uploaded_files(uploaded_files):
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = DATA_RAW / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path.name)

    return saved_files


def clear_old_processed_data():
    if CHUNKS_PATH.exists():
        CHUNKS_PATH.unlink()
    if EMBEDDINGS_PATH.exists():
        EMBEDDINGS_PATH.unlink()


def clear_raw_pdfs():
    if DATA_RAW.exists():
        for pdf_file in DATA_RAW.glob("*.pdf"):
            pdf_file.unlink()


def get_sources(results):
    seen = set()
    sources = []
    for r in results:
        label = f"{r['source']} (p.{r['page']})"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


# ---------- Session State ----------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores (question, answer, sources)

# ---------- UI ----------

st.title(" Lecture RAG Bot")
st.write("Chat with your lecture PDFs")

# ---------- Sidebar ----------

with st.sidebar:
    st.header(" Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(" Process PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            try:
                clear_raw_pdfs()
                clear_old_processed_data()

                saved = save_uploaded_files(uploaded_files)

                with st.spinner("Processing PDFs..."):
                    pipeline = RAGPipeline()
                    pipeline.ingest_pdfs(DATA_RAW)

                st.session_state.pipeline = pipeline
                st.session_state.processed = True
                st.session_state.chat_history = []

                st.success("PDFs processed successfully!")
                for f in saved:
                    st.write(f"• {f}")

            except Exception as e:
                st.error("Error processing PDFs.")
                st.expander("Details").write(str(e))

# ---------- Load existing data ----------

if st.session_state.pipeline is None and CHUNKS_PATH.exists():
    try:
        pipeline = RAGPipeline()
        pipeline.load_processed_data()
        st.session_state.pipeline = pipeline
        st.session_state.processed = True
    except Exception as e:
        st.error("Failed to load existing data.")
        st.expander("Details").write(str(e))

# ---------- Chat Interface ----------

if st.session_state.processed and st.session_state.pipeline:

    # Display chat history
    for q, a, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant"):
            st.write(a)
            if sources:
                st.caption("Sources: " + ", ".join(sources))

    # User input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        pipeline = st.session_state.pipeline

        # Show user message
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                try:
                    results = pipeline.retrieve(question, top_k=3)
                    sources = get_sources(results)

                    try:
                        answer = pipeline.generate_answer(question, results)
                    except Exception:
                        answer = " Could not generate answer (API issue). Showing relevant content instead."

                    st.write(answer)

                    if sources:
                        st.caption("Sources: " + ", ".join(sources))

                    # Save to history
                    st.session_state.chat_history.append((question, answer, sources))

                except Exception as e:
                    st.error("Something went wrong during retrieval.")
                    st.expander("Details").write(str(e))

else:
    st.info("Upload and process PDFs to start chatting.")