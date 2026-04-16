import shutil
from pathlib import Path

import streamlit as st

from src.rag_pipeline import (
    RAGPipeline,
    DATA_RAW,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
)

st.set_page_config(page_title="Lecture RAG Bot", page_icon="📘", layout="wide")


def save_uploaded_files(uploaded_files):
    """Save uploaded PDFs into data/raw"""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = DATA_RAW / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path.name)

    return saved_files


def clear_old_processed_data():
    """Delete old processed files so new PDFs can be reprocessed"""
    if CHUNKS_PATH.exists():
        CHUNKS_PATH.unlink()

    if EMBEDDINGS_PATH.exists():
        EMBEDDINGS_PATH.unlink()


def clear_raw_pdfs():
    """Delete old PDFs from data/raw"""
    if DATA_RAW.exists():
        for pdf_file in DATA_RAW.glob("*.pdf"):
            pdf_file.unlink()


def get_unique_sources(results):
    """Extract unique sources from retrieved chunks"""
    unique_sources = []
    seen = set()

    for result in results:
        source_label = f"{result['source']} - Page {result['page']}"
        if source_label not in seen:
            seen.add(source_label)
            unique_sources.append(source_label)

    return unique_sources


st.title("📘 Lecture RAG Bot")
st.write("Upload lecture PDFs, ask questions, and get answers grounded in your documents.")

# Initialize pipeline once
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "results" not in st.session_state:
    st.session_state.results = None

if "answer" not in st.session_state:
    st.session_state.answer = None

# Sidebar for upload and processing
with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            try:
                clear_raw_pdfs()
                clear_old_processed_data()

                saved_files = save_uploaded_files(uploaded_files)

                pipeline = RAGPipeline()
                pipeline.ingest_pdfs(DATA_RAW)

                st.session_state.pipeline = pipeline
                st.session_state.processed = True

                st.success("PDFs processed successfully.")
                st.write("Uploaded files:")
                for file_name in saved_files:
                    st.write(f"- {file_name}")

            except Exception as e:
                st.error(f"Error processing PDFs: {e}")

# Load existing processed data if available
if st.session_state.pipeline is None and CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists():
    try:
        pipeline = RAGPipeline()
        pipeline.load_processed_data()
        st.session_state.pipeline = pipeline
        st.session_state.processed = True
    except Exception as e:
        st.error(f"Error loading processed data: {e}")

# Main question interface
if st.session_state.processed and st.session_state.pipeline is not None:
    st.subheader("Ask a Question")

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            pipeline = st.session_state.pipeline

            try:
                results = pipeline.retrieve(question, top_k=3)
                st.session_state.results = results

                try:
                    answer = pipeline.generate_answer(question, results)
                    st.session_state.answer = answer
                except Exception as e:
                    st.session_state.answer = None
                    st.warning(f"LLM answer generation failed: {e}")

            except Exception as e:
                st.error(f"Error during retrieval: {e}")

    # Show answer
    if st.session_state.answer:
        st.subheader("Final Answer")
        st.write(st.session_state.answer)

    # Show sources
    if st.session_state.results:
        st.subheader("Sources")
        sources = get_unique_sources(st.session_state.results)
        for source in sources:
            st.write(f"- {source}")

        with st.expander("Show Retrieved Chunks"):
            for i, result in enumerate(st.session_state.results, start=1):
                st.markdown(
                    f"**Result {i}** | "
                    f"Chunk ID: `{result['chunk_id']}` | "
                    f"Source: `{result['source']}` | "
                    f"Page: `{result['page']}` | "
                    f"Score: `{result['score']:.4f}`"
                )
                st.write(result["chunk"])
                st.markdown("---")
else:
    st.info("Upload and process PDFs first to start asking questions.")