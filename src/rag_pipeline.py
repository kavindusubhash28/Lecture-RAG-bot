import json
import os
from pathlib import Path

import numpy as np
import pdfplumber
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# Paths for data storage
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

# File locations
PDF_PATH = DATA_RAW / "lecture.pdf"
CHUNKS_PATH = DATA_PROCESSED / "chunks.json"
EMBEDDINGS_PATH = DATA_PROCESSED / "embeddings.npy"


class RAGPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load embedding model
        self.embedding_model = SentenceTransformer(model_name)

        # OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Each chunk will be a dict:
        # {"chunk_id": 0, "page": 1, "text": "..."}
        self.chunks = []
        self.embeddings = None

    def extract_pages_from_pdf(self, pdf_path: Path):
        """Extract text page by page from a PDF"""
        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "page": page_number,
                        "text": text
                    })

        return pages

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100):
        """Split text into smaller chunks with overlap"""
        words = text.split()
        chunks = []

        start = 0
        step = chunk_size - overlap

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])

            if chunk.strip():
                chunks.append(chunk)

            start += step

        return chunks

    def create_chunk_records(self, pages, chunk_size: int = 500, overlap: int = 100):
        """Create chunk records with chunk_id, page number, and text"""
        chunk_records = []
        chunk_id = 0

        for page_data in pages:
            page_number = page_data["page"]
            page_text = page_data["text"]

            page_chunks = self.chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)

            for chunk in page_chunks:
                chunk_records.append({
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk
                })
                chunk_id += 1

        return chunk_records

    def create_embeddings(self, chunks):
        """Convert chunk texts into vector embeddings"""
        chunk_texts = [chunk["text"] for chunk in chunks]
        return self.embedding_model.encode(chunk_texts, convert_to_numpy=True)

    def save_processed_data(self):
        """Save chunks and embeddings to disk"""
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        np.save(EMBEDDINGS_PATH, self.embeddings)

    def load_processed_data(self):
        """Load chunks and embeddings from disk"""
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.embeddings = np.load(EMBEDDINGS_PATH)

    def cosine_similarity(self, query_embedding, doc_embeddings):
        """Compute cosine similarity between query and all chunk embeddings"""
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)

        similarities = np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm + 1e-10)
        return similarities

    def retrieve(self, question: str, top_k: int = 3):
        """Find top-k most relevant chunks for a question"""
        query_embedding = self.embedding_model.encode(question, convert_to_numpy=True)
        similarities = self.cosine_similarity(query_embedding, self.embeddings)

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "page": self.chunks[idx]["page"],
                "chunk": self.chunks[idx]["text"],
                "score": float(similarities[idx])
            })

        return results

    def generate_answer(self, question: str, retrieved_chunks):
        """Generate final answer using retrieved chunks as context"""

        # Combine retrieved chunks into one context block
        context = "\n\n".join(
            [f"(Page {chunk['page']}) {chunk['chunk']}" for chunk in retrieved_chunks]
        )

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"I could not find the answer in the lecture notes."
"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions only from the provided lecture context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    def ingest_pdf(self, pdf_path: Path):
        """Full pipeline: PDF -> pages -> chunk records -> embeddings -> save"""
        pages = self.extract_pages_from_pdf(pdf_path)
        self.chunks = self.create_chunk_records(pages)
        self.embeddings = self.create_embeddings(self.chunks)
        self.save_processed_data()