import json
from pathlib import Path

import numpy as np
import pdfplumber
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

        # Store chunks and embeddings
        self.chunks = []
        self.embeddings = None

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract all text from a PDF file"""
        all_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)

        # Combine all pages into one string
        return "\n".join(all_text)

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100):
        """Split text into smaller chunks with overlap"""
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])

            if chunk.strip():
                chunks.append(chunk)

            # Move forward with overlap
            start += chunk_size - overlap

        return chunks

    def create_embeddings(self, chunks):
        """Convert text chunks into vector embeddings"""
        return self.embedding_model.encode(chunks, convert_to_numpy=True)

    def save_processed_data(self):
        """Save chunks and embeddings to disk"""
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

        # Save chunks as JSON
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # Save embeddings as numpy file
        np.save(EMBEDDINGS_PATH, self.embeddings)

    def load_processed_data(self):
        """Load chunks and embeddings from disk"""
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.embeddings = np.load(EMBEDDINGS_PATH)

    def cosine_similarity(self, query_embedding, doc_embeddings):
        """Compute cosine similarity between query and all chunks"""
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)

        similarities = np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm + 1e-10)
        return similarities

    def retrieve(self, question: str, top_k: int = 3):
        """Find top-k most relevant chunks for a question"""
        # Convert question to embedding
        query_embedding = self.embedding_model.encode(question, convert_to_numpy=True)

        # Compute similarity scores
        similarities = self.cosine_similarity(query_embedding, self.embeddings)

        # Get indices of top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Collect results
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(similarities[idx])
            })

        return results

    def ingest_pdf(self, pdf_path: Path):
        """Full pipeline: PDF → chunks → embeddings → save"""
        text = self.extract_text_from_pdf(pdf_path)
        self.chunks = self.chunk_text(text)
        self.embeddings = self.create_embeddings(self.chunks)

        # Save processed data
        self.save_processed_data()