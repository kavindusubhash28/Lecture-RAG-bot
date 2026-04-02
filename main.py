from pathlib import Path
from src.rag_pipeline import (
    RAGPipeline,
    PDF_PATH,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
)


def main():
    pipeline = RAGPipeline()

    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    if CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists():
        print("Loading processed data...")
        pipeline.load_processed_data()
        print("Processed data loaded successfully.\n")
    else:
        print("Processing PDF...")
        pipeline.ingest_pdf(PDF_PATH)
        print("PDF processed successfully.\n")

    while True:
        question = input("Ask a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        # Step 1: retrieve relevant chunks
        results = pipeline.retrieve(question, top_k=3)

        # Step 2: generate final answer
        answer = pipeline.generate_answer(question, results)

        print("\nFinal Answer:\n")
        print(answer)

        print("\nTop retrieved chunks:\n")
        for i, result in enumerate(results, start=1):
            print(
                f"--- Result {i} | Chunk ID: {result['chunk_id']} | "
                f"Page: {result['page']} | Score: {result['score']:.4f} ---"
            )
            print(result["chunk"][:500])
            print()


if __name__ == "__main__":
    main()