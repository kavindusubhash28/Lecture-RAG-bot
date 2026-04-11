from src.rag_pipeline import RAGPipeline, DATA_RAW, CHUNKS_PATH, EMBEDDINGS_PATH


def main():
    pipeline = RAGPipeline()

    if CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists():
        print("Loading processed data...")
        pipeline.load_processed_data()
        print("Processed data loaded successfully.\n")
    else:
        print("Processing PDFs...")
        pipeline.ingest_pdfs(DATA_RAW)
        print("PDFs processed successfully.\n")

    while True:
        question = input("Ask a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        results = pipeline.retrieve(question, top_k=3)

        try:
            answer = pipeline.generate_answer(question, results)
            print("\nFinal Answer:\n")
            print(answer)
        except Exception as e:
            print("\nLLM answer generation failed.")
            print(f"Reason: {e}")
            print("\nShowing retrieved chunks instead:\n")

        unique_sources = []
        seen = set()

        for result in results:
            source_label = f"{result['source']} - Page {result['page']}"
            if source_label not in seen:
                seen.add(source_label)
                unique_sources.append(source_label)

        print("\nSources:")
        for source in unique_sources:
            print(f"- {source}")

        print("\nTop retrieved chunks:\n")
        for i, result in enumerate(results, start=1):
            print(
                f"--- Result {i} | Chunk ID: {result['chunk_id']} | "
                f"Source: {result['source']} | Page: {result['page']} | "
                f"Score: {result['score']:.4f} ---"
            )
            print(result["chunk"][:500])
            print()


if __name__ == "__main__":
    main()