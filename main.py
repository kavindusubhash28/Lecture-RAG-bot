from pathlib import Path
from src.rag_pipeline import RAGPipeline, PDF_PATH


def main():
    # Create pipeline instance (loads embedding model)
    pipeline = RAGPipeline()

    # Check if the PDF file exists
    if not Path(PDF_PATH).exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    # Process the PDF (extract - chunk - embed - save)
    print("Processing PDF...")
    pipeline.ingest_pdf(PDF_PATH)
    print("PDF processed successfully.\n")

    # Start question loop
    while True:
        # Get user input
        question = input("Ask a question (or type 'exit'): ").strip()

        # Exit condition
        if question.lower() == "exit":
            break

        # Retrieve top 3 relevant chunks
        results = pipeline.retrieve(question, top_k=3)

        print("\nTop retrieved chunks:\n")

        # Display results
        for i, result in enumerate(results, start=1):
            print(f"--- Result {i} | Score: {result['score']:.4f} ---")

            # Show first 500 characters of chunk
            print(result["chunk"][:500])
            print()


# Run the program
if __name__ == "__main__":
    main()