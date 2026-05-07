# Lecture RAG Bot

This project is a retrieval-augmented generation app for lecture PDFs.

## How it works

1. Put PDF files in `data/raw/` or upload them from the Streamlit app.
2. The pipeline extracts text page by page with `pdfplumber`.
3. Text is split into overlapping chunks.
4. Chunks are embedded with `sentence-transformers` using `all-MiniLM-L6-v2`.
5. Embeddings are stored in `data/processed/` so later runs can reuse them.
6. When you ask a question, the app retrieves the most relevant chunks.
7. Gemini generates the final answer from the retrieved context.

## Gemini setup

Create a `.env` file in the project root with:

```env
GEMINI_API_KEY=your_api_key_here
```

The app reads `GEMINI_API_KEY`.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

If you want the command-line version instead, run:

```bash
python main.py
```
