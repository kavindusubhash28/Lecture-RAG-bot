from pathlib import Path
import streamlit as st

from src.rag_pipeline import (
    RAGPipeline,
    DATA_RAW,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
)
import base64
import html
from streamlit.components.v1 import html as components_html

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


def make_pdf_viewer_html(pdf_bytes: bytes, page: int, highlight: str) -> str:
        b64 = base64.b64encode(pdf_bytes).decode('utf-8')
        safe_highlight = html.escape(highlight or '')
        # PDF.js from CDN, simple single-page renderer with text-layer search+highlight
        return f"""
<!doctype html>
<html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
            body {{ margin: 0; }}
            #viewer {{ width: 100%; height: 100%; overflow: auto; background: #888; }}
            .textLayer span.highlight {{ background: yellow; }}
        </style>
    </head>
    <body>
        <div id="viewer"><canvas id="the-canvas"></canvas><div id="text-layer" class="textLayer"></div></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.min.js"></script>
        <script>
            const pdfData = atob('{b64}');
            const pdfBytes = new Uint8Array(pdfData.length);
            for (let i = 0; i < pdfData.length; i++) pdfBytes[i] = pdfData.charCodeAt(i);
            const loadingTask = pdfjsLib.getDocument({data: pdfBytes});
            loadingTask.promise.then(async function(pdf) {{
                const pageNum = {page};
                const pdfPage = await pdf.getPage(pageNum);
                const viewport = pdfPage.getViewport({scale: 1.5});
                const canvas = document.getElementById('the-canvas');
                const context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;
                const renderContext = {{ canvasContext: context, viewport: viewport }};
                await pdfPage.render(renderContext).promise;

                // render text layer
                const textContent = await pdfPage.getTextContent();
                const textLayer = document.getElementById('text-layer');
                textLayer.style.position = 'absolute';
                textLayer.style.left = '0';
                textLayer.style.top = '0';
                textLayer.style.height = canvas.height + 'px';
                textLayer.style.width = canvas.width + 'px';

                // create spans for items
                const frag = document.createDocumentFragment();
                textContent.items.forEach(function(item, idx) {{
                    const span = document.createElement('span');
                    span.textContent = item.str + ' ';
                    // basic positioning approximation
                    span.style.whiteSpace = 'pre';
                    frag.appendChild(span);
                }});
                textLayer.appendChild(frag);

                // simple highlight by searching text content
                const query = `{safe_highlight}`.trim();
                if (query) {{
                    const text = textLayer.textContent;
                    const idx = text.toLowerCase().indexOf(query.toLowerCase());
                    if (idx !== -1) {{
                        // wrap matched substring in span.highlight
                        const full = textLayer.textContent;
                        const before = full.slice(0, idx);
                        const match = full.slice(idx, idx + query.length);
                        const after = full.slice(idx + query.length);
                        textLayer.innerHTML = '';
                        const b = document.createElement('span'); b.textContent = before;
                        const m = document.createElement('span'); m.textContent = match; m.className = 'highlight';
                        const a = document.createElement('span'); a.textContent = after;
                        textLayer.appendChild(b); textLayer.appendChild(m); textLayer.appendChild(a);
                    }}
                }}
            }}, function(reason) {{
                document.body.innerText = 'Error loading PDF: ' + reason;
            }});
        </script>
    </body>
</html>
"""



# ---------- Session State ----------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores (question, answer, sources_list)

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
    for qi, (q, a, sources) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant"):
            st.write(a)
            if sources:
                st.markdown("**Sources:**")
                for idx, item in enumerate(sources):
                    # item is expected to be a dict with label, source, page, text
                    label = item['label'] if isinstance(item, dict) else item
                    key = f"hist_src_{qi}_{idx}_{label}"
                    if st.button(label, key=key):
                        try:
                            if isinstance(item, dict):
                                pdf_path = DATA_RAW / item['source']
                                page = int(item.get('page', 1))
                                text = item.get('text', '')
                            else:
                                # fallback: try parse label
                                pdf_path = DATA_RAW / label.split(' (p.')[0]
                                page = 1
                                text = ''

                            if not pdf_path.exists():
                                st.error(f"PDF not found: {pdf_path}")
                            else:
                                pdf_bytes = pdf_path.read_bytes()
                                viewer = make_pdf_viewer_html(pdf_bytes, page, text)
                                components_html(viewer, height=800)
                        except Exception as e:
                            st.error('Failed to open PDF viewer.')
                            st.expander('Details').write(str(e))

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
                    # build unique source entries preserving order
                    seen = set()
                    source_items = []  # list of dicts {label, source, page, text}
                    for r in results:
                        label = f"{r['source']} (p.{r['page']})"
                        if label in seen:
                            continue
                        seen.add(label)
                        source_items.append({
                            'label': label,
                            'source': r.get('source'),
                            'page': r.get('page', 1),
                            'text': r.get('text', '')
                        })

                    try:
                        answer = pipeline.generate_answer(question, results)
                    except Exception as e:
                        answer = f"Could not generate answer. Error: {str(e)}"

                    st.write(answer)

                    if source_items:
                        st.markdown("**Sources:**")
                        for item in source_items:
                            # use a unique key to avoid Streamlit key collisions
                            if st.button(item['label'], key=f"src_{item['label']}"):
                                try:
                                    pdf_path = DATA_RAW / item['source']
                                    if not pdf_path.exists():
                                        st.error(f"PDF not found: {pdf_path}")
                                    else:
                                        pdf_bytes = pdf_path.read_bytes()
                                        viewer = make_pdf_viewer_html(pdf_bytes, int(item['page']), item.get('text',''))
                                        components_html(viewer, height=800)
                                except Exception as e:
                                    st.error('Failed to open PDF viewer.')
                                    st.expander('Details').write(str(e))

                    # Save to history
                    st.session_state.chat_history.append((question, answer, source_items))

                except Exception as e:
                    st.error("Something went wrong during retrieval.")
                    st.expander("Details").write(str(e))

else:
    st.info("Upload and process PDFs to start chatting.")