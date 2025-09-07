# RAG Q&A System (Hugging Face + FAISS) — No API Keys

An offline-friendly Retrieval-Augmented Generation (RAG) app.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (dim=384)
- Vector DB: FAISS (cosine similarity)
- Generator: `google/flan-t5-base` via `transformers` pipeline
- Backend: FastAPI
- Frontend: Streamlit
- Single-document upload (PDF / Markdown / HTML)

## Windows Quick Start

1) **Extract** this zip and open PowerShell in the folder.
2) **Create venv & install deps**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3) **Run backend** (keep this window open)
```powershell
uvicorn backend.app:app --reload
```
Backend docs: http://127.0.0.1:8000/docs

4) **Run frontend** in a **new** PowerShell window
```powershell
venv\Scripts\activate
streamlit run frontend/app.py
```
Frontend: http://localhost:8501

> On first run, `transformers` will download models automatically (few hundred MB).

## Repo Structure
```
backend/
  app.py                # FastAPI: upload + ask
  __init__.py
  document_parser.py    # PDF/MD/HTML to clean text
  embeddings.py         # HF embeddings + FAISS index
  rag.py                # retrieval + generation
frontend/
  app.py                # Streamlit UI
requirements.txt
```

## Notes
- This version avoids OpenAI completely: no keys, no quotas.
- If you want to reset the index, just restart the backend (in-memory FAISS).
- For production, persist FAISS to disk and add multi-doc support.
