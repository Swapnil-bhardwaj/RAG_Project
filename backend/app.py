import os
import shutil
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from .document_parser import parse_document
from .rag import ingest_text, answer_question

app = FastAPI(title="RAG Q&A (HF + FAISS)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running. Use /upload/ then /ask/ or open /docs."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload/")
async def upload_file(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = parse_document(file_path)
    ingest_text(text, source_name=file.filename)
    return {"message": "File processed successfully", "filename": file.filename}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    result = answer_question(question, k=4)
    return result
