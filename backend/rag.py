from typing import List, Tuple
from .embeddings import add_texts, search, get_doc, get_meta
from transformers import pipeline

_gen = None

def _get_generator():
    global _gen
    if _gen is None:
        _gen = pipeline("text2text-generation", model="google/flan-t5-base")
    return _gen

def chunk_text(text: str, max_words: int = 180) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def ingest_text(full_text: str, source_name: str = "document"):
    chunks = chunk_text(full_text)
    metas = [{"source": source_name, "chunk_id": i} for i in range(len(chunks))]
    add_texts(chunks, metas)

def retrieve_context(question: str, k: int = 4) -> List[Tuple[str, float, dict]]:
    results = search(question, k=k)
    out = []
    for idx, score in results:
        out.append((get_doc(idx), score, get_meta(idx)))
    return out

def answer_question(question: str, k: int = 4, max_new_tokens: int = 256) -> dict:
    ctx = retrieve_context(question, k=k)
    context_text = "\n\n".join([f"[{i+1}] {c[0]}" for i, c in enumerate(ctx)])
    prompt = (
        "You are a helpful assistant. Use ONLY the context to answer.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    generator = _get_generator()
    out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    answer = out[0]["generated_text"].strip()
    citations = [
        {"source": c[2].get("source"), "chunk_id": c[2].get("chunk_id"), "score": c[1]}
        for c in ctx
    ]
    return {"answer": answer, "citations": citations}
