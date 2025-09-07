import faiss
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer

_embedder = None
_dim = 384
_index = faiss.IndexFlatIP(_dim)  # cosine with normalized vectors
_docs: List[str] = []
_meta: List[Dict] = []

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

def embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_embedder()
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

def add_texts(chunks: List[str], metadatas: List[Dict]):
    global _index, _docs, _meta
    if not chunks:
        return
    vecs = embed_texts(chunks)
    _index.add(vecs)
    _docs.extend(chunks)
    _meta.extend(metadatas)

def search(query: str, k: int = 5) -> List[Tuple[int, float]]:
    if _index.ntotal == 0:
        return []
    qv = embed_texts([query])
    D, I = _index.search(qv, k)
    return [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i != -1]

def get_doc(idx: int) -> str:
    return _docs[idx]

def get_meta(idx: int) -> Dict:
    return _meta[idx]
