import faiss
import pickle
from sentence_transformers import SentenceTransformer
from memory.store import load_memory

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def retrieve_memory(query, k=3):
    index, documents, metadata = load_memory()

    if index is None:
        return []

    q_emb = model.encode([query])

    D, I = index.search(q_emb, k)

    results = []
    for idx in I[0]:
        results.append(metadata[idx])

    return results
