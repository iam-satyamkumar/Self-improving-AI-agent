import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

MEMORY_INDEX_PATH = "memory/memory.faiss"
MEMORY_META_PATH = "memory/meta.pkl"


def load_memory():
    if not os.path.exists(MEMORY_INDEX_PATH):
        return None, [], []

    index = faiss.read_index(MEMORY_INDEX_PATH)

    with open(MEMORY_META_PATH, "rb") as f:
        data = pickle.load(f)

    return index, data["documents"], data["metadata"]


def save_memory(index, documents, metadata):
    faiss.write_index(index, MEMORY_INDEX_PATH)

    with open(MEMORY_META_PATH, "wb") as f:
        pickle.dump({"documents": documents, "metadata": metadata}, f)


def add_memory(entry):
    text = f"""
Query: {entry['query']}
Mistake: {entry['bad_answer']}
Correction: {entry['correct_answer']}
Reason: {entry['reason']}
"""

    emb = model.encode([text])

    index, documents, metadata = load_memory()

    if index is None:
        index = faiss.IndexFlatL2(len(emb[0]))
        documents = []
        metadata = []

    index.add(emb)

    documents.append(text)
    metadata.append(entry)

    save_memory(index, documents, metadata)

    print("✅ Memory stored")
