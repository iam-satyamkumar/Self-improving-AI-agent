import faiss
import pickle
from sentence_transformers import SentenceTransformer
from config import *

model = SentenceTransformer(EMBEDDING_MODEL)

index = faiss.read_index(FAISS_INDEX_PATH)

with open("data/meta.pkl", "rb") as f:
    documents, metadata = pickle.load(f)


def retrieve(query, k=5):
    q_emb = model.encode([query])
    D, I = index.search(q_emb, k)

    results = []
    for idx in I[0]:
        results.append({"text": documents[idx], "file": metadata[idx]["file"]})

    return results
