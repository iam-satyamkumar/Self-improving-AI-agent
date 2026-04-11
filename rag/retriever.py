import faiss
import pickle
from sentence_transformers import SentenceTransformer
from config import *

model = SentenceTransformer(EMBEDDING_MODEL)

index = faiss.read_index(FAISS_INDEX_PATH)

with open("data/meta.pkl", "rb") as f:
    documents, metadata = pickle.load(f)


def is_useful_chunk(text, query):
    """
    Check if chunk is relevant to query using keywords
    """
    query_words = query.lower().split()

    # basic keyword overlap
    return any(word in text.lower() for word in query_words)


def retrieve(query, k=20):
    q_emb = model.encode([query])

    D, I = index.search(q_emb, k)

    results = []
    seen = set()

    for dist, idx in zip(D[0], I[0]):
        chunk = documents[idx]
        file = metadata[idx]["file"]

        # 🔥 1. Skip very weak matches
        if dist > 1.5:  # you can tune this
            continue

        # 🔥 2. Skip tiny/useless chunks
        if len(chunk.strip()) < 30:
            continue

        # 🔥 3. Keyword relevance check
        if not is_useful_chunk(chunk, query):
            continue

        # 🔥 4. Deduplicate
        if chunk in seen:
            continue
        seen.add(chunk)

        results.append({"text": chunk, "file": file, "score": float(dist)})

    # 🔥 fallback if nothing found
    if not results:
        for idx in I[0][:3]:
            results.append(
                {"text": documents[idx], "file": metadata[idx]["file"], "score": None}
            )

    return results[:5]
