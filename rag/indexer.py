import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from config import *

model = SentenceTransformer(EMBEDDING_MODEL)

ALLOWED_EXTENSIONS = [".js", ".ts", ".jsx", ".tsx", ".py"]
IGNORE_DIRS = ["node_modules", ".git", "venv", "__pycache__", "dist", "build"]
MAX_CHUNKS = 2000


def chunk_text(text, size=CHUNK_SIZE):
    return [text[i : i + size] for i in range(0, len(text), size)]


def index_codebase():
    documents = []
    metadata = []

    for root, dirs, files in os.walk(CODEBASE_PATH):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if not any(file.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                continue

            path = os.path.join(root, file)

            with open(path, "r", errors="ignore") as f:
                content = f.read()

            chunks = chunk_text(content)

            for chunk in chunks:
                documents.append(chunk)
                metadata.append({"file": path})

                if len(documents) > MAX_CHUNKS:
                    print("⚠️ Chunk limit reached")
                    break
    print(f"📊 Total chunks: {len(documents)}")
    print("⚡ Generating embeddings...")
    embeddings = model.encode(documents)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open("data/meta.pkl", "wb") as f:
        pickle.dump((documents, metadata), f)

    print("✅ Index built successfully")


if __name__ == "__main__":
    index_codebase()
