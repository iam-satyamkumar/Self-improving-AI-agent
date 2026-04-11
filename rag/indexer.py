import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from config import *

model = SentenceTransformer(EMBEDDING_MODEL)

ALLOWED_EXTENSIONS = [".js", ".ts", ".jsx", ".tsx", ".py"]
IGNORE_DIRS = ["node_modules", ".git", "venv", "__pycache__", "dist", "build"]
MAX_CHUNKS = 2000


import re


def chunk_code(content):
    lines = content.split("\n")

    chunks = []
    buffer = []
    brace_count = 0

    for line in lines:
        stripped = line.strip()

        # detect function start
        if (
            "function " in stripped
            or "=>" in stripped
            or stripped.startswith("def ")
            or stripped.startswith("class ")
        ):
            if buffer:
                chunks.append("\n".join(buffer))
                buffer = []

            brace_count = 0

        buffer.append(line)

        # track braces (for JS/TS)
        brace_count += line.count("{")
        brace_count -= line.count("}")

        # if function block ends
        if brace_count == 0 and buffer:
            chunks.append("\n".join(buffer))
            buffer = []

    if buffer:
        chunks.append("\n".join(buffer))

    return chunks


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

            chunks = chunk_code(content)

            for chunk in chunks:
                documents.append(chunk)
                metadata.append({"file": path, "preview": chunk[:100]})

                if len(documents) > MAX_CHUNKS:
                    print("⚠️ Chunk limit reached")
                    break

            if len(documents) > MAX_CHUNKS:
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
